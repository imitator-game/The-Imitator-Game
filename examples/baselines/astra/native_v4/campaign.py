"""Explicitly launched native 200-case campaign; timeout is terminal and does not stop admissions."""
import argparse
import concurrent.futures
import json
import signal
from pathlib import Path
import subprocess
import sys
import threading
from budget_runner import PACKAGE, read, save, utc, validate_protocol
from model_profiles import PROFILES
from report import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=list(PROFILES), default='sol')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--gpus', default='1')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.protocol = args.protocol or PACKAGE / f'protocol_200_{args.profile}.json'
    if args.execute and args.dry_run:
        parser.error('Choose execution or dry-run')
    protocol = read(args.protocol)
    cases = validate_protocol(protocol)
    if protocol.get('model_profile', 'sol') != args.profile:
        parser.error('Profile and protocol differ')
    gpus = args.gpus.split(',')
    if not all(g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        parser.error('Supply distinct GPU indices')
    if not args.execute:
        print(json.dumps(dict(cases=len(cases), tasks=10, levels=4, seeds=list(range(5)),
             seen=100, unseen=100, planning_time_limit_seconds=1800, wall_time_cap_seconds=None, model=protocol['model'],
             effort=protocol['reasoning_effort'], gpu_workers=gpus, launches_model=False)))
        return 0
    root = PACKAGE / 'campaigns' / args.profile
    # Never silently mix retries or reuse earlier diagnostic trials.
    root.mkdir(parents=True, exist_ok=False)
    lock = threading.Lock()
    stop = threading.Event()
    iterator = iter(cases)
    active = {}
    state = dict(status='running', started_utc=utc(), protocol=str(args.protocol.resolve()),
                 gpu_workers=gpus, launched=[])
    save(root / 'campaign_state.json', state)

    def interrupt(signum, frame):
        stop.set()
        # Each budget supervisor owns and cleans up its native/model children.
        for proc in list(active.values()):
            if proc.poll() is None:
                proc.terminate()
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def worker(gpu):
        while not stop.is_set():
            with lock:
                if stop.is_set() or (root / 'STOP_ADMISSIONS.json').exists():
                    return
                case = next(iterator, None)
                if case is None:
                    return
                state['launched'].append(case['case_id'])
                save(root / 'campaign_state.json', state)
            command = [sys.executable, '-u', str(PACKAGE / 'budget_runner.py'),
                       '--profile', args.profile,
                       '--case-id', case['case_id'], '--gpu', gpu,
                       '--protocol', str(args.protocol.resolve()), '--results-root', str(root)]
            with (root / (case['run_id'] + '.supervisor.log')).open('x') as log:
                with lock:
                    if stop.is_set():
                        return
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    active[gpu] = proc
                proc.wait()
            result = read(root / case['run_id'] / 'result.json')
            with lock:
                active.pop(gpu, None)
                if result.get('status') not in ['complete', 'time_exhausted'] or proc.returncode not in [0, 124]:
                    stop.set()
                    save(root / 'STOP_ADMISSIONS.json', dict(case_id=case['case_id'], exit_code=proc.returncode,
                         reason='infrastructure_failure', result_status=result.get('status')))
                save(root / 'summary.json', summarize(protocol, root))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))
    state.update(status='stopped_after_infrastructure_failure' if stop.is_set() else 'finished', ended_utc=utc())
    save(root / 'campaign_state.json', state)
    save(root / 'summary.json', summarize(protocol, root))
    return 2 if stop.is_set() else 0


if __name__ == '__main__':
    raise SystemExit(main())
