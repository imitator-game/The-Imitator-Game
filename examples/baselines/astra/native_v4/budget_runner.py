"""Supervise cumulative planning time and preserve authoritative native success.

Simulation, host tools, initialization and post-success accounting are not charged
to the planning budget. A native success is never erased by runtime bookkeeping.
"""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from model_profiles import PROFILES, get_profile
from planning_budget import PLANNING_LIMIT_SECONDS, used_seconds
from grading import grade_native_run

PACKAGE = Path(__file__).resolve().parent
VERSION = 'video200_native_success_planning30_v4'
CAP_SECONDS = PLANNING_LIMIT_SECONDS


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_protocol(protocol):
    cases = protocol.get('cases', [])
    if len(cases) != 200 or protocol.get('seeds') != list(range(5)):
        raise ValueError('Expected 200 cases and environment seeds 0..4')
    if protocol.get('planning_time_limit_seconds') != CAP_SECONDS or protocol.get('wall_time_cap_seconds') is not None:
        raise ValueError('This version requires a cumulative 1800-second planning limit, not a wall limit')
    profile = get_profile(protocol.get('model_profile', 'sol'))
    if (protocol.get('model'), protocol.get('reasoning_effort')) != (profile['model'], profile['effort']):
        raise ValueError('Protocol does not match its explicit model binding')
    groups = collections.defaultdict(list)
    for case in cases:
        if case['conditioning'] != 'video' or case.get('demonstration_episode') != 0:
            raise ValueError('The native fixed-ep0 video conditioning must be preserved')
        if case['max_episode_steps'] != 500:
            raise ValueError('The physical step budget must remain 500')
        groups[(case['task_id'], case['level'])].append(case['seed'])
    if len(groups) != 40 or any(sorted(seeds) != list(range(5)) for seeds in groups.values()):
        raise ValueError('Every task/level must contain exactly five distinct seeds')
    if len({c['task_id'] for c in cases}) != 10 or {c['level'] for c in cases} != {'L0', 'L1', 'L2', 'L3'}:
        raise ValueError('Expected ten tasks and L0-L3')
    if collections.Counter(c['split'] for c in cases) != {'seen': 100, 'unseen': 100}:
        raise ValueError('Expected seen-100 and unseen-100')
    for key in ['case_id', 'run_id']:
        if len({c[key] for c in cases}) != 200:
            raise ValueError(f'Duplicate {key}')
    return cases


def process_identity(pid):
    """Linux process identity, including start ticks to guard against PID reuse."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return (int(pid), int(fields[19]), int(fields[2]), fields[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def descendants(pid):
    rows = subprocess.run(['ps', '-eo', 'pid=,ppid=,pgid='], text=True,
                          capture_output=True, check=True).stdout.splitlines()
    table = [tuple(map(int, line.split())) for line in rows if len(line.split()) == 3]
    found = {pid}
    while True:
        expanded = found | {child for child, parent, _ in table if parent in found}
        if expanded == found:
            break
        found = expanded
    return found


def remember_worker(native_output, proc, known):
    pid = read(native_output / 'controller/worker.json').get('pid')
    if pid and pid not in known and pid in descendants(proc.pid):
        identity = process_identity(pid)
        if identity:
            known[pid] = identity


def signal_group(identity, sig):
    if not identity:
        return
    pid, born, group, _ = identity
    current = process_identity(pid)
    if current and current[:3] == (pid, born, group) and group == pid:
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            pass


def stop_owned(proc, known, grace=20):
    # Capture separate app-server sessions before the runner can reparent them.
    if proc.poll() is None:
        for pid in descendants(proc.pid):
            identity = process_identity(pid)
            if identity and pid != proc.pid and identity[2] == pid:
                known[pid] = identity
    root_identity = process_identity(proc.pid)
    for identity in known.values():
        signal_group(identity, signal.SIGTERM)
    if proc.poll() is None:
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # Let native finally blocks flush trajectory, outcome, and video files.
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        signal_group(root_identity, signal.SIGKILL)
        proc.wait(timeout=5)
    for identity in known.values():
        signal_group(identity, signal.SIGKILL)
    # kill(2) queues a signal; do not report cleanup finished while the model can
    # still be scheduled. Orphan zombies cannot issue requests and await init reap.
    confirm_deadline = time.monotonic() + 2
    while True:
        live = [identity for identity in known.values()
                if (current := process_identity(identity[0]))
                and current[:3] == identity[:3] and current[3] != 'Z']
        if not live:
            return
        if time.monotonic() >= confirm_deadline:
            raise RuntimeError('An owned model process survived SIGKILL')
        time.sleep(.01)


def supervise(command, *, native_output, result_dir, cwd, env, cap_seconds=CAP_SECONDS,
              cleanup_seconds=600, stop_grace=20, poll_seconds=.1, profile=None,
              setup_timeout=300, host_timeout=600):
    result_dir.mkdir(parents=True, exist_ok=False)
    profile = profile or get_profile('sol')
    began = time.monotonic()
    known = {}
    stop_reason = None
    planning = {}
    planner_cutoff = None
    with (result_dir/'policy.log').open('x') as log:
        proc = subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(result_dir/'launch.json',dict(started_utc=utc(),pid=proc.pid,
             planning_time_limit_seconds=cap_seconds,wall_time_cap_seconds=None,
             native_output=str(native_output)))
        last_save=0.
        try:
            while True:
                now=time.monotonic();elapsed=now-began
                remember_worker(native_output,proc,known)
                planning=read(native_output/'planning_budget.json')
                progress=read(native_output/'progress.json')
                goal=read(native_output/'native_success_stop.json')
                consumed=used_seconds(planning,now) if planning else 0.
                if planning and float(planning.get('limit_seconds',-1))!=float(cap_seconds):
                    stop_reason='planning_budget_mismatch';break
                if goal or progress.get('finished'):
                    # Native goal/step termination must not leave an API worker alive.
                    for identity in known.values():signal_group(identity,signal.SIGTERM)
                if not goal and planning and (planning.get('exhausted') or consumed>=cap_seconds):
                    stop_reason='planning_time_exhausted';break
                if proc.poll() is not None:break
                if not planning and elapsed>setup_timeout:
                    stop_reason='initialization_timeout';break
                if planning and planning.get('active_since_monotonic') is None:
                    idle=now-float(planning.get('updated_monotonic',now))
                    limit=cleanup_seconds if planning.get('closed') else host_timeout
                    if idle>limit:
                        stop_reason='host_or_artifact_timeout';break
                if elapsed-last_save>=1:
                    save(result_dir/'progress.json',dict(updated_utc=utc(),wall_elapsed_seconds=elapsed,
                         planning_seconds=consumed,planning_remaining_seconds=max(0,cap_seconds-consumed),
                         planning_phase=planning.get('phase','initializing'),native=progress,
                         native_success_observed=bool(goal)))
                    last_save=elapsed
                time.sleep(poll_seconds)
        except BaseException:
            stop_reason='supervisor_interrupted'
            raise
        finally:
            planner_cutoff=time.monotonic()
            planning_at_cancellation=read(native_output/'planning_budget.json')
            if stop_reason:
                save(result_dir/'stop_event.json',dict(reason=stop_reason,utc=utc(),
                     wall_elapsed_seconds=time.monotonic()-began,
                     planning_seconds=used_seconds(planning) if planning else 0.,
                     planning_time_limit_seconds=cap_seconds,progress=read(native_output/'progress.json')))
                # Cancel API work first; let the native alarm/finally blocks persist evidence.
                for identity in known.values():signal_group(identity,signal.SIGTERM)
                try:proc.wait(timeout=stop_grace)
                except subprocess.TimeoutExpired:pass
            stop_owned(proc,known,grace=stop_grace)
            planning=read(native_output/'planning_budget.json')
            if planning and planning.get('active_since_monotonic') is not None:
                # Forced process termination must not keep accruing time during cleanup.
                save(result_dir/'planning_snapshot_before_cleanup.json',planning_at_cancellation)
                planning.update(used_seconds=used_seconds(planning,planner_cutoff),
                    active_since_monotonic=None,closed=True,closure_source='supervisor',
                    stop_reason=stop_reason or 'native_process_exit')
                save(native_output/'planning_budget.json',planning)
            expired=stop_reason=='planning_time_exhausted' or bool(planning.get('exhausted'))
            result=grade_native_run(native_output,proc.returncode,expired)
            result.update(protocol_version=VERSION,model=profile['model'],reasoning_effort=profile['effort'],
                model_profile=profile['name'],elapsed_seconds=time.monotonic()-began,
                planning_seconds=used_seconds(planning) if planning else None,
                planning_time_limit_seconds=cap_seconds,wall_time_cap_seconds=None,
                stop_reason=stop_reason or planning.get('stop_reason'),exit_code=proc.returncode,
                native_output=str(native_output),planning_budget=planning)
            save(result_dir/'result.json',result)
            if native_output.exists():save(native_output/'budget_outcome.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--profile', choices=list(PROFILES), default='sol')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--results-root', type=Path, default=PACKAGE / 'budget_results')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    profile = get_profile(args.profile)
    args.protocol = args.protocol or PACKAGE / f'protocol_200_{args.profile}.json'
    protocol = read(args.protocol)
    cases = validate_protocol(protocol)
    if protocol.get('model_profile', 'sol') != args.profile:
        parser.error('Selected profile and protocol differ')
    case = next((c for c in cases if c['case_id'] == args.case_id), None)
    if case is None:
        parser.error('Case is not in the frozen 200-case protocol')
    run = args.run_id or case['run_id']
    if not run.replace('_', '').replace('-', '').isalnum():
        parser.error('Invalid run identifier')
    native_output = PACKAGE / 'runs' / run
    result_dir = args.results_root / run
    if native_output.exists() or result_dir.exists():
        parser.error('Run already exists; no implicit retry or overwrite')
    if args.dry_run:
        print(json.dumps(dict(case=case, run_id=run, planning_time_limit_seconds=CAP_SECONDS, wall_time_cap_seconds=None,
             model=protocol['model'], reasoning_effort=protocol['reasoning_effort'], profile=args.profile, gpu=args.gpu,
             native_output=str(native_output), result_dir=str(result_dir), launches_model=False)))
        return 0
    root = next(parent for parent in PACKAGE.parents if (parent / 'run_native.py').exists())
    sys.path.insert(0, str(root))
    from run_native import PYTHON, child_environment
    # Native runner itself also checks all hashes before creating the model session.
    for path, expected in protocol.get('budget_source_sha256', {}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Budget source drift: ' + path)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    env = child_environment()
    credential = read(root.parent / '.config/gpt-baseline' / profile['credential'])
    if credential.get('base_url', '').rstrip('/') != 'https://xfast.transcengram.com/v1':
        raise RuntimeError('Model credential is not for the approved xfast endpoint')
    env['XFAST_API_KEY'] = credential['api_key']
    command = [PYTHON, '-B', '-u', str(PACKAGE / 'run_episode.py'), '--case-id', args.case_id,
               '--profile', args.profile,
               '--run-id', run, '--gpu', args.gpu, '--protocol', str(args.protocol.resolve())]
    result = supervise(command, native_output=native_output, result_dir=result_dir,
                       cwd=root / 'workspace/The-Imitator-Game', env=env, profile=profile)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] == 'complete' else 124 if result['status'] == 'time_exhausted' else 2


if __name__ == '__main__':
    raise SystemExit(main())
