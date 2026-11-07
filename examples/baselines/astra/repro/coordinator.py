"""Resumable five-slot coordinator; the frozen v4 evaluator is not modified."""
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path('/mnt/sdf/dataset/zhouxunzhe/hongyun/gpt-baseline-native-xfast')
PACKAGE = ROOT / 'workspace/The-Imitator-Game/astra_benchmark/paired200_20261031_native_success_v4'
CAMPAIGN = ROOT / 'experiments/astra200_native_p30_c5_20261031'


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def utc(timestamp=None):
    return datetime.datetime.fromtimestamp(time.time() if timestamp is None else timestamp,
        datetime.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def is_zero_call_initialize_timeout(attempt):
    """Certify the recorded startup fault; never retry a policy that already ran."""
    native=Path(attempt['native_output']);result_dir=Path(attempt['result_dir'])
    result=read(result_dir/'result.json');budget=read(native/'planning_budget.json')
    outcome=read(native/'outcome.json');closed=read(native/'controller/api_closed.json')
    if result.get('status')!='infrastructure_failure' or result.get('scored') is not False:
        return False
    if outcome.get('steps')!=0 or outcome.get('decisions')!=0 or outcome.get('native_success_ever'):
        return False
    if budget.get('used_seconds')!=0 or budget.get('active_since_monotonic') is not None:
        return False
    if closed.get('process_returncode') is None or closed.get('before_host_hold_step')!=0:
        return False
    if (native/'controller/worker.json').exists() or (native/'controller/session.json').exists():
        return False
    try:
        requests=[json.loads(line) for line in (native/'controller/rpc_in.jsonl').read_text().splitlines()]
        policy_log=(result_dir/'policy.log').read_text()
    except (OSError,json.JSONDecodeError):
        return False
    return len(requests)==1 and requests[0].get('method')=='initialize' and \
        'Timed out waiting for Codex RPC initialize' in policy_log


def lock_coordinator(path):
    stream = Path(path).open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError('Another coordinator already owns the shared five-slot campaign lock')
    return stream


class RealBackend:
    def __init__(self, root):
        self.root = root
        self.procs = {}
        self.mem_checked = 0.
        self.mem = {}
        sys.path.insert(0, str(PACKAGE))
        import budget_runner
        self.native = budget_runner

    def verify(self, plan):
        protocol = read(plan['protocol_source'])
        cases = self.native.validate_protocol(protocol)
        assert digest(protocol) == plan['protocol_semantic_sha256'], 'Frozen protocol drift'
        assert protocol['model'] == 'gpt-6-astra' and protocol['reasoning_effort'] == 'xhigh'
        assert protocol['primary_metric'] == 'native_success_once' and protocol['model_finish_tool_enabled'] is False
        assert {c['case_id']: c for c in cases} == {c['case_id']: c for c in plan['ordered_cases']}
        assert len(cases) == 200 and len(plan['first20_case_ids']) == 20
        assert plan['scheduling']['maximum_active_model_sessions'] == 5
        assert len(plan['scheduling']['slots']) == 5
        checks = 0
        for group in ['source_sha256', 'toolchain_sha256', 'budget_source_sha256']:
            for filename, expected in protocol[group].items():
                assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
                checks += 1
        for filename, expected in {(c['demonstration'], c['demonstration_sha256']) for c in cases}:
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
        return dict(utc=utc(), passed=True, hash_checks=checks, cases=200, model='gpt-6-astra',
            effort='xhigh', maximum_concurrency=5, evaluator_changed=False)

    def billing(self):
        cfg = read(ROOT.parent / '.config/gpt-baseline/xfast.json')
        assert cfg['base_url'].rstrip('/') == 'https://xfast.transcengram.com/v1'
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        request = urllib.request.Request(cfg['base_url'].rstrip('/') + '/usage',
            headers={'Authorization': 'Bearer ' + cfg['api_key']})
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
            with opener.open(request, timeout=10) as response:
                data = json.load(response)
            row = next(x for x in data['model_stats'] if x['model'] == 'gpt-6-astra')
            return dict(status='captured', timestamp=time.time(), utc=utc(), valid=data.get('isValid'),
                remaining_usd=data.get('remaining'), actual_cost=float(row['actual_cost']),
                unit=data.get('unit'), usage={k:row.get(k, 0) for k in
                    ['requests','input_tokens','cache_read_tokens','output_tokens','total_tokens','cost','actual_cost']})
        except Exception as exc:
            return dict(status='unavailable', timestamp=time.time(), utc=utc(), error_type=type(exc).__name__)

    def gpu_available(self, gpu):
        if time.monotonic() - self.mem_checked > 5:
            result = subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=10)
            self.mem = {int(line.split(',')[0]):int(line.split(',')[1]) for line in result.stdout.splitlines()}
            self.mem_checked = time.monotonic()
        return self.mem.get(int(gpu), 0) >= 8192

    def launch(self, attempt, case, protocol):
        command = [sys.executable, '-B', '-u', str(PACKAGE/'budget_runner.py'), '--profile','astra',
            '--case-id',case['case_id'],'--run-id',attempt['run_id'],'--gpu',str(attempt['gpu']),
            '--protocol',protocol,'--results-root',str(self.root/'attempts')]
        attempt['command'] = command
        with (self.root/'logs'/f"{attempt['run_id']}.log").open('x') as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        self.procs[attempt['run_id']] = proc
        return self.native.process_identity(proc.pid)

    def _find_supervisor(self, attempt):
        # Recover the narrow crash window between spawning and persisting the PID.
        matches = []
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():
                continue
            try:
                args = (p/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            args = [x.decode(errors='replace') for x in args if x]
            if str(PACKAGE/'budget_runner.py') in args and '--run-id' in args and '--results-root' in args:
                if args[args.index('--run-id')+1] == attempt['run_id'] and args[args.index('--results-root')+1] == str(self.root/'attempts'):
                    matches.append(self.native.process_identity(int(p.name)))
        matches = [x for x in matches if x and x[3] != 'Z']
        if len(matches) > 1:
            raise RuntimeError('Duplicate owned supervisor for one attempt')
        return matches[0] if matches else None

    def _same_alive(self, identity):
        if not identity:
            return False
        current = self.native.process_identity(identity[0])
        return bool(current and tuple(current[:3]) == tuple(identity[:3]) and current[3] != 'Z')

    def is_alive(self, attempt):
        proc = self.procs.get(attempt['run_id'])
        if proc:
            proc.poll()  # Reap owned exits; adopted processes are checked by identity.
        identity = attempt.get('supervisor_identity')
        if identity is None:
            identity = self._find_supervisor(attempt)
            if identity:
                attempt['supervisor_identity'] = identity
        native = Path(attempt['native_output'])
        worker = read(native/'controller/worker.json')
        if self._same_alive(identity):
            if worker and not attempt.get('model_identity'):
                pid = worker.get('pid')
                if pid in self.native.descendants(identity[0]):
                    attempt['model_identity'] = self.native.process_identity(pid)
            if worker and (worker.get('model'), worker.get('reasoning_effort')) != ('gpt-6-astra','xhigh'):
                attempt['binding_error'] = True
                self.native.signal_group(identity, signal.SIGTERM)
            return True
        model = attempt.get('model_identity')
        if model and self._same_alive(model):
            # A dead supervisor must not leave a sixth model behind after slot reuse.
            first = attempt.setdefault('orphan_cleanup_started', time.time())
            self.native.signal_group(model, signal.SIGTERM if time.time()-first < 8 else signal.SIGKILL)
            return True
        if worker and not model:
            pid = worker.get('pid', 0)
            candidate = self.native.process_identity(pid)
            if candidate and candidate[3] != 'Z':
                try:
                    env = (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\0')
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    env = []
                marker = ('ROLLOUT_CODEX_STATE_DIR='+str(native/'controller/state')).encode()
                if marker in env:
                    attempt['model_identity'] = candidate
                    self.native.signal_group(candidate, signal.SIGTERM)
                    return True
                # PID may have been reused. Never signal an unrelated process.
        return False

    def result(self, attempt):
        value = read(Path(attempt['result_dir'])/'result.json')
        if not value:
            return dict(status='infrastructure_failure', scored=False, task_success=None,
                error='Supervisor exited without a result file')
        if (value.get('model'), value.get('reasoning_effort'), value.get('planning_time_limit_seconds')) != ('gpt-6-astra','xhigh',1800):
            return dict(status='infrastructure_failure', scored=False, task_success=None, error='Runtime binding mismatch')
        return value

    def bootstrap_retryable(self, attempt):
        return is_zero_call_initialize_timeout(attempt)

    def repair_runtime_cache(self):
        # This cache contains launcher shims, not sessions, credentials or policy state.
        # Only rotate it after every model worker has exited, and retain all old files.
        for process in Path('/proc').iterdir():
            if not process.name.isdigit():
                continue
            try:
                args=[x.decode(errors='replace') for x in (process/'cmdline').read_bytes().split(b'\0') if x]
            except (OSError,ProcessLookupError):
                continue
            if 'app-server' in args and any(str(ROOT) in arg for arg in args):
                identity=self.native.process_identity(int(process.name))
                if identity and identity[3]!='Z':
                    raise RuntimeError('Refusing startup-cache repair while an owned app-server is alive')
        home=ROOT/'runtime/codex_home';temp=home/'tmp'
        assert not temp.is_symlink(), 'Unexpected symlink at the owned startup-cache root'
        config_before=hashlib.sha256((home/'config.toml').read_bytes()).hexdigest()
        sessions_inode=(home/'sessions').stat().st_ino
        archive=self.root/'runtime_cache_quarantine'/f'tmp_{time.time_ns()}'
        archive.parent.mkdir(exist_ok=True)
        locks=[]
        try:
            if temp.exists():
                for lockfile in temp.glob('arg0/*/.lock'):
                    stream=lockfile.open('a+')
                    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        stream.close();raise RuntimeError('An active launcher still owns a startup-cache lock')
                    locks.append(stream)
                temp.rename(archive)
            temp.mkdir(mode=0o700,exist_ok=True)
        finally:
            for stream in locks:stream.close()
        assert hashlib.sha256((home/'config.toml').read_bytes()).hexdigest()==config_before
        assert (home/'sessions').stat().st_ino==sessions_inode
        record=dict(kind='quarantined_startup_tmp',utc=utc(),archive=str(archive),
            config_unchanged=True,sessions_unchanged=True,model_generation_calls=0)
        save(archive.parent/(archive.name+'.json'),record)
        return record


class Campaign:
    def __init__(self, root, plan, backend, *, resume=False, retry_infrastructure=False,
                 now=time.time, settle_seconds=60, bootstrap_recovery=False):
        self.root, self.plan, self.backend, self.now = Path(root), plan, backend, now
        self.settle_seconds = settle_seconds
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'logs').mkdir(exist_ok=True)
        self.signature = digest(plan)
        self.state = read(self.root/'ledger.json')
        if self.state:
            if not resume:
                raise RuntimeError('Campaign exists; use --resume rather than restarting cases')
            assert self.state['plan_sha256'] == self.signature, 'Campaign plan drift'
            if (self.state.get('halt_reason') or {}).get('kind') == 'infrastructure_failure' and not retry_infrastructure:
                raise RuntimeError('Repair the recorded infrastructure issue, then explicitly retry unscored infrastructure attempts')
            if retry_infrastructure:
                for case in self.state['cases']:
                    if case['status'] == 'infrastructure_failure' and not case.get('selected'):
                        case['status'] = 'pending'
                self.state.setdefault('recovery_events', []).append(dict(utc=utc(now()), kind='explicit_infrastructure_recovery'))
            self.state['halt_reason'] = None
        else:
            self.state = dict(schema=1, plan_sha256=self.signature, started_at=now(), started_utc=utc(now()),
                status='ready', halt_reason=None, first20_gate_passed=False, billing_last_attempt=0,
                cases=[dict(case_id=c['case_id'], rank=i, status='pending', attempts=[], selected=None)
                    for i,c in enumerate(plan['ordered_cases'])])
        if bootstrap_recovery and not self.state.get('bootstrap_recovery_enabled'):
            self.state['bootstrap_recovery_enabled']=True
            self.state.setdefault('recovery_events',[]).append(dict(utc=utc(now()),
                kind='enable_bounded_zero_call_startup_recovery',maximum_retries_per_case=1,
                policy='Only initialize timeout before thread/model/action execution; never replay a scored case'))
        self.lookup = {c['case_id']:c for c in plan['ordered_cases']}
        self.persist()

    def persist(self):
        self.state['updated_utc'] = utc(self.now())
        save(self.root/'ledger.json', self.state)
        save(self.root/'summary.json', self.summary())

    def summary(self):
        cases = self.state['cases']
        def group(rows):
            selected = [c['selected'] for c in rows if c.get('selected')]
            successes = sum(s.get('task_success') is True for s in selected)
            endpoints = [s for s in selected if s.get('native_success_at_500') is not None]
            return dict(planned=len(rows), scored=len(selected), pending=len(rows)-len(selected),
                native_successes=successes, success_rate_scored=successes/len(selected) if selected else None,
                official_success_rate=successes/len(rows) if len(selected)==len(rows) else None,
                planning_timeout_failures=sum(s['status']=='time_exhausted' and s['task_success'] is False for s in selected),
                endpoint_observed=len(endpoints), endpoint_successes=sum(s['native_success_at_500'] is True for s in endpoints),
                infrastructure_attempts=sum(a.get('status')=='infrastructure_failure' for c in rows for a in c['attempts']))
        selected = [c['selected'] for c in cases if c.get('selected')]
        recent = sorted(selected,key=lambda x:x['finished_at'])[-20:]
        mean_wall = sum(x.get('elapsed_seconds') or 0 for x in recent)/len(recent) if recent else None
        last = self.state.get('billing_latest', {})
        baseline = self.state.get('billing_before', {})
        spent = last.get('actual_cost', 0)-baseline.get('actual_cost', 0) if last and baseline else None
        active = [dict(case_id=c['case_id'], **c['attempts'][-1],
            native_progress=read(Path(c['attempts'][-1]['native_output'])/'progress.json'),
            planning_progress=read(Path(c['attempts'][-1]['result_dir'])/'progress.json'))
            for c in cases if c['status']=='running']
        return dict(campaign_id=self.plan['campaign_id'],status=self.state['status'],updated_utc=utc(self.now()),
            model='gpt-6-astra',effort='xhigh',maximum_concurrency=5,active_cases=active,
            primary_metric='native_success_once',planning_time_limit_seconds=1800,wall_time_cap_seconds=None,
            overall=group(cases),groups={s:group([c for c in cases if self.lookup[c['case_id']]['split']==s]) for s in ['seen','unseen']},
            by_level={s:group([c for c in cases if self.lookup[c['case_id']]['level']==s]) for s in ['L0','L1','L2','L3']},
            first20_gate_passed=self.state['first20_gate_passed'],halt_reason=self.state.get('halt_reason'),
            actual_cost_delta_usd=spent,remaining_shared_balance_usd=last.get('remaining_usd'),
            cost_scope='Astra shared-key billing window; includes any concurrent external Astra use',
            recent_mean_wall_seconds=mean_wall,
            estimated_remaining_hours=(len(cases)-len(selected))*mean_wall/5/3600 if mean_wall else None,
            rows=[dict(case_id=c['case_id'],status=c['status'],run_id=c.get('selected',{}).get('run_id') if c.get('selected') else None,
                task_success=c.get('selected',{}).get('task_success') if c.get('selected') else None) for c in cases])

    def billing(self, force=False):
        now = self.now()
        if not force and now-self.state['billing_last_attempt'] < 60:
            return
        self.state['billing_last_attempt'] = now
        value = self.backend.billing()
        with (self.root/'billing_snapshots.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n')
        if value.get('status') == 'captured':
            self.state.setdefault('billing_before', value)
            self.state['billing_latest'] = value
            self.state['billing_last_success'] = now
        else:
            self.state['billing_last_error'] = value

    def halt(self, kind, **details):
        if not self.state.get('halt_reason'):
            self.state['halt_reason'] = dict(kind=kind,utc=utc(self.now()),**details)

    def harvest(self):
        for case in self.state['cases']:
            if case['status'] != 'running':
                continue
            attempt = case['attempts'][-1]
            if self.backend.is_alive(attempt):
                continue
            result = self.backend.result(attempt)
            valid = result.get('scored') is True and isinstance(result.get('task_success'), bool)
            attempt.update(status=result.get('status','infrastructure_failure'),finished_utc=utc(self.now()))
            if valid:
                case['selected'] = {k:result.get(k) for k in ['status','task_status','task_success','native_success_once',
                    'native_success_at_500','planning_seconds','elapsed_seconds']}
                case['selected'].update(run_id=attempt['run_id'],finished_at=self.now(),
                    result_file=str(Path(attempt['result_dir'])/'result.json'),outcome=result.get('outcome',{}))
                case['status'] = 'scored'
            else:
                case['status'] = 'infrastructure_failure'
                attempt['status'] = 'infrastructure_failure'
            if (not valid and self.state.get('bootstrap_recovery_enabled') and len(case['attempts'])==1
                    and len(self.state.get('runtime_cache_repairs',[]))<5
                    and self.backend.bootstrap_retryable(attempt)):
                case['status']='pending'
                attempt['controlled_recovery']='zero_call_initialize_timeout'
                self.state['runtime_cache_recovery_required']=True
                self.persist()
                continue
            if not valid or result.get('status') not in ['complete','time_exhausted'] or attempt.get('binding_error'):
                self.halt('infrastructure_failure',case_id=case['case_id'],run_id=attempt['run_id'],
                    error=result.get('error'),result_status=result.get('status'))
            self.persist()

    def step(self):
        self.harvest()
        self.billing()
        now = self.now()
        if all(c.get('selected') for c in self.state['cases']):
            ended=self.state.setdefault('cases_finished_at',now)
            self.state['status']='finalizing_billing'
            if now-ended >= self.settle_seconds:
                self.billing(force=True)
                self.state.update(status='completed',completed_utc=utc(now))
            self.persist()
            return self.state['status']
        latest = self.state.get('billing_latest', {})
        baseline = self.state.get('billing_before', {})
        if (self.root/'STOP_ADMISSIONS.json').exists():
            self.halt('operator_pause')
        if latest and latest['actual_cost'] < baseline['actual_cost']:
            self.halt('billing_counter_reset')
        if latest and latest['actual_cost']-baseline['actual_cost'] >= self.plan['budget_design']['pause_new_cases_at_observed_spend_usd']:
            self.halt('spend_pause',threshold_usd=self.plan['budget_design']['pause_new_cases_at_observed_spend_usd'])
        if latest and (latest.get('valid') is False or float(latest.get('remaining_usd') or 0)<50):
            self.halt('insufficient_shared_balance')
        active = [c for c in self.state['cases'] if c['status']=='running']
        if self.state.get('halt_reason'):
            self.state['status']='draining' if active else 'paused'
            self.persist()
            return self.state['status']
        if not latest or now-self.state.get('billing_last_success',0)>300:
            self.state['status']='waiting_for_billing'
            self.persist()
            return self.state['status']
        if self.state.get('runtime_cache_recovery_required'):
            if active:
                self.state['status']='draining_for_startup_cache_recovery'
                self.persist()
                return self.state['status']
            try:
                repair=self.backend.repair_runtime_cache()
            except Exception as exc:
                self.halt('runtime_cache_recovery_failed',error=str(exc))
                self.state['status']='paused';self.persist();return 'paused'
            self.state.setdefault('runtime_cache_repairs',[]).append(repair)
            self.state['runtime_cache_recovery_required']=False
            self.persist()
        first = self.state['cases'][:20]
        if all(c.get('selected') for c in first) and not self.state['first20_gate_passed']:
            finished = self.state.setdefault('first20_finished_at',now)
            if now-finished < self.settle_seconds:
                self.state['status']='first20_billing_settlement'
                self.persist()
                return self.state['status']
            self.billing(force=True)
            if self.state.get('billing_last_success',0) < now:
                self.state['status']='waiting_for_first20_billing'
                self.persist()
                return self.state['status']
            self.state['first20_gate_passed']=True
            report=self.summary()
            report.update(gate='passed',success_rate_does_not_control_admissions=True,
                projected_200_cost_usd=(report.get('actual_cost_delta_usd') or 0)*10)
            save(self.root/'first20_report.json',report)
            # Re-check fresh billing before admitting case 21.
            self.persist()
            return self.step()
        allowed = self.state['cases'] if self.state['first20_gate_passed'] else first
        occupied = {c['attempts'][-1]['slot'] for c in active}
        launched = False
        for slot in self.plan['scheduling']['slots']:
            if slot['slot'] in occupied or not self.backend.gpu_available(slot['gpu']):
                continue
            case = next((c for c in allowed if c['status']=='pending'), None)
            if case is None:
                break
            run=f"a200_{case['case_id'].lower()}_a{len(case['attempts'])+1:02d}"
            attempt=dict(run_id=run,slot=slot['slot'],gpu=slot['gpu'],status='running',
                started_at=now,started_utc=utc(now),supervisor_identity=None,
                native_output=str(PACKAGE/'runs'/run),result_dir=str(self.root/'attempts'/run))
            case['attempts'].append(attempt);case['status']='running'
            self.persist()  # Write the immutable run ID before spawning a paid worker.
            attempt['supervisor_identity']=self.backend.launch(attempt,self.lookup[case['case_id']],self.plan['protocol_source'])
            occupied.add(slot['slot']);launched=True
            assert len(occupied)<=5
            self.persist()
        self.state['status']='running' if occupied else 'waiting_for_resources'
        self.persist()
        return self.state['status']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=CAMPAIGN)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--retry-infrastructure',action='store_true')
    parser.add_argument('--retry-pre-model-startup-once',action='store_true')
    args=parser.parse_args()
    plan=read(args.root/'launch_plan.json')
    backend=RealBackend(args.root)
    verification=backend.verify(plan)
    save(args.root/'prelaunch_verification.json',verification)
    if not args.execute:
        print(json.dumps(dict(verification=verification,launches_model=False)))
        return 0
    if args.retry_infrastructure and not args.resume:
        parser.error('Infrastructure recovery requires --resume')
    lock=lock_coordinator(ROOT/'runtime/astra200_coordinator.lock')
    campaign=Campaign(args.root,plan,backend,resume=args.resume,retry_infrastructure=args.retry_infrastructure,
        bootstrap_recovery=args.retry_pre_model_startup_once)
    campaign.state['coordinator_pid']=os.getpid()
    campaign.state['coordinator_revision_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def pause(signum, frame):
        campaign.halt('operator_pause',signal=signum)
    signal.signal(signal.SIGTERM,pause)
    signal.signal(signal.SIGINT,pause)
    try:
        campaign.billing(force=True)
        while True:
            status=campaign.step()
            summary=campaign.summary()
            save(ROOT/'active_experiment.json',dict(batch_id=plan['campaign_id'],experiment_dir=str(args.root),
                status=status,coordinator_pid=os.getpid(),model_profiles=['astra'],model='gpt-6-astra',effort='xhigh',
                maximum_concurrency=5,planned_cases=200,running_cases=len(summary['active_cases']),
                scored_cases=summary['overall']['scored'],native_successes=summary['overall']['native_successes'],
                planning_time_limit_seconds=1800,wall_time_cap_seconds=None,
                summary=str(args.root/'summary.json'),ledger=str(args.root/'ledger.json'),
                full_200_case_campaign_started=any(c['attempts'] for c in campaign.state['cases'])))
            if status in ['completed','paused']:
                break
            time.sleep(2)
    except BaseException:
        save(args.root/'coordinator_error.json',dict(utc=utc(),traceback=traceback.format_exc()))
        campaign.halt('coordinator_error')
        campaign.state['status']='coordinator_error_active_workers_remain_bounded'
        campaign.persist()
        # Independent per-episode supervisors retain their planning deadlines and results.
        raise
    finally:
        lock.close()
    print(json.dumps({k:v for k,v in campaign.summary().items() if k not in ['rows','active_cases']},ensure_ascii=False))
    return 0 if campaign.state['status']=='completed' else 2


if __name__=='__main__':
    raise SystemExit(main())
