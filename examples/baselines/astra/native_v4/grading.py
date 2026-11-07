"""Native success is authoritative; runtime completion and endpoint evidence are separate."""
from pathlib import Path
import json


def read(path):
    try: return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError): return {}


def native_evidence(folder, outcome):
    file = folder/'evaluator/trajectory.jsonl'
    truncated_tail = False
    try:
        lines = file.read_text().splitlines()
        rows = []
        for index,line in enumerate(lines):
            try: rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines)-1: raise
                truncated_tail = True
        flags = [bool(row['info']['success'][0]) for row in rows]
        prefix_valid = [row['step'] for row in rows] == list(range(1,len(rows)+1))
        valid = prefix_valid and not truncated_tail and len(rows) == outcome.get('steps')
    except (FileNotFoundError, KeyError, IndexError, json.JSONDecodeError):
        rows, flags = [], []
        prefix_valid = valid = outcome.get('steps') == 0 and (folder/'evaluator/initial_state.json').exists()
    return dict(valid=valid, prefix_valid=prefix_valid, rows=rows, success_once=any(flags),
                success_at_stop=flags[-1] if flags else False,
                first_success_step=next((i+1 for i,v in enumerate(flags) if v),None))


def grade_native_run(folder, returncode, planning_exhausted=False):
    folder = Path(folder)
    manifest = read(folder/'manifest.json')
    outcome = manifest.get('outcome') or read(folder/'outcome.json')
    evidence = native_evidence(folder,outcome)
    rows = evidence.pop('rows')
    verified_goal = evidence['prefix_valid'] and evidence['success_once']
    checks = dict(trace_valid=evidence['valid'], runner_exit_zero=returncode == 0,
        manifest_complete=manifest.get('status') == 'complete',
        physical_500=outcome.get('complete') is True and len(rows) == 500,
        images_verified=read(folder/'controller/model_image_audit.json').get('enough_images_for_actions') is True)
    videos = read(folder/'video_audit.json')
    checks['five_videos_501_frames'] = len(videos)==5 and all(
        str(v.get('streams',[{}])[0].get('nb_read_frames'))=='501' for v in videos.values())
    stop = read(folder/'native_success_stop.json')
    if stop:
        closed = read(folder/'controller/api_closed.json');step=stop['step']
        checks['native_stop_matches_trace'] = evidence['first_success_step'] == step
        checks['api_closed_before_hold'] = closed.get('process_returncode') is not None and closed.get('before_host_hold_step')==step
        checks['frozen_native_hold'] = outcome.get('post_native_success_hold_steps')==500-step and all(
            row.get('control_source')=='post_native_success_hold' and row['actions']==stop['hold_controller_actions']
            for row in rows[step:])
    complete = all(checks.values())
    scored = bool(verified_goal or complete or (planning_exhausted and evidence['valid']))
    task_success = bool(verified_goal) if scored else None
    endpoint = evidence['success_at_stop'] if evidence['valid'] and len(rows)==500 else None
    return dict(outcome=outcome, checks=checks, native_evidence=evidence,
        status='complete' if complete else 'time_exhausted' if planning_exhausted else 'infrastructure_failure',
        task_status='success' if verified_goal else 'planning_time_exhausted' if planning_exhausted and scored else 'step_limit_failure' if complete else 'unscored_infrastructure',
        scored=scored, task_success=task_success, native_success_once=task_success,
        budgeted_success_once=task_success, native_success_at_500=endpoint,
        task_success_at_500=endpoint, endpoint_observed=endpoint is not None,
        native_success_at_stop=evidence['success_at_stop'] if evidence['valid'] else None)
