"""Native success is primary; endpoint coverage and execution failures are separate."""
import argparse
import collections
import json
from pathlib import Path
from budget_runner import PACKAGE, VERSION, read, save, validate_protocol
from model_profiles import PROFILES


def summarize(protocol, results_root):
    cases = validate_protocol(protocol)
    rows = []
    for case in cases:
        result = read(results_root / case['run_id'] / 'result.json')
        rows.append(dict(case_id=case['case_id'], run_id=case['run_id'], task_id=case['task_id'],
            title=case['title'], level=case['level'], split=case['split'], seed=case['seed'],
            status=result.get('status', 'not_started'), scored=result.get('scored', False),
            task_success=result.get('task_success'),
            success_at_end=result.get('native_success_at_500'),
            success_once=result.get('native_success_once'),
            planning_seconds=result.get('planning_seconds'),
            steps=result.get('outcome', {}).get('steps')))

    def group(items):
        scored = [r for r in items if r['scored']]
        succeeded = sum(r['task_success'] is True for r in scored)
        endpoints = [r for r in items if r['success_at_end'] is not None]
        return dict(planned=len(items), scored=len(scored), pending=len(items)-len(scored),
            status_counts=dict(collections.Counter(r['status'] for r in items)),
            task_successes=succeeded,
            successes_at_end=sum(r['success_at_end'] is True for r in endpoints),
            endpoint_observed=len(endpoints),
            successes_once=sum(r['success_once'] is True for r in scored),
            time_exhausted=sum(r['status'] == 'time_exhausted' for r in items),
            planning_timeout_failures=sum(r['status']=='time_exhausted' and r['task_success'] is False for r in items),
            success_rate_scored=succeeded/len(scored) if scored else None,
            official_success_rate=succeeded/len(items) if len(scored)==len(items) and not any(r['status']=='infrastructure_failure' for r in items) else None)
    return dict(protocol_version=VERSION, planning_time_limit_seconds=1800, wall_time_cap_seconds=None,
        primary_metric='native_success_once',
        overall=group(rows), groups={split: group([r for r in rows if r['split'] == split])
                                     for split in ['seen', 'unseen']}, rows=rows,
        note='Native success is never erased by timeout or missing endpoint coverage. A planning timeout with no native success is a task failure. Endpoint success is reported only when step 500 was observed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=list(PROFILES), default='sol')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.protocol = args.protocol or PACKAGE / f'protocol_200_{args.profile}.json'
    args.results_root = args.results_root or PACKAGE / 'campaigns' / args.profile
    result = summarize(read(args.protocol), args.results_root)
    if args.output:
        save(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))
