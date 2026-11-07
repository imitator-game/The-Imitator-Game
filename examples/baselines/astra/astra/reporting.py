"""Export the selected 200-case cohort without dropping planning timeouts."""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

FIELDS = ['policy', 'backbone', 'label', 'num_tasks', 'split', 'env_id',
          'level', 'task', 'SR', 'SubSR', 'episodes', 'timestamp', 'task_base']
SUBSR_FIELDS = {'final_normalized_reward', 'peak_normalized_reward',
                'mean_step_normalized_reward'}


def read_json(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def validate_records(records):
    if len(records) != 200 or len({r['case_id'] for r in records}) != 200:
        raise ValueError('The main export requires 200 distinct selected cases')
    groups = collections.defaultdict(list)
    for r in records:
        if r['model'] != 'gpt-6-astra' or r['reasoning_effort'] != 'xhigh':
            raise ValueError('Unexpected model identity')
        if type(r['success_once']) is not bool:
            raise ValueError('Missing authoritative native success')
        if r['planning_limit_seconds'] != 1800 or r['demonstration_episode'] != 0:
            raise ValueError('Protocol mismatch')
        if r['status'] not in {'success', 'planning_time_exhausted', 'step_limit_failure'}:
            raise ValueError('An unscored/infrastructure attempt cannot become a task result')
        groups[(r['split'], r['task_id'], r['level'])].append(r)
    if len(groups) != 40 or {r['level'] for r in records} != {'L0','L1','L2','L3'}:
        raise ValueError('Expected 40 task-level groups')
    if collections.Counter(r['split'] for r in records) != {'seen':100,'unseen':100}:
        raise ValueError('Expected 100 seen and 100 unseen task-group cases')
    if any(sorted(r['seed'] for r in rr) != list(range(5)) for rr in groups.values()):
        raise ValueError('Each task-level requires exactly seeds 0..4')
    return groups


def collect(protocol, campaign_root):
    campaign_root = Path(campaign_root)
    ledger = read_json(campaign_root/'ledger.json')
    expected = {c['case_id']: c for c in protocol['cases']}
    if len(expected) != 200 or len(ledger['cases']) != 200:
        raise ValueError('Not the 200-case primary campaign')
    records=[]
    for c in ledger['cases']:
        if c['status'] != 'scored' or not c.get('selected'):
            raise ValueError('The primary cohort is incomplete')
        s=c['selected'];case=expected[c['case_id']]
        path=campaign_root/'attempts'/s['run_id']/'result.json'
        result=read_json(path);o=result['outcome']
        if not result.get('scored') or result['native_success_once'] != s['native_success_once']:
            raise ValueError('Selected result disagrees with the ledger')
        finished=dt.datetime.fromtimestamp(s['finished_at'],ZoneInfo('Asia/Shanghai'))
        records.append(dict(case_id=c['case_id'],run_id=s['run_id'],task_id=case['task_id'],
            title=case['title'],split=case['split'],level=case['level'],seed=case['seed'],
            environment=case['environment'],demonstration=Path(case['demonstration']).name,
            demonstration_episode=case['demonstration_episode'],demonstration_sha256=case['demonstration_sha256'],
            model=result['model'],reasoning_effort=result['reasoning_effort'],status=result['task_status'],
            success_once=result['native_success_once'],success_at_500=result['native_success_at_500'],
            endpoint_observed=result['endpoint_observed'],first_success_step=o['native_success_first_step'],
            steps=o['steps'],decisions=o['decisions'],planning_seconds=result['planning_seconds'],
            elapsed_seconds=result['elapsed_seconds'],planning_limit_seconds=result['planning_time_limit_seconds'],
            final_normalized_reward=o['final_normalized_reward'],peak_normalized_reward=o['peak_normalized_reward'],
            normalized_return=o['normalized_return'],
            mean_step_normalized_reward=o['normalized_return']/o['steps'] if o['steps'] else None,
            timestamp=finished.strftime('%Y%m%d_%H%M%S'),
            result_relative_path=f"attempts/{s['run_id']}/result.json",
            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    validate_records(records)
    return sorted(records,key=lambda r:(r['split'],r['level'],r['task_id'],r['seed']))


def means(rows, subsr_definition):
    sr=sum(r['success_once'] for r in rows)/len(rows)
    field=subsr_definition.get('field') if subsr_definition else None
    if field:
        if field not in SUBSR_FIELDS or not subsr_definition.get('source'):
            raise ValueError('SubSR needs an explicit supported field and definition source')
        values=[r.get(field) for r in rows]
        sub=None if any(v is None for v in values) else sum(values)/len(values)
    else:sub=None
    return {'SR':sr,'SubSR':sub}


def export(records, output, subsr_definition=None, source_metadata=None):
    groups=validate_records(records);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    table=[]
    for (split,task_id,level),rows in sorted(groups.items()):
        task=rows[0]['environment'].removesuffix('-v1')
        metrics=means(rows,subsr_definition)
        table.append(dict(policy='astra',backbone='gpt-6-astra',label='astra',num_tasks=0,
            split=split,env_id=f'{level}_{rows[0]["environment"]}',level=level,
            task=task,**metrics,episodes=len(rows),timestamp=max(r['timestamp'] for r in rows),task_base=task))
    with (out/'master_records.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');w.writeheader();w.writerows(table)
    summary={}
    for split in ['seen','unseen']:
        rows=[r for r in records if r['split']==split]
        level={lv:means([r for r in rows if r['level']==lv],subsr_definition) for lv in ['L0','L1','L2','L3']}
        rounded=lambda x:{k:round(v,4) if v is not None else None for k,v in x.items()}
        summary[split]={'0':{'astra':dict(policy='astra',backbone='gpt-6-astra',
            overall=rounded(means(rows,subsr_definition)),by_level={lv:rounded(v) for lv,v in level.items()})}}
    dump(out/'summary.json',summary)
    with (out/'episode_records.jsonl').open('w',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    observed=[r for r in records if r['endpoint_observed']]
    metadata=dict(schema_version=1,cohort=(source_metadata or {}).get('cohort','astra200_native_p30_c5_20261031'),episodes=200,task_level_rows=40,
        model='gpt-6-astra',reasoning_effort='xhigh',conditioning='video',
        num_tasks_semantics='0 means no IG source-task parameter training; not the ten evaluation tasks.',
        split_semantics='seen/unseen are the fixed IG source/held-out task groups, not Astra pretraining exposure.',
        SR=dict(field='native_success_once',successes=sum(r['success_once'] for r in records),denominator=200,
                value=sum(r['success_once'] for r in records)/200),
        SubSR=subsr_definition or dict(status='unconfirmed',field=None,
            reason='Reference exporter mapping has not yet been verified; null is not zero.'),
        overall=means(records,subsr_definition),
        reference_schema_note='Column layout matches the supplied Zero-WAM files. Their SR uses success_at_end; this primary Astra SR intentionally retains the approved success_once definition.',
        subsr_horizon_note='Last normalized reward at observed evaluation end. Timeouts retain their recorded prefix endpoint; no missing future simulator steps are invented.',
        planning_budget_seconds=1800,planning_time_includes_api_waits_and_retries=True,
        timeout_cases=sum(r['status']=='planning_time_exhausted' for r in records),
        endpoint=dict(observed=len(observed),successes=sum(r['success_at_500'] is True for r in observed),
            total_cohort_endpoint_rate=None,reason='Timeout cases were not advanced to step 500.'),
        timestamp_timezone='Asia/Shanghai',timestamp_semantics='Latest selected episode completion within the task-level group.',
        supplementary_continuations_excluded=True,
        historical_80_runs_excluded=True,source=source_metadata or {})
    dump(out/'export_metadata.json',metadata)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--campaign-root',type=Path)
    g.add_argument('--episode-records',type=Path)
    p.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'configs/protocol_200_astra.json')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--subsr-definition',type=Path,default=Path(__file__).resolve().parents[1]/'configs/metric_definition.json')
    a=p.parse_args()
    if a.campaign_root:
        records=collect(read_json(a.protocol),a.campaign_root)
        source={'cohort':a.campaign_root.name,'ledger_sha256':hashlib.sha256((a.campaign_root/'ledger.json').read_bytes()).hexdigest()}
    else:
        records=[json.loads(l) for l in a.episode_records.read_text().splitlines() if l.strip()]
        source={'episode_records_sha256':hashlib.sha256(a.episode_records.read_bytes()).hexdigest()}
    definition=read_json(a.subsr_definition) if a.subsr_definition else None
    meta=export(records,a.output_dir,definition,source)
    print(json.dumps({'rows':40,'episodes':200,'SR':meta['SR'],'SubSR':meta['SubSR']},ensure_ascii=False))


if __name__=='__main__':main()
