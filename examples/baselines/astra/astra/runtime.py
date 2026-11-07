"""Materialize the unchanged historical harness outside the source checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1]
NATIVE_REL = Path('workspace/The-Imitator-Game/astra_benchmark/paired200_20261031_native_success_v4')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def link(source, target):
    source=Path(source).resolve();target=Path(target)
    if target.is_symlink() and target.resolve()==source:return
    if target.exists() or target.is_symlink():raise ValueError(f'Would overwrite {target}')
    target.symlink_to(source,target_is_directory=source.is_dir())


def load_config(path):
    cfg=read(path)
    for key in ['archive_root','work_root','python_executable','credentials_file','human_root']:
        if not Path(cfg[key]).is_absolute():raise ValueError(f'{key} must be absolute')
    root=Path(cfg['work_root']).resolve();archive=Path(cfg['archive_root']).resolve()
    checkout=PACKAGE.parents[2]
    if root==archive or root.is_relative_to(archive) or root.is_relative_to(checkout):
        raise ValueError('Use a new external work root, separate from the source archive and checkout')
    if not (archive/NATIVE_REL/'run_episode.py').is_file():raise ValueError('Frozen source archive not found')
    if not Path(cfg['python_executable']).is_file():raise ValueError('Python executable not found')
    if not Path(cfg['credentials_file']).is_file():raise ValueError('External credentials file not found')
    if not Path(cfg['human_root']).is_dir():raise ValueError('Human demonstration directory not found')
    name=cfg.get('campaign_id','astra200_30min_reproduction')
    if not name.replace('_','').replace('-','').isalnum():raise ValueError('Invalid campaign ID')
    return cfg


def verify_package():
    manifest=read(PACKAGE/'repro/frozen_code_manifest.json')
    for name,digest in manifest.items():
        if hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Frozen harness changed: {name}')
    return len(manifest)


def materialize(config_path):
    cfg=load_config(config_path);verify_package()
    root=Path(cfg['work_root']);archive=Path(cfg['archive_root']);stamp=root/'prepared.json'
    cfg_sha=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if stamp.exists():
        if read(stamp)['config_sha256']!=cfg_sha:raise ValueError('Prepared runtime configuration changed')
        return cfg
    if root.exists() and any(root.iterdir()):raise ValueError('Choose an empty work root')
    root.mkdir(parents=True,exist_ok=True)
    native=root/NATIVE_REL;native.mkdir(parents=True)
    workspace=root/'workspace';project=workspace/'The-Imitator-Game'
    for p in (archive/'workspace').iterdir():
        if p.name!='The-Imitator-Game':link(p,workspace/p.name)
    for p in (archive/'workspace/The-Imitator-Game').iterdir():
        if p.name!='astra_benchmark':link(p,project/p.name)
    benchmark=project/'astra_benchmark'
    for p in (archive/'workspace/The-Imitator-Game/astra_benchmark').iterdir():
        if p.name!=NATIVE_REL.name:link(p,benchmark/p.name)
    for p in (PACKAGE/'native_v4').iterdir():
        if p.is_file():shutil.copy2(p,native/p.name)
    for name in ['robot_model','toolchain']:
        link(archive/NATIVE_REL/name,native/name)

    # Only the deployment helper's two machine-local paths change. Native policy,
    # prompts, image bridge, action bounds, native success and budget code are copied verbatim.
    helper=(PACKAGE/'repro/run_native.py').read_text()
    lines=helper.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('PYTHON = '):lines[i]='PYTHON = '+repr(cfg['python_executable'])
        if line.startswith('CREDENTIALS = '):lines[i]='CREDENTIALS = '+repr(cfg['credentials_file'])
    (root/'run_native.py').write_text('\n'.join(lines)+'\n')
    credential_dir=root.parent/'.config/gpt-baseline';credential_dir.mkdir(parents=True,exist_ok=True)
    link(cfg['credentials_file'],credential_dir/'xfast.json')
    provider=(archive/'xfast_provider.toml').read_text()
    if 'sk-' in provider:raise ValueError('Do not copy an inline secret into runtime configuration')
    (root/'xfast_provider.toml').write_text(provider)
    (root/'runtime/codex_home').mkdir(parents=True)
    (root/'runtime/codex_home/config.toml').write_text(provider)

    protocol=read(PACKAGE/'configs/protocol_200_astra.json')
    for group in ['source_sha256','toolchain_sha256','budget_source_sha256']:
        resolved={}
        for rel,digest in protocol[group].items():
            candidate=root/rel
            if group=='budget_source_sha256' and candidate.is_file():path=candidate
            else:path=archive/rel
            resolved[str(path)]=digest
        protocol[group]=resolved
    for p in (PACKAGE/'native_v4').glob('*.py'):
        protocol['source_sha256'][str(native/p.name)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for case in protocol['cases']:
        case['demonstration']=str(Path(cfg['human_root'])/case['demonstration'])
    protocol_path=native/'protocol_200_astra.json';save(protocol_path,protocol)

    template=read(PACKAGE/'configs/campaign_template.json')
    by_id={c['case_id']:c for c in protocol['cases']}
    template.update(campaign_id=cfg.get('campaign_id','astra200_30min_reproduction'),
        status='prepared_not_launched',model='gpt-6-astra',reasoning_effort='xhigh',provider='xfast',
        protocol_source=str(protocol_path),
        protocol_semantic_sha256=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest(),
        ordered_cases=[by_id[k] for k in template.pop('ordered_case_ids')])
    if 'gpu_slots' in cfg:
        if len(cfg['gpu_slots'])!=5:raise ValueError('The recorded campaign requires five slots')
        template['scheduling']['slots']=[{'slot':i,'gpu':g} for i,g in enumerate(cfg['gpu_slots'])]
    template['budget_design']['pause_new_cases_at_observed_spend_usd']=cfg.get('spend_pause_usd',500)
    campaign=root/'experiments'/template['campaign_id'];campaign.mkdir(parents=True)
    save(campaign/'launch_plan.json',template)
    scheduler=(PACKAGE/'repro/coordinator.py').read_text()
    scheduler=scheduler.replace("ROOT = Path('/mnt/sdf/dataset/zhouxunzhe/hongyun/gpt-baseline-native-xfast')",'ROOT = Path('+repr(str(root))+')')
    scheduler=scheduler.replace("CAMPAIGN = ROOT / 'experiments/astra200_native_p30_c5_20261031'",'CAMPAIGN = Path('+repr(str(campaign))+')')
    (campaign/'coordinator.py').write_text(scheduler)
    save(stamp,{'config_sha256':cfg_sha,'native_source_unchanged':True,'campaign_dir':str(campaign),
        'historical_protocol':'video200_native_success_planning30_v4','model_calls_started':0})
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-config',type=Path,required=True)
    p.add_argument('--case-id',help='Optional single case; otherwise verify/run the five-slot campaign')
    p.add_argument('--run-id')
    p.add_argument('--gpu',default='1')
    p.add_argument('--execute',action='store_true',help='Start paid model calls; omitted by default')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();cfg=materialize(a.runtime_config)
    root=Path(cfg['work_root']);native=root/NATIVE_REL
    campaign=root/'experiments'/cfg.get('campaign_id','astra200_30min_reproduction')
    if a.case_id:
        if a.resume:p.error('Use campaign --resume; individual attempts cannot be overwritten')
        cmd=[sys.executable,str(native/'budget_runner.py'),'--profile','astra','--case-id',a.case_id,
             '--gpu',a.gpu,'--protocol',str(native/'protocol_200_astra.json'),'--results-root',str(campaign/'attempts')]
        if a.run_id:cmd+=['--run-id',a.run_id]
        if not a.execute:cmd+=['--dry-run']
    else:
        cmd=[sys.executable,str(campaign/'coordinator.py'),'--root',str(campaign)]
        if a.execute:cmd+=['--execute']
        if a.resume:cmd+=['--resume']
    return subprocess.call(cmd,cwd=root)


if __name__=='__main__':raise SystemExit(main())
