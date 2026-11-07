"""Native evaluator terminates model planning; planning has a cumulative 30-minute limit."""
import argparse
import base64
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
REPRO=ROOT.parents[2]/'robodojo_gpt6_repro'
sys.path.insert(0,str(REPRO/'GPT-as-Policy'))
os.environ['ROLLOUT_AUTH_PROFILE']='codex_a'
from hybrid_rollout.robodojo.skill.transport import StdioAppServer
from hybrid_rollout.robodojo.skill.run import agent_config,toml_value
from hybrid_rollout.robodojo.skill.network_recovery import is_network_error
from environment import Episode,InvalidAction,save
from harness import IMAGE_BRIDGE,configure,launcher,inspect_artifact,image_audit
from PIL import Image
from model_profiles import PROFILES, get_profile
from planning_budget import PLANNING_LIMIT_SECONDS, PlanningBudget, PlanningTimeExhausted

COMMON_PROMPT = """You are the autonomous robot policy for one Imitator Game evaluation.
Complete the demonstrated or instructed manipulation with the two Panda arms in the live
scene. Ground references in current images and adapt to visible geometry. The simulation
is paused while you think.

First call imitation_start exactly once. It returns task conditioning, current RGB views,
camera calibration, joint proprioception, world-coordinate TCP poses and robot-only URDF.
Use imitation_act, checking new observations after every action, to complete the visible
manipulation, including appropriate release and withdrawal. The host determines task
completion using the environment's native success criterion and closes the model session
as soon as that criterion is met. Do not stop based on a self-reported completion claim.
Continue using the supplied tools until the host ends the session, the 500-step physical
budget is reached, or the cumulative 30-minute model planning budget is exhausted.
The native success value is not provided as an observation. Resets, replay, other
policies and sub-agents are unavailable.

Actions are absolute TCP position [x,y,z] in world meters, unit quaternion [w,x,y,z], and
gripper_closed boolean for BOTH arms. left/right name arm indices 0/1; locate them from
their base/EEF poses and calibration, not from image-left. Panda TCP local +Z points along
the fingers; the initial quaternion is provided. The host converts world targets to each
robot's base and uses native robot-only IK. No object-aware planner or grasp macro exists.
Each target must be within 0.05 meters EUCLIDEAN distance and 0.35 radians of the latest
measured TCP. Each call executes 1-5 steps at control_dt, holding the target. To hold an
arm, copy its current pose and desired gripper state. True closes, false opens. Rejected
inputs execute no physics and may be corrected in the same episode. Use camera geometry
and measured TCP. Lift before moving laterally, clear rims, and release inside containers.

Use the supplied tools and in-memory JavaScript for calculations and image inspection. Permitted
evidence is only the provided task conditioning, robot-only URDF/meshes, this episode's
RGB, calibration, proprioception and own actions. Do not read evaluator/, manifests,
environment_metadata.json, source_snapshot/, controller logs, task definitions, simulator
source/state, dataset annotations, other runs, checkpoints or protocol files. Do not open
a second simulator or network connection. Do not modify evidence or the running program.
Keep notes and helper code in the agent workspace. Explain actions briefly in English
using visible evidence and purpose, without private chain-of-thought.
"""
CONDITION_PROMPTS = {
    'video': """Infer the task from the supplied human demonstration. imitation_start includes a
chronological 16-frame video timeline with timestamps. Use imitation_reference_frame
(seconds) for additional full-resolution frames. Only this episode's reference.mp4 may
be read as a demonstration. No task name or text description is supplied. Reproduce its
goal in the robot scene, adapting to differences in object appearance and layout.
""",
    'detailed': """Follow the supplied detailed English task instruction. imitation_start includes the
instruction verbatim. Ground it in current robot images. No human demonstration video,
timeline or reference frames are available or permitted.
""",
}


def prompt_for(conditioning):
    return COMMON_PROMPT + "\n" + CONDITION_PROMPTS[conditioning] + "\n" + IMAGE_BRIDGE


def obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


def specs(conditioning):
    array=lambda n:dict(type='array',items={'type':'number'},minItems=n,maxItems=n)
    arm=obj(dict(position=array(3),quaternion_wxyz=array(4),gripper_closed={'type':'boolean'}))
    result = [dict(type='function',name='imitation_start',description='Start this fixed episode once and receive task conditioning and robot RGB/proprioception.',inputSchema=obj({})),
        dict(type='function',name='imitation_act',description='Execute bounded dual-Panda world-frame TCP targets for 1-5 control steps, returning fresh observations.',inputSchema=obj(dict(
            request_id={'type':'string'},steps={'type':'integer','minimum':1,'maximum':5},
            target=obj(dict(left=arm,right=arm)),reason={'type':'string'})))]

    result.append(dict(type='function',name='imitation_inspect',
        description='Read a provided observation/reference image or the robot-only URDF. Other paths are rejected.',
        inputSchema=obj(dict(path={'type':'string'}))))
    if conditioning == 'video':
        result.append(dict(type='function',name='imitation_reference_frame',
            description='Inspect a human demonstration frame at a timestamp without advancing simulation.',
            inputSchema=obj(dict(seconds={'type':'number','minimum':0}))))
    return result


def content(packet):
    items=[dict(type='inputText',text=json.dumps(packet,separators=(',',':')))]
    for image in packet.get('images',[]):
        im=Image.open(image['path']).convert('RGB')
        edge=1536 if image['name']=='human_demonstration_timeline' else 640
        im.thumbnail((edge,edge))
        buf=io.BytesIO();im.save(buf,format='JPEG',quality=90)
        items.append(dict(type='inputImage',imageUrl='data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode()))
    return items


def run_policy(episode, output, profile=None):
    profile = profile or get_profile('sol')
    budget = PlanningBudget(output/'planning_budget.json')
    stop_reason = 'infrastructure_error'
    PROMPT = prompt_for(episode.conditioning)
    tools = specs(episode.conditioning)
    audit=output/'controller';audit.mkdir()
    agent=audit/'agent';agent.mkdir()
    (agent/'scratch').mkdir()
    (agent/'AGENTS.md').write_text(PROMPT)
    workspace=dict(python_executable=sys.executable,observations=str(episode.visible),
        robot_urdf=str(episode.robot_model/'panda_v3.urdf'),notes=str(agent/'NOTES.md'))
    workspace['instruction_file' if episode.conditioning=='detailed' else 'reference_video']=str(
        output/('instruction.txt' if episode.conditioning=='detailed' else 'reference.mp4'))
    save(agent/'workspace.json',workspace)
    (agent/'NOTES.md').write_text('# Episode notes\nNo observations yet.\n')
    (audit/'PROMPT.md').write_text(PROMPT)
    save(audit/'tools.json',tools)
    os.environ['ROLLOUT_CODEX_STATE_DIR']=str(audit/'state')
    config=configure(agent_config(audit,agent))
    config.update(model=profile['model'], model_reasoning_effort=profile['effort'])
    argv=launcher()
    for key,value in config.items():argv+=['-c',key+'='+toml_value(value)]
    transport=StdioAppServer(argv,audit)
    try:
        transport.request('initialize',dict(clientInfo=dict(name='imitator-astra-benchmark',version='1'),
            capabilities=dict(experimentalApi=True)),120)
        transport.notify('initialized',{})
        response=transport.request('thread/start',dict(cwd=str(agent),model=profile['model'],modelProvider='xfast',
            config=dict(model_reasoning_effort=profile['effort']),developerInstructions=PROMPT,dynamicTools=tools,
            ephemeral=False,allowProviderModelFallback=False,approvalPolicy='never',approvalsReviewer='auto_review',
            permissions='rollout_agent',runtimeWorkspaceRoots=[str(agent)]),120)
        if (response.get('model'),response.get('reasoningEffort'))!=(profile['model'],profile['effort']):
            raise RuntimeError('Requested model or reasoning effort was not honored')
        thread=response['thread']['id']
        save(audit/'worker.json',dict(model=response['model'],reasoning_effort=response['reasoningEffort'],
            thread_id=thread,pid=transport.process.pid,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
            model_profile=profile['name'],credential_group=profile['credential']))
        def turn(message):
            budget.resume()
            result=transport.request('turn/start',dict(threadId=thread,model=profile['model'],effort=profile['effort'],
                approvalPolicy='never',approvalsReviewer='auto_review',permissions='rollout_agent',
                cwd=str(agent),runtimeWorkspaceRoots=[str(agent)],input=[dict(type='text',text=message)]),120)
            return result['turn']['id']
        save(audit/'session.json',dict(path=response['thread']['path']))
        turn_id=turn('Call imitation_start and display its images with image(). Follow the task conditioning. Continue using the tools; the host controls task termination.')
        deadline=time.monotonic()+3600
        calls=recoveries=0
        while not episode.finished:
            event=transport.next_message(max(.1,deadline-time.monotonic()))
            method=event.get('method');params=event.get('params',{})
            if method=='item/tool/call':
                budget.pause('host_tool')
                if params.get('threadId')!=thread or params.get('turnId')!=turn_id:
                    raise RuntimeError('Tool thread or turn mismatch')
                name=params['tool'];args=params['arguments']
                save(audit/f'call_{calls:04d}_request.json',dict(tool=name,arguments=args))
                try:
                    if isinstance(args,str):
                        try:args=json.loads(args)
                        except json.JSONDecodeError as e:raise InvalidAction(str(e)) from e
                    handlers={'imitation_start':episode.start,'imitation_act':episode.act,
                        'imitation_inspect':lambda path:inspect_artifact(episode,path)}
                    if episode.conditioning=='video':handlers['imitation_reference_frame']=episode.reference_frame
                    if name not in handlers or not isinstance(args,dict):raise InvalidAction('Invalid tool/arguments')
                    expected={'imitation_start':set(),'imitation_act':{'request_id','steps','target','reason'},'imitation_reference_frame':{'seconds'},'imitation_inspect':{'path'}}[name]
                    if set(args)!=expected:raise InvalidAction('Tool argument keys differ from schema')
                    packet=handlers[name](**args);success=True
                except InvalidAction as error:
                    packet=dict(error=str(error),no_execution=True,request_id=getattr(episode,'request_id',None),
                        current_eef=episode.poses(),step_id=episode.tick)
                    success=False
                save(audit/f'call_{calls:04d}_result.json',packet)
                if episode.native_success_ever or episode.finished:
                    stop_reason = 'native_success' if episode.native_success_ever else 'step_limit'
                    budget.close(stop_reason)
                    # Do not reply: a tool response could trigger another paid request.
                    save(audit/'terminal_boundary.json', dict(tool_call_index=calls,stop_reason=stop_reason,
                        model_control_step=episode.tick, request_received=True,
                        reply_sent=False, next_turn_started=False))
                    print(json.dumps(dict(event='native_terminal',reason=stop_reason,step=episode.tick,
                        decisions=episode.decisions)),flush=True)
                    break
                rendered = content(packet)  # Image serialization is host work, not LLM planning.
                budget.resume()
                transport.reply(event['id'],dict(success=success,contentItems=rendered))
                calls+=1;deadline=time.monotonic()+3600
                print(json.dumps(dict(event='tool_done',tool=name,step=episode.tick,decisions=episode.decisions,finished=episode.finished,rejected=not success)),flush=True)
            elif method=='thread/tokenUsage/updated':
                save(output/'token_usage.json',params.get('tokenUsage',{}))
            elif method in ('item/started','item/completed'):
                item=params.get('item',{})
                if item.get('type') in ('commandExecution','imageView','fileChange','agentMessage'):
                    with (audit/'agent_events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
                    deadline=time.monotonic()+3600
            elif method=='turn/completed' and params.get('threadId')==thread:
                result=params['turn']
                if result.get('id')!=turn_id:continue
                if result.get('status')=='failed':
                    if not is_network_error(result.get('error')) or recoveries>=5:
                        raise RuntimeError('Model turn failed: '+json.dumps(result.get('error')))
                    recoveries+=1
                    time.sleep(10)
                turn_id=turn('Continue the same episode from the latest observation using the supplied tools. Do not reset or replay. The host controls termination; current step '+str(episode.tick)+'.')
                deadline=time.monotonic()+3600
            elif 'id' in event and method:
                raise RuntimeError('Unexpected approval or server request: '+str(method))
    except PlanningTimeExhausted:
        stop_reason = 'planning_time_exhausted'
        raise
    finally:
        budget.close(stop_reason)
        transport.close()
        save(audit/'api_closed.json',dict(pid=transport.process.pid,
            process_returncode=transport.process.poll(),stop_reason=stop_reason,
            closed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            before_host_hold_step=episode.tick))
        if transport.process.poll() is None:
            raise RuntimeError('Refusing further execution while the model process is alive')
        session=audit/'session.json'
        if session.exists():
            audit_result=image_audit(json.loads(session.read_text())['path'])
            # The final tool response need not be seen: no next action follows it.
            audit_result['minimum_images_for_actions']=4*episode.decisions+(1 if episode.conditioning=='video' else 0)
            audit_result['enough_images_for_actions']=audit_result['input_images']>=audit_result['minimum_images_for_actions']
            save(audit/'model_image_audit.json',audit_result)
            if (episode.finished or episode.native_success_ever) and not audit_result['enough_images_for_actions']:
                raise RuntimeError('Insufficient rendered model image inputs; refusing a valid evaluation score')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-id',required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--gpu',default='1')
    p.add_argument('--profile',choices=list(PROFILES),default='sol')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--model-smoke',action='store_true')
    p.add_argument('--proxy',default='http://127.0.0.1:7897')
    p.add_argument('--protocol',type=Path)
    args=p.parse_args()
    profile=get_profile(args.profile)
    args.protocol=args.protocol or ROOT/f'protocol_200_{args.profile}.json'
    if args.smoke and args.model_smoke:raise ValueError('Choose one smoke mode')
    if not args.run_id.replace('_','').replace('-','').isalnum():raise ValueError('Invalid run-id')
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,VK_ICD_FILENAMES='/etc/vulkan/icd.d/nvidia_icd.json')
    for key in ('all_proxy','ALL_PROXY'):os.environ.pop(key,None)
    for key in ('http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY'):os.environ[key]=args.proxy
    os.environ.update(NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1')
    protocol=json.loads(args.protocol.read_text())
    if (protocol.get('model'),protocol.get('reasoning_effort')) != (profile['model'],profile['effort']):
        raise RuntimeError('Protocol model does not match requested profile')
    if protocol.get('planning_time_limit_seconds') != PLANNING_LIMIT_SECONDS:
        raise RuntimeError('Every evaluation requires the cumulative 30-minute planning limit')
    case=next(c for c in protocol['cases'] if c['case_id']==args.case_id)
    for source,expected in (protocol['source_sha256']|protocol['toolchain_sha256']).items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=expected:raise RuntimeError('Source drift: '+source)
    output=ROOT/('preflight' if args.smoke else 'model_preflight' if args.model_smoke else 'runs')/args.run_id;output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='starting',case=case,model=profile['model'],reasoning_effort=profile['effort'],
        model_profile=profile['name'],planning_time_limit_seconds=PLANNING_LIMIT_SECONDS,
        kind='infrastructure_smoke' if args.smoke else 'model_smoke' if args.model_smoke else 'evaluation',started_at=datetime.datetime.now().isoformat(),
        launcher_pid=os.getpid(),protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        evaluation='video200_native_success_planning30_v4',proxy=args.proxy)
    snapshots=output/'source_snapshot';snapshots.mkdir()
    manifest['adapter_source_sha256']={}
    for name in ('run_episode.py','environment.py','harness.py','README.md',
                 'planning_budget.py','grading.py','model_profiles.py','budget_runner.py'):
        payload=(ROOT/name).read_bytes();(snapshots/name).write_bytes(payload)
        manifest['adapter_source_sha256'][name]=hashlib.sha256(payload).hexdigest()
    (snapshots/'protocol.json').write_bytes(args.protocol.read_bytes())
    save(output/'manifest.json',manifest)
    started=time.monotonic();episode=None
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        episode=Episode(case,output,10 if args.smoke or args.model_smoke else case['max_episode_steps'])
        manifest['status']='running';save(output/'manifest.json',manifest)
        if args.smoke:
            from preflight_checks import check_episode
            check_episode(episode,case,output,specs(episode.conditioning))
            manifest['status']='smoke_complete'
        else:
            run_policy(episode,output,profile)
            if episode.native_success_ever:
                episode.hold_after_native_success(model_closed=True)
            if not episode.finished:raise RuntimeError('Model stopped before fixed-budget completion')
            manifest['status']='model_smoke_complete' if args.model_smoke else 'complete'
    except BaseException as error:
        if isinstance(error,PlanningTimeExhausted) and episode is not None:
            episode.termination_reason='planning_time_exhausted'
        manifest.update(status='time_exhausted' if isinstance(error,PlanningTimeExhausted) else 'error',
            error_type=type(error).__name__,error=str(error))
        raise
    finally:
        if episode is not None:
            episode.close();manifest['outcome']=episode.outcome()
        manifest.update(ended_at=datetime.datetime.now().isoformat(),elapsed_seconds=round(time.monotonic()-started,2))
        save(output/'manifest.json',manifest)
        probes={}
        for path in output.glob('*.mp4'):
            if path.name=='reference.mp4':continue
            result=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(path)],capture_output=True,text=True)
            if result.returncode==0:probes[path.name]=json.loads(result.stdout)
        save(output/'video_audit.json',probes)
        print(json.dumps(dict(status=manifest['status'],output=str(output))),flush=True)


if __name__=='__main__':main()
