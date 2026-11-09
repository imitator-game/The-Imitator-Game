"""Shared Astra harness configuration and explicit code-mode image presentation."""
import json
from pathlib import Path
from environment import InvalidAction

ROOT=Path(__file__).resolve().parent
IMAGE_BRIDGE = r'''
Tool image presentation is essential. The dynamic tools return a string containing a
JSON line followed by one data:image/... URL per image. A text(result) call alone DOES
NOT display images. For EVERY imitation tool result, display its images with this pattern:
const r = await tools.imitation_start({}); // use the actual tool and arguments
for (const line of r.split('\n')) {
  if (line.startsWith('data:image/')) image(line);
  else text(line);
}
After each imitation_act result, finish that exec cell, inspect the displayed images,
and only then decide the next action. Do not loop or batch multiple actions without
returning the images for visual reasoning between actions.
Use JavaScript in exec for arithmetic, vectors, rotations and notes (store/load).
Use imitation_inspect for provided image paths or the robot URDF. Native shell and
view_image tools are disabled because their nested calls stall in this CLI setup.
Only the imitation tools and in-memory JavaScript calculations are permitted.
'''


def configure(config):
    config.update({'features.shell_tool':False,'features.view_image':False,
        'features.multi_agent':False,'web_search':'disabled'})
    config['model_provider'] = 'xfast'
    config['model'] = 'gpt-6.1-sol'
    config['model_reasoning_effort'] = 'xhigh'
    config['features.code_mode'] = True
    config['features.code_mode_only'] = True
    return config


def launcher():
    return [str(ROOT/'toolchain/node_modules/.bin/codex'),'--disable','apps','--disable','plugins',
        '-c','features.fast_mode=false','app-server','--stdio','--strict-config']


def inspect_artifact(episode,path):
    if not episode.started or episode.finished:
        raise InvalidAction('Episode is not active')
    if not isinstance(path,str):raise InvalidAction('path must be a string')
    item=Path(path).resolve()
    visible=episode.visible.resolve();robot=episode.robot_model.resolve()
    if item.is_relative_to(visible) and item.suffix.lower() in ('.png','.jpg','.jpeg') and item.is_file():
        return dict(images=[dict(name='provided_image',path=str(item))])
    if item==robot/'panda_v3.urdf':
        return dict(path=str(item),text=item.read_text())
    raise InvalidAction('Only this episode\'s observation/reference image files and robot URDF may be inspected')


def image_audit(path):
    counts=[]
    for line in Path(path).read_text().splitlines():
        event=json.loads(line);item=event.get('payload',{})
        if event.get('type')!='response_item' or item.get('type') not in ('function_call_output','custom_tool_call_output'):
            continue
        output=item.get('output')
        if isinstance(output,list):
            n=sum(x.get('type')=='input_image' for x in output)
            if n:counts.append(n)
    return dict(input_images=sum(counts),image_output_batches=counts,session_path=str(path))
