"""Unscored real-physics interface checks, shared by both prompt conditions."""
import copy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from environment import InvalidAction, save


def check_episode(episode,case,output,tools):
    original=episode.env.unwrapped.evaluate
    force_success=[False]
    def synthetic_success():
        info=original();info['success']=info['success'].clone().fill_(force_success[0]);return info
    episode.env.unwrapped.evaluate=synthetic_success
    packet=episode.start()
    common={'step_id','max_episode_steps','remaining_steps','request_id','current_eef',
        'images','camera_calibration','control_dt','rollout_finished','robot_description'}
    is_video=case['conditioning']=='video'
    assert set(packet)==common|({'reference_duration_seconds'} if is_video else {'instruction'})
    images={x['name'] for x in packet['images']}
    assert images==set(packet['camera_calibration'])|({'human_demonstration_timeline'} if is_video else set())
    assert len(packet['camera_calibration'])==4
    assert {x['name'] for x in tools}=={'imitation_start','imitation_act','imitation_inspect'}|({'imitation_reference_frame'} if is_video else set())
    from harness import inspect_artifact
    for forbidden in (output/'manifest.json',output/'evaluator/initial_state.json',Path(__file__)):
        try:
            inspect_artifact(episode,str(forbidden))
            raise AssertionError('Inspector accepted a forbidden path')
        except InvalidAction:pass
    if is_video:
        assert not (output/'instruction.txt').exists()
        frame=episode.reference_frame(0.0)
        assert Path(frame['images'][0]['path']).is_file()
        assert episode.tick==0
        assert hashlib.sha256((output/'reference.mp4').read_bytes()).hexdigest()==case['demonstration_sha256']
    else:
        assert packet['instruction']==case['instruction']
        assert not (output/'reference.mp4').exists()
    urdf=Path(packet['robot_description']['urdf'])
    assert ET.parse(urdf).find("link[@name='panda_hand_tcp']") is not None
    assert all(Path(a.urdf_path).read_bytes()==urdf.read_bytes() for _,a in episode.arms.values())
    def targets(p):
        return {k:dict(position=v['position'].copy(),quaternion_wxyz=v['quaternion_wxyz'].copy(),
            gripper_closed=False) for k,v in p['current_eef'].items()}
    packet=episode.act(packet['request_id'],3,targets(packet),'Infrastructure check: hold measured TCP poses.')
    assert episode.tick==3 and not episode.finished and not episode.native_success_ever
    invalid=copy.deepcopy(targets(packet));invalid['left']['position'][2]+=.1
    try:
        episode.act(packet['request_id'],5,invalid,'Infrastructure check: reject excessive displacement.')
        raise AssertionError('Accepted invalid action')
    except InvalidAction:
        assert episode.tick==3
    target=targets(packet);measured=np.array(target['left']['position']);target['left']['position'][2]+=.02
    packet=episode.act(packet['request_id'],4,target,'Infrastructure check: raise TCP by two centimeters.')
    change=np.array(packet['current_eef']['left']['position'])-measured
    assert np.isfinite(change).all()
    assert episode.tick==7 and not episode.finished
    force_success[0]=True
    packet=episode.act(packet['request_id'],3,targets(packet),'Infrastructure check: native success terminates policy actions.')
    assert episode.tick==8 and not episode.finished and episode.native_success_stop['step']==8
    episode.hold_after_native_success(model_closed=True)  # This unscored smoke has no model process.
    assert episode.tick==10 and episode.finished and episode.post_native_success_steps==2
    assert episode.outcome()['success_once'] is True
    # Unobstructed remote scene validates world/base and quaternion conversion physically.
    if case['task_id']==4:
        assert change[2]>.005 and np.linalg.norm(change[:2])<.02, change
    save(output/'smoke_checks.json',dict(conditioning=case['conditioning'],exact_conditioning=True,
        four_rgb_with_calibration=True,only_whitelisted_observation_fields=True,
        robot_urdf_matches_both_runtime_arms=True,native_success_stops_policy_actions=True,
        stops_at_budget=True,invalid_action_rejected_without_execution=True,
        measured_raise_m=change.tolist(),synthetic_success_unscored=True,model_api_calls=0))
    episode.env.unwrapped.evaluate=original
