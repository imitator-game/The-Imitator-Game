"""Audited RGB/proprioception interface. Evaluation truth stays on the host."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation


def plain(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def numpy_array(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


def resolve_dual_arms(agent):
    agents = agent.agents_dict
    expected = {'panda_wristcam-0', 'panda_wristcam-1'}
    if set(agents) != expected:
        raise RuntimeError(f'Unexpected dual-arm keys: {sorted(agents)}')
    return {'left': ('panda_wristcam-0', agents['panda_wristcam-0']),
            'right': ('panda_wristcam-1', agents['panda_wristcam-1'])}


class InvalidAction(ValueError):
    """Rejected before any physics step."""


class Episode:
    def __init__(self, case, output, max_steps=500):
        import gymnasium as gym
        import mani_skill
        from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils as levels
        self.case, self.output = case, Path(output)
        self.visible = self.output/'observations'
        self.visible.mkdir()
        self.private = self.output/'evaluator'
        self.private.mkdir()
        self.tick = self.decisions = 0
        self.finished = self.success = self.started = False
        self.max_steps = max_steps
        self.native_success_ever = False
        self.native_success_first_step = None
        self.termination_reason = 'incomplete' 
        self.last_reward = self.peak_reward = 0.0
        self.episode_return = 0.0
        self.writers = {}
        self.history = []
        self.last_actions = None
        self.last_target = None
        self.native_success_stop = None
        self.post_native_success_steps = 0
        # Match examples/baselines/diffusion_policy/eval_dp_imitator.py,
        # including its standalone L3 ID and fixed (unmirrored) robot roots.
        for level in ('L1', 'L2', 'L3'):
            os.environ.pop('MANI_SKILL_'+level, None)
        if case['level'] != 'L0':
            os.environ['MANI_SKILL_'+case['level']] = '1'
        levels.set_l1_enabled(case['level'] == 'L1')
        levels.set_l2_enabled(case['level'] == 'L2')
        levels.set_l3_enabled(case['level'] == 'L3')
        levels.set_lr_mirror_enabled(None)
        levels.set_lr_mirror_robot_pose_enabled(False)
        env_id = case['environment']
        if case['level'] == 'L3':
            env_id = env_id.replace('-v1', 'L3-v1')
        self.env = gym.make(env_id, num_envs=1, obs_mode='rgb', control_mode='pd_ee_pose',
            reward_mode='normalized_dense', render_mode='rgb_array', sim_backend='cpu',
            render_backend='cuda:0', max_episode_steps=max_steps, hi_res=True,
            sensor_configs={'width': 640, 'height': 480, 'shader_pack': 'default'})
        self.obs, self.info = self.env.reset(seed=case['seed'])
        self.arms = resolve_dual_arms(self.env.unwrapped.agent)
        self.control_dt = float(self.env.unwrapped.control_timestep)
        save(self.private/'initial_state.json', self.env.unwrapped.get_state_dict())
        self.fps = round(1/self.control_dt)
        self.record_frame()
        self.robot_model = self.output/'robot_model'
        shutil.copytree(Path(__file__).resolve().parent/'robot_model', self.robot_model)
        self.conditioning = case['conditioning']
        if self.conditioning == 'detailed':
            self.instruction = case['instruction']
            assert hashlib.sha256(self.instruction.encode()).hexdigest() == case['instruction_sha256']
            (self.output/'instruction.txt').write_text(self.instruction+'\n')
        elif self.conditioning == 'video':
            self.prepare_reference()
        else:
            raise ValueError('Unknown conditioning')
        # Only static robot geometry is exported. No object geometry/state is supplied.
        self.robot_description = dict(urdf=str(self.robot_model/'panda_v3.urdf'),
            tcp_link='panda_hand_tcp', arm_indices={'left':0,'right':1},
            description='Both arms use this Panda wrist-camera URDF. Base poses and joint states are in current_eef.')
        save(self.private/'initial_info.json', self.info)
        save(self.output/'environment_metadata.json', {'env_id':env_id,'sim_backend':'cpu',
            'renderer':'SAPIEN CUDA','control_mode':'pd_ee_pose','control_dt':self.control_dt,
            'max_steps':max_steps,'registered_steps':gym.spec(env_id).max_episode_steps,
            'conditioning':self.conditioning,
            'level_flags':{k:getattr(levels,'is_'+k.lower()+'_enabled')() for k in ('L1','L2','L3')},
             'mirror_robot_pose':levels.is_lr_mirror_robot_pose_enabled(),
             'arm_uid_mapping':{side:uid for side,(uid,_) in self.arms.items()},
            'ignore_terminations':True,
            'instruction_sha256':case.get('instruction_sha256'),
            'demonstration_sha256':case.get('demonstration_sha256'),
            'robot_urdf_sha256':hashlib.sha256((self.robot_model/'panda_v3.urdf').read_bytes()).hexdigest(),
            'source':mani_skill.__file__})

    def prepare_reference(self):
        self.reference = self.output/'reference.mp4'
        shutil.copyfile(self.case['demonstration'], self.reference)
        if hashlib.sha256(self.reference.read_bytes()).hexdigest() != self.case['demonstration_sha256']:
            raise RuntimeError('Reference hash differs from frozen protocol')
        cap = cv2.VideoCapture(str(self.reference))
        try:
            self.ref_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.ref_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if self.ref_frames < 1 or self.ref_fps <= 0:
                raise RuntimeError('Invalid demonstration video')
            self.duration = self.ref_frames/self.ref_fps
            tiles = []
            for frame in np.linspace(0, self.ref_frames-1, 16).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                ok, im = cap.read()
                if not ok:
                    raise RuntimeError('Cannot decode demonstration frame')
                im = cv2.resize(im, (384, 216))
                cv2.putText(im, f'{frame/self.ref_fps:.2f}s', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                tiles.append(im)
        finally:
            cap.release()
        sheet = np.concatenate([np.concatenate(tiles[i:i+4], 1) for i in range(0,16,4)], 0)
        self.sheet = self.visible/'demonstration_timeline.png'
        cv2.imwrite(str(self.sheet), sheet)

    def reference_frame(self, seconds):
        if self.conditioning != 'video' or not self.started or self.finished:
            raise InvalidAction('Reference frames unavailable')
        if type(seconds) not in (float,int) or not np.isfinite(seconds) or not 0 <= seconds < self.duration:
            raise InvalidAction(f'seconds must be in [0,{self.duration})')
        cap = cv2.VideoCapture(str(self.reference))
        index = min(int(seconds*self.ref_fps),self.ref_frames-1)
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,frame=cap.read();cap.release()
        if not ok:
            raise RuntimeError('Reference decode failed')
        path=self.visible/f'reference_{index:05d}.png'
        cv2.imwrite(str(path),frame)
        return {'seconds':index/self.ref_fps,'images':[{'name':'human_reference','path':str(path)}]}

    def poses(self):
        result = {}
        for side, (_, agent) in self.arms.items():
            pose = plain(agent.tcp.pose.raw_pose)[0]
            result[side] = dict(position=pose[:3], quaternion_wxyz=pose[3:],
                robot_base_pose=plain(agent.robot.pose.raw_pose)[0],
                qpos=plain(agent.robot.get_qpos())[0],qvel=plain(agent.robot.get_qvel())[0])
        return result

    def record_frame(self):
        frames = {}
        for name, data in self.obs['sensor_data'].items():
            rgb = numpy_array(data['rgb'])[0,:,:,:3]
            frames[name] = rgb
            if name not in self.writers:
                self.writers[name] = imageio.get_writer(str(self.output/f'{name}.mp4'), fps=self.fps,
                    codec='libx264', quality=7, macro_block_size=1)
            self.writers[name].append_data(rgb)
        ordered = [cv2.resize(frames[k],(640,480)) for k in sorted(frames)]
        while len(ordered) < 4:
            ordered.append(np.zeros_like(ordered[0]))
        mosaic = np.concatenate([np.concatenate(ordered[:2],1),np.concatenate(ordered[2:4],1)],0)
        if 'multiview' not in self.writers:
            self.writers['multiview'] = imageio.get_writer(str(self.output/'multiview.mp4'), fps=self.fps,
                codec='libx264', quality=7, macro_block_size=1)
        self.writers['multiview'].append_data(mosaic)

    def packet(self, include_instruction=False):
        folder = self.visible/f'step_{self.tick:04d}'
        folder.mkdir(exist_ok=True)
        images = []
        for name, data in self.obs['sensor_data'].items():
            rgb = numpy_array(data['rgb'])[0,:,:,:3]
            path = folder/f'{name}.png'
            cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            images.append({'name':name,'path':str(path)})
        self.request_id = uuid.uuid4().hex
        result = dict(step_id=self.tick, max_episode_steps=self.max_steps,
            remaining_steps=self.max_steps-self.tick, request_id=self.request_id,
            current_eef=self.poses(), images=images, camera_calibration=plain(self.obs['sensor_param']),
            control_dt=self.control_dt, rollout_finished=self.finished)
        if include_instruction:
            result['robot_description'] = self.robot_description
            if self.conditioning == 'detailed':
                result['instruction'] = self.instruction
            else:
                result['reference_duration_seconds'] = self.duration
                images.insert(0, {'name':'human_demonstration_timeline','path':str(self.sheet)})
        if self.finished:
            result['evaluation_outcome'] = self.outcome()
        save(folder/'observation.json',result)
        return result

    def start(self):
        if self.started:
            raise InvalidAction('Episode is already started; resetting is unavailable')
        self.started = True
        return self.packet(include_instruction=True)

    def act(self, request_id, steps, target, reason):
        if not self.started or self.finished or self.native_success_stop is not None:
            raise InvalidAction('Episode is not active')
        if request_id != self.request_id:
            raise InvalidAction('Stale observation request_id')
        if type(steps) is not int or not 1 <= steps <= 5:
            raise InvalidAction('steps must be an integer from 1 to 5')
        if not isinstance(reason,str) or not reason.strip():
            raise InvalidAction('A brief visible-evidence/action-purpose reason is required')
        measured = self.poses()
        actions = {}
        for side,(uid,agent) in self.arms.items():
            try:
                p=np.asarray(target[side]['position'],dtype=float)
                q=np.asarray(target[side]['quaternion_wxyz'],dtype=float)
                closed=target[side]['gripper_closed']
            except (KeyError,TypeError,ValueError) as e:
                raise InvalidAction('Both arms require position, quaternion_wxyz, gripper_closed') from e
            if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(q).all() or type(closed) is not bool:
                raise InvalidAction('Pose dimensions/finiteness or gripper boolean invalid')
            if abs(np.linalg.norm(q)-1)>1e-3:
                raise InvalidAction('Quaternion must have unit norm')
            if np.linalg.norm(p-measured[side]['position'])>.050001:
                raise InvalidAction(f'{side}: target exceeds 5cm Euclidean displacement')
            rot=Rotation.from_quat(q[[1,2,3,0]])
            old=np.asarray(measured[side]['quaternion_wxyz'])
            if (rot*Rotation.from_quat(old[[1,2,3,0]]).inv()).magnitude()>.350001:
                raise InvalidAction(f'{side}: target exceeds 0.35rad rotation')
            base=np.asarray(measured[side]['robot_base_pose'])
            base_rot=Rotation.from_quat(base[[4,5,6,3]])
            local_p=base_rot.inv().apply(p-base[:3])
            local_rot=base_rot.inv()*rot
            # Native euler_angles_to_matrix(...,"XYZ") uses intrinsic XYZ.
            actions[uid]=np.r_[local_p,local_rot.as_euler('XYZ'),-1. if closed else 1.].astype(np.float32)
        self.last_actions = {uid: action.copy() for uid, action in actions.items()}
        self.last_target = plain(target)
        executed = self._advance(actions, steps, 'model_action')
        self.decisions+=1
        self.history.append(dict(decision=self.decisions,reason=reason,target=target,executed_steps=executed))
        save(self.output/'action_history.json',self.history)
        result=self.packet()
        save(self.output/'progress.json',dict(step=self.tick,decisions=self.decisions,finished=self.finished,
            phase='model_control'))
        return result

    def hold_after_native_success(self, *, model_closed):
        """Optional endpoint accounting after native success; no further policy calls."""
        if not model_closed or self.native_success_stop is None or self.last_actions is None:
            raise RuntimeError('Native success and a closed model process are required')
        start = self.tick
        actions = {uid: action.copy() for uid, action in self.last_actions.items()}
        executed = self._advance(actions, self.max_steps - self.tick, 'post_native_success_hold')
        self.packet()  # Preserve final images locally; no model receives this packet.
        save(self.output/'post_native_success_hold.json', dict(start_step=start,
            final_step=self.tick, executed_steps=executed,
            controller_actions=plain(actions), model_decisions=self.decisions))
        save(self.output/'progress.json', dict(step=self.tick, decisions=self.decisions,
            finished=self.finished, phase='complete', post_native_success_steps=self.post_native_success_steps))

    def _advance(self, actions, steps, control_source):
        executed=[]
        for _ in range(min(steps,self.max_steps-self.tick)):
            self.obs,reward,terminated,truncated,self.info=self.env.step(actions)
            self.tick+=1
            self.last_reward=float(np.asarray(plain(reward)).reshape(-1)[0])
            if not np.isfinite(self.last_reward):
                raise RuntimeError('Non-finite native reward')
            self.peak_reward=max(self.peak_reward,self.last_reward)
            self.episode_return += self.last_reward
            self.success=bool(np.asarray(plain(self.info['success'])).reshape(-1)[0])
            self.native_success_ever = self.native_success_ever or self.success
            if self.success and self.native_success_first_step is None:
                self.native_success_first_step = self.tick
            time_limit = bool(np.asarray(plain(truncated)).any()) or self.tick >= self.max_steps
            # Preserve the native predicate; stop policy actions at its first success.
            self.finished = time_limit
            self.termination_reason = 'time_limit' if time_limit else 'incomplete'
            self.record_frame()
            record=dict(step=self.tick,actions=plain(actions),eef=self.poses(),reward=self.last_reward,
                control_source=control_source,
                info=plain(self.info), native_terminated=plain(terminated), native_truncated=plain(truncated),
                simulation_state=plain(self.env.unwrapped.get_state_dict()))
            with (self.private/'trajectory.jsonl').open('a') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
            executed.append(self.tick)
            if control_source == 'model_action' and self.native_success_ever:
                self.native_success_stop = dict(step=self.tick, source='native_evaluator',
                    native_success_first_step=self.native_success_first_step,
                    hold_target=self.last_target, hold_controller_actions=plain(self.last_actions),
                    observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                save(self.output/'native_success_stop.json', self.native_success_stop)
                break  # Do not execute remaining steps of this action chunk after success.
            if control_source == 'post_native_success_hold':
                self.post_native_success_steps += 1
                if self.tick % 5 == 0 or self.finished:
                    save(self.output/'progress.json',dict(step=self.tick,decisions=self.decisions,
                        finished=self.finished,phase='post_native_success_hold',post_native_success_steps=self.post_native_success_steps))
            if self.finished:
                break
        return executed

    def outcome(self):
        return dict(complete=self.finished,
            success=self.native_success_ever,
            success_once=self.native_success_ever,
            success_at_end=self.success if self.finished else None,strict_success=None,
            review_status='pending_visual_review',native_success_at_end=self.success if self.finished else None,
            native_success_at_stop=self.success,
            native_success_ever=self.native_success_ever,native_success_first_step=self.native_success_first_step,
            steps=self.tick,decisions=self.decisions,
            policy_stop_reason='native_success' if self.native_success_stop is not None else self.termination_reason,
            native_success_stop_step=(self.native_success_stop or {}).get('step'),
            post_native_success_hold_steps=self.post_native_success_steps,
            final_normalized_reward=self.last_reward,peak_normalized_reward=self.peak_reward,
            normalized_return=self.episode_return,
            termination=self.termination_reason,
            video_frames_expected=self.tick+1)

    def close(self):
        for writer in self.writers.values():
            writer.close()
        self.env.close()
        save(self.output/'outcome.json',self.outcome())
