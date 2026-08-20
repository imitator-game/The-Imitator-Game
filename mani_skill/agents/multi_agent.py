from typing import Dict, Generic, List, TypeVar

import torch
from gymnasium import spaces

from mani_skill.agents.base_agent import BaseAgent

T = TypeVar("T")


class MultiAgent(BaseAgent, Generic[T]):
    agents: T

    def __init__(self, agents: List[BaseAgent], wrist_sensor: bool = False):
        self.agents = agents
        self.agents_dict: Dict[str, BaseAgent] = dict()
        self.scene = agents[0].scene
        self.sensor_configs = []
        for i, agent in enumerate(self.agents):
            if wrist_sensor:
                for sensor_config in agent._sensor_configs:
                    # Multi-agent robots can expose sensors with the same uid
                    # (for example, both Panda arms use "hand_camera"). Give
                    # each config a stable per-agent uid before BaseEnv parses
                    # the combined sensor list.
                    sensor_config.uid = f"{agent.uid}_{i}_{sensor_config.uid}"
                    self.sensor_configs.append(sensor_config)
            self.agents_dict[f"{agent.uid}-{i}"] = agent

    @property
    def _sensor_configs(self):
        return self.sensor_configs

    def get_proprioception(self):
        proprioception = dict()
        for i, agent in enumerate(self.agents):
            proprioception[f"{agent.uid}-{i}"] = agent.get_proprioception()
        return proprioception

    @property
    def control_mode(self):
        """Get the currently activated controller uid of each robot"""
        return {uid: agent.control_mode for uid, agent in self.agents_dict.items()}

    def set_control_mode(self, control_mode: List[str] = None):
        """Set the controller, drive properties, and reset for each agent. If given control mode is None, will set defaults"""
        if control_mode is None:
            for agent in self.agents:
                agent.set_control_mode()
        else:
            assert len(control_mode) == len(
                self.agents
            ), "For task with multiple agents, setting control mode on the MultiAgent object requires a control mode for each agent"
            for cm, agent in zip(control_mode, self.agents):
                agent.set_control_mode(cm)

    @property
    def controller(self):
        return {uid: agent.controller for uid, agent in self.agents_dict.items()}

    @property
    def action_space(self):
        return spaces.Dict(
            {uid: agent.action_space for uid, agent in self.agents_dict.items()}
        )

    @property
    def single_action_space(self):
        return spaces.Dict(
            {uid: agent.single_action_space for uid, agent in self.agents_dict.items()}
        )

    def set_action(self, action):
        """
        Set the agent's action which is to be executed in the next environment timestep
        """
        for uid, agent in self.agents_dict.items():
            agent.set_action(action[uid])

    def before_simulation_step(self):
        for agent in self.agents:
            agent.controller.before_simulation_step()

    def get_controller_state(self):
        """
        Get the state of the controller.
        """
        return {
            uid: agent.get_controller_state() for uid, agent in self.agents_dict.items()
        }

    def set_controller_state(self, state: Dict):
        for uid, agent in self.agents_dict.items():
            agent.set_controller_state(state[uid])

    # -------------------------------------------------------------------------- #
    # Other
    # -------------------------------------------------------------------------- #
    def reset(self, init_qpos=None):
        """
        Reset the robot to a rest position or a given q-position
        """
        for uid, agent in self.agents_dict.items():
            if init_qpos is not None and uid in init_qpos:
                agent.reset(init_qpos=[uid])
            else:
                agent.reset()
