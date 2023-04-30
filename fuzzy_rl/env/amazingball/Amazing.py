__credits__ = ["Hanzhi Zhu, Rithvik, Conor"]
from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
import pid
import utils
import time
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class AmazingEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10):
        self.pygame = pid
        self.g = g
        self.max_angle = np.pi / 6
        self.action_space = spaces.Box(low=-self.max_angle, high=self.max_angle, shape=(2,), dtype=np.float32)
        self.world = pid.run_simulation(False)
        high = np.array([0.5, 0.3], dtype=np.float32) #dimesion of plate: 0.4, 0.25
        self.observation_space = spaces.Box(low=-high, high=high,dtype=np.float32) #dimension of the board
        
    def reset(self):
        del self.world
        self.world = pid.run_simulation(False)
        obs = self.pygame.observe()
        return obs, {}

    def step(self, action):
        angle_x, angle_y = action
        self.pygame.action(angle_x, angle_y, self.world) #where is parameter action defined?
        obs = self.pygame.observe()
        reward = self.pygame.reward()

        return obs, reward, False, False, {}
    
    def render(self):
        self.pygame.run_simulation(True)

