''' Implements Q-learning '''

import gym
import gym_minigrid
import logging

ENV_NAME = 'MiniGrid-DoorKey-5x5-v0'

env = gym.make(ENV_NAME)
logging.info(ENV_NAME)
print(env.action_space)
print(env.observation_space)
