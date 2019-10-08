'''Implements Q-learning'''

import gym
import gym_minigrid
import logging
import numpy as np
from collections import defaultdict
from lib import plotting
#from utils.format import get_obss_preprocessor
import torch
from scipy.stats import entropy


#ENV_NAME = 'MiniGrid-DoorKey-5x5-v0'
#ENV_NAME = 'MiniGrid-Empty-5x5-v0'
ENV_NAME = 'CartPole-v0'
#ENV_NAME = 'Taxi-v2'
env = gym.make(ENV_NAME)
#env = gym_minigrid.wrappers.FlatObsWrapper(gym.make(ENV_NAME))
logging.info(ENV_NAME)
# A function that will format the observation space for easier use
#_, preprocess_obss = get_obss_preprocessor(env.observation_space)


def make_policy(q, num_actions, epsilon):
    """Make an epsilon-greedy policy from the provided Q-function

    :param q: Q-function
    :param num_actions: Size of action space
    :param epsilon: Probability of taking a random action
    :returns: Stochastic policy as a function of the observation
    :rtype: fn: observation -> [action probability]

    """
    def policy(observation):
        best_action_idx = np.argmax(q[observation])
        distribution = []
        for action_idx in range(num_actions):
            probability = epsilon / num_actions
            if action_idx == best_action_idx:
                probability += 1 - epsilon
            distribution.append(probability)
        return distribution
    return policy


def q_learning(env, num_episodes, alpha, gamma, epsilon, *, max_entropy):
    """Find the optimal policy using off-policy Q-learning

    :param env: OpenAI environment
    :param num_episodes: Number of episodes to run
    :param alpha: Learning rate
    :param gamma: Discount factor
    :param epsilon: Probability of taking a random action
    :returns: Optimal Q-function and statistics
    :rtype: dictionary of state -> action -> action-value, plotting.EpisodeStats

    """
    statistics = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    nA = env.action_space.n
    q = defaultdict(lambda: np.zeros(nA))
    for episode_idx in range(num_episodes):
        if (episode_idx + 1) % 10 == 0:
            print("\nEpisode {}/{}"
                  .format(episode_idx + 1, num_episodes))
        observation = torch.tensor(env.reset())
        terminal = False
        t = 0
        while not terminal:
            policy = make_policy(q, env.action_space.n, epsilon)
            action_distribution = policy(observation)
            action = np.random.choice(np.arange(len(action_distribution)),
                                      p=action_distribution)
            next_observation, reward, done, _ = env.step(action)
            next_observation = torch.tensor(next_observation)
            statistics.episode_rewards[episode_idx] += reward
            statistics.episode_lengths[episode_idx] = t
            next_action_values = [q[next_observation][next_action]
                                  for next_action
                                  in np.arange(nA)]
            best_next_q = max(q[next_observation])
            entropy_bonus = entropy(action_distribution)
            if max_entropy:
                q[observation][action] += alpha * (reward + gamma * best_next_q - q[observation][action] + entropy_bonus)
            else:
                q[observation][action] += alpha * (reward + gamma * best_next_q - q[observation][action])
            if done:
                terminal = True
            else:
                observation = next_observation
                t += 1
    return q, statistics



if __name__ == '__main__':
    q, stats = q_learning(env, 500, .5, .99, .1, max_entropy=False)
    plotting.plot_episode_stats(stats)
