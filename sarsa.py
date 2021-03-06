import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys

if "../" not in sys.path:
  sys.path.append("../")

from collections import defaultdict
from gym.envs.denny.windy_gridworld import WindyGridworldEnv
from gym.envs.denny import plotting

matplotlib.style.use('ggplot')


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # generate episode
        state = env.reset()
        # get the policy for this state using the latest Q values and epsilon.
        action_prob = policy(state)
        # now choose an action using the policy distribution vs uniform distribution
        action = np.random.choice(len(action_prob), p=action_prob) # choice applies the distribution passed in

        # now take steps until the end of episode is reached and update Q[state][action]
        while True:
            # Take a step
            next_state, reward, done, prob = env.step(action)

            # Get policy for next_state using latest Q-values; choose an action using the epsilon-greedy distri., etc.
            action_prob = policy(next_state)
            next_action = np.random.choice(len(action_prob), p=action_prob)

            # Update Q
            Q[state][action] += alpha * ( (reward + discount_factor*Q[next_state][next_action]) - Q[state][action])

            # reset vars
            state = next_state
            action = next_action
            stats.episode_lengths[i_episode] +=1
            stats.episode_rewards[i_episode] += reward

            # end of episode
            if done == True:
                break

    return Q, stats

env = WindyGridworldEnv()
Q, stats = sarsa(env, 300)
print(Q)

plotting.plot_episode_stats(stats)