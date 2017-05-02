import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

from gym.envs.denny.blackjack import BlackjackEnv
from gym.envs.denny import plotting

matplotlib.style.use('ggplot')




def mc_prediction(policy, env, num_episodes, discount_factor=1.0, print_debug=False):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final value function
    V = defaultdict(float)
    # defaultdict() by querying with V[state] you are adding "state" if it is not in V already.

    for i in range(num_episodes):
        # Generate an episode
        episode = generate_episode(env, print_debug)
        if print_debug: print("Episode generated: \n", episode)

        # Accumulate data about each state from the episode
        set_states = set()
        for state, action, reward, next_state in episode:
            set_states.add(state)  # need a unique list of states in the episode for V calc below.

        if print_debug: print("Return_count: \n", returns_count)
        if print_debug: print("Return_sum: \n", returns_sum)
        if print_debug: print("Set of states: \n", set_states)

        # Denny's implementation... quite different:
        for state in set_states:
            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]

        if print_debug: print("V: \n", V)
    return V

def mc_prediction_original(policy, env, num_episodes, discount_factor=1.0, print_debug=False):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final value function
    V = defaultdict(float)
    # defaultdict() by querying with V[state] you are adding "state" if it is not in V already.

    for i in range(num_episodes):
        # Generate an episode
        episode = generate_episode(env, print_debug)
        if print_debug: print ("Episode generated: \n", episode)

        # Accumulate data about each state from the episode
        set_states = set()
        for state, action, reward, next_state in episode:
            returns_count[state] +=1
            returns_sum[state] += reward # the associated reward for each episode/game the state was involved in.
            set_states.add(state) # need a unique list of states in the episode for V calc below.

        if print_debug: print("Return_count: \n", returns_count)
        if print_debug: print("Return_sum: \n", returns_sum)
        if print_debug: print("Set of states: \n", set_states)

        # Calculate V for each state in episode using: V(S) = V(S) + (1/N)(G_t - V(S))
        # Analysis: I appear to be doing every instance of state implementation, and my return calc is wrong.
        # I am also using incremental mean... which probably works for every instance...?
        # In Text every instance is covered in ch7 so stick with first instance for now.
        for state in set_states:
            if returns_count[state] > 0:
                V[state] = V[state] + (1/returns_count[state]) * (returns_sum[state] - V[state])

        if print_debug: print("V: \n", V)
    return V


def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    Observation is equivalent to state. 
    Observation includes: your score, dealer score, useable Ace boolean
    """
    score, dealer_score, usable_ace = observation
    # Cannot think of why this is better than just returning stick = 0, hit = 1 as per notebook....
    # return np.array([1.0, 0.0]) if score >= 20 else np.array([0.0, 1.0])
    # Stick (action 0) if the score is > 20, hit (action 1) otherwise
    return 0 if score >= 20 else 1

def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print("Player Score: {} (Usable Ace: {}), Dealer Score: {}".format(
          score, usable_ace, dealer_score))

def generate_episode(env, print_debug):
    observation = env.reset()
    episode = []
    while True:
        if print_debug: print_observation(observation)
        action = sample_policy(observation)
        if print_debug: print("Taking action: {}".format( ["Stick", "Hit"][action]))
        next_observation, reward, done, _ = env.step(action)
        episode.append([observation, action, reward, next_observation])
        observation = next_observation
        if done:
            if print_debug: print_observation(observation)
            if print_debug: print("Game end. Reward: {}\n".format(float(reward)))
            break
    return episode

env = BlackjackEnv()

#mc_prediction(sample_policy, env, num_episodes=10, discount_factor=1.0, print_debug=True)

# V_100 = mc_prediction(sample_policy, env, num_episodes=100)
# plotting.plot_value_function(V_100, title="100 Steps")

# V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
# plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")