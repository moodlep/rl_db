
'''
MC Control snippet: discounted return from timestep t or first occurence to end of episode
Calculating the first occurence of a state in an episode and then calculating the discounted return. 
Requires: 
    episode[(state, reward, next state, done)]
    discount_factor
    set of unique states in the episode
'''
# some sample data to run with:
episode = [[(12, 4, False), 1, 0, (19, 4, False)],
           [(19, 4, False), 1, -1, (27, 4, False)],
           [(12, 4, False), 1, 0, (19, 4, False)],
           [(8, 6, False), 1, -1, (27, 4, False)] ]
discount_factor = 1.0
state = (12, 4, False)

# Find the first occurance of the state in the episode
first_occurence_idx = next(i for i ,x in enumerate(episode) if x[0] == state)
# Sum up all rewards since the first occurance
G = sum([x[2 ] *( discount_factor **i) for i ,x in enumerate(episode[first_occurence_idx:])])




# SARSA snipper: create Q[state][action] = q-value
from collections import defaultdict
import numpy as np
from gym.envs.denny.windy_gridworld import WindyGridworldEnv
env = WindyGridworldEnv()

Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Useful instantiations:
# Creating an empty dict of lists for each action:
# IN: {a: [] for a in range(nA)}    .... where nA = 4
# OUT: {0: [], 1: [], 2: [], 3: []}
