import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../")
from gym.envs.denny.gridworld import GridworldEnv

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])  # policy started from zero position

    while True:
        delta = 0.0

        # Evaluate the value function for 1 iteration using Bellman's optimality eqn
        for s in range(env.nS):
            v = np.zeros(env.nA)

            for a, a_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v[a] = prob * (reward + discount_factor * V[next_state])

            # At the state level: Update V and improve policy immediately
            v_max = v[v.argmax()]
            delta = max(delta, abs(v_max-V[s]))
            V[s] = v_max
            policy[s] = np.zeros(env.nA)
            policy[s][v.argmax()] = 1.0

        if delta < theta:
            break

    # Implement!
    return policy, V



pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)