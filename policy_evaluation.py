import numpy as np
from gym.envs.denny.gridworld import GridworldEnv

env = GridworldEnv()

""" 
Exercise from Denny Britz. Implementation differs but result ok
https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation.ipynb 
"""

def policy_eval_pm(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    k = 0
    while True:
        delta = 0.00
        # calculate the V for the next iteration:
        for s, v in enumerate(env.P.values()):
            v_s = 0.0
            for a, t in enumerate(v.values()):
                prob, next_state, reward, done = t[0]

                v_s += policy[s][a] * (reward + discount_factor * (prob * V[next_state]))

            delta = max(delta, np.abs(v_s - V[s]))
            V[s] = v_s

        #print("Iteration ", k, " with delta: ", delta)
        #print(V)
        k+=1

        if delta < theta:
            break

    return np.array(V)

random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval_pm(random_policy, env)


# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)


