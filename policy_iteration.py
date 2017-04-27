import numpy as np
from gym.envs.denny.gridworld import GridworldEnv
from policy_evaluation import policy_eval_pm

env = GridworldEnv()

def policy_improvement_pm(env, policy_eval_fn=policy_eval_pm, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    iterations = 0

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA

    # Capture the Q values
    Q = np.zeros([env.nS, env.nA])

    while True:

        iterations +=1
        # get the value function for this policy
        print("Policy Eval {} - Started".format(iterations))
        v = policy_eval_fn(policy, env)
        print("Policy Eval - Done")
        print (v)

        # set a flag to determine stability of this policy
        optimal_policy_flag = True

        # for each state estimate the action value and decide which action is best for this state:
        for s, v_s in enumerate(v):

            q = np.zeros(env.nA)

            # calc the state,action values (q) for this state. In this case q[no_of_actions] or q[4]
            for a, action_prob in enumerate(policy[s]):
                # For each action, calculate the action-value q
                for prob, next_state, reward, done in env.P[s][a]:
                    q[a] = prob * (reward + discount_factor * v[next_state])

            # Improve Policy for every state, every iteration
            # for this state s, choose the highest q value. This is the greedy action to follow
            best_action = np.argmax(q)
            current_action = np.argmax(policy[s])
            policy[s] = np.zeros(env.nA)
            policy[s][best_action] = 1.0

            # Print variables
            Q[s] = q

            #Check stopping condition: if no improvements this iteration can stop
            if best_action != current_action:
                # policy is still being improved so don't stop yet
                optimal_policy_flag = False

        print("Updated Policy this iteration: \n", policy)

        if optimal_policy_flag:
            print("Optimal Policy achieved in {} iterations".format(iterations))
            break

    return policy, v, Q

# Tests:

policy, v , q = policy_improvement_pm(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Action Value Function:")
print(q)
print("")

#Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

