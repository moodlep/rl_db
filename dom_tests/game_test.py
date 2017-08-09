import Agent
import ManualAgent
import Environment



max_rounds = 20
env = Environment.DominionEnvironment()

userA = ManualAgent.User(max_rounds, env, "userA", True)
userB = ManualAgent.User(max_rounds, env, "userB", True)

userA.play()
userB.play()

for i in range(max_rounds):
    userA.deal()
    userA.play()
    print("State after A: ", env.count_of_cards)
    userB.deal()
    userB.play()
    print("State after B: ", env.count_of_cards)

# score the game:
print("Final Scores: ")
userA.final_score()
userB.final_score()

