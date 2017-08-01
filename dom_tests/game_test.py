import Agent
import Environment



max_rounds = 16
env = Environment.DominionEnvironment()

userA = Agent.User(max_rounds, env, "userA")
userB = Agent.User(max_rounds, env, "userB")

userA.play()
userB.play()

for i in range(max_rounds):
    userA.deal()
    userA.play()

    userB.deal()
    userB.play()

# score the game:
print("Final Scores: ")
userA.final_score()
userB.final_score()
