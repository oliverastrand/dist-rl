import gym

env = gym.make("MountainCar-v0")
# Move left with 1
# Move right with 2
# Do nothing with 0

#env.render()
env.reset()

print(env.action_space)
env.render()
"""
for i in range(50):
    print(env.step(1))
env.render()
"""

for i in range(50):
    print(env.step(2))
env.render()


while True:
    pass
    

