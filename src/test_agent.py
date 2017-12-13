import gym
from src.agent import Agent
from src.learner import Learner
env = gym.make("MountainCar-v0")
options = {"env_size": (2, 4), "DNN": (5,), "env_actions": 3, "loss": "MSE"}
learner = Learner(options)
agent = Agent(learner, env)
agent.epsilon = 0.1

for i in range(1000):
    agent.step(True)
    #print("hejsan")
    #print(agent.epsilon)

print(agent.memory.experience_list)

env.render()