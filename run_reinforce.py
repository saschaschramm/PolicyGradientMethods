"""
Example with T = 3:
S0, A0 -> R1
S1, A1 -> R2
S2, A2 -> R3

t = 0: G0 = discount_rate^0 * R1 + discount_rate^1 * R2 + discount_rate^2 * R3
t = 1: G1 = discount_rate^0 * R2 + discount_rate^1 * R3
t = 2: G2 = discount_rate^0 * R3

t = 0: discount_rate^0 * G0
t = 1: discount_rate^1 * G1
t = 2: discount_rate^2 * G2
"""

from reinforce import *
from catch import Catch
from utilities import discount

def main():
    global_seed(0)
    discount_rate = 0.99
    env = Catch(5)
    model = Model(env.observation_space, env.action_space, 0.1)
    observations = []
    rewards = []
    actions = []
    epsiodes = 10000

    for episode in range(0, epsiodes):
        observation = env.reset()

        while True:
            action = model.predict_action(observation)
            observations.append(observation)
            observation, reward, done = env.step(action)
            rewards.append(reward)
            actions.append(action)
            print_score(reward, 10000)

            if done:
                t = 0
                for observation, reward, action in zip(observations, rewards, actions):
                    discounted_reward = pow(discount_rate, t) * discount(rewards, discount_rate, t)
                    model.train(observation, discounted_reward, action)
                    t += 1

                observations = []
                rewards = []
                actions = []
                break

if __name__ == '__main__':
   main()





