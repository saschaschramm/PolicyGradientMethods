from reinforce_baseline import *
from catch import Catch
from utilities import discount

def main():
    global_seed(0)
    discount_rate = 0.99
    env = Catch(5)
    model = Model(env.observation_space, env.action_space, 0.1, discount_rate)
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
                    discounted_reward = discount(rewards, discount_rate, t)
                    model.train(observation, discounted_reward, action, t)
                    t += 1

                observations = []
                rewards = []
                actions = []
                break

if __name__ == '__main__':
   main()