from reinforce_actor_critic import *
from catch import Catch

discount_rate = 0.99

def main():
    env = Catch(5)
    model = Model(env.observation_space, env.action_space, 0.01)
    epsiodes = 10000

    for episode in range(0, epsiodes):

        I = 1
        observation = env.reset()
        last_value = model.predict_value(observation)

        while True:
            action = model.predict_action(observation) # S_t
            observation2, reward, done = env.step(action) # S_t+1, R_t+1

            print_score(reward, 10000)
            value = model.predict_value(observation2) # v(S_t+1)

            td_target = reward + discount_rate * value
            td_error = td_target - last_value
            model.train(observation, td_error, action, I)

            observation = observation2
            last_value = value
            I = discount_rate * I

            if done:
                break

if __name__ == '__main__':
   main()