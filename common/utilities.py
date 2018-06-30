import tensorflow as tf
import random
import numpy as np
from gym.envs.registration import register
import gym

def global_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def init_env():
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False}
    )

    env = gym.make('FrozenLakeNotSlippery-v0')
    env.seed(0)
    return env

def fully_connected(input, scope, in_size, out_size):
    with tf.variable_scope(scope):
        weights = tf.get_variable("weights", [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [out_size])
        return tf.matmul(input, weights) + bias

def action_with_policy(policy):
    rand = random.uniform(0, 1)
    cumulated_sum = np.cumsum(policy)
    for i in range(0, len(cumulated_sum)):
        if rand <= cumulated_sum[i]:
            return i
    return 0

def discount(rewards, discount_rate, t):
    discounted_reward = 0
    rewards = rewards[t:]
    for k in range(0, len(rewards)):
        reward = rewards[k]
        discounted_reward += pow(discount_rate, k) * reward
    return discounted_reward