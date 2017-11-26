import tensorflow as tf
import random
import numpy as np

def fully_connected(input, scope, out_size):
    with tf.variable_scope(scope):
        in_size = input.get_shape()[1].value
        weights = tf.get_variable("weights", [in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        return tf.matmul(input, weights)

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


step_count = 0
scores = []
def print_score(reward, limit):
    global scores
    global step_count
    scores.append(reward)

    if len(scores) == limit:
        print("{}: {}".format(step_count, sum(scores)))
        step_count += 1
        scores = []