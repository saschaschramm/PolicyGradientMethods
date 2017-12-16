from utilities import *

class Model:
    def __init__(self, observation_space, action_space, learning_rate, discount_rate):
        self.action = None
        self.reward = None
        self.input = None
        self.policy = None
        self.value = None
        self.train_policy = None
        self.train_value = None
        self.t = None
        self.session = tf.Session()
        self.build_graph(observation_space, action_space, learning_rate, discount_rate)

    def build_graph(self, observation_space, action_space, learning_rate, discount_rate):
        self.action = tf.placeholder(tf.uint8)
        self.reward = tf.placeholder(tf.float32)
        self.t = tf.placeholder(tf.float32)

        width = observation_space[0]
        height = observation_space[1]

        self.input = tf.placeholder(tf.float32, [width, height])
        inputs_reshaped = tf.reshape(self.input, [1, width * height])
        self.value = fully_connected(inputs_reshaped, "value", 1)

        # policy
        logits = fully_connected(inputs_reshaped, "policy", action_space)
        self.policy = tf.nn.softmax(logits)
        action_mask = tf.one_hot(self.action, action_space)
        eligibility_vector = action_mask * tf.log(self.policy + 1e-13)
        parameters_policy = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
        gradients_policy = tf.gradients(eligibility_vector, parameters_policy)
        delta_policy = parameters_policy[0] + learning_rate * tf.pow(discount_rate, self.t) * (self.reward - self.value) * gradients_policy[0]
        self.train_policy = parameters_policy[0].assign(delta_policy)

        # value
        parameters_value = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value')
        gradients_value = tf.gradients(self.value, parameters_value)
        delta_value = parameters_value[0] + learning_rate * (self.reward - self.value) * gradients_value[0]
        self.train_value = parameters_value[0].assign(delta_value)
        self.session.run(tf.global_variables_initializer())

    def train(self, input, reward, action, t):
        self.session.run([self.train_policy, self.train_value], feed_dict={
                                                        self.input: input,
                                                        self.reward: reward,
                                                        self.action: action,
                                                        self.t: t})

    def predict_action(self, input):
        policy = self.session.run(self.policy, feed_dict={self.input: input})
        action = action_with_policy(policy)
        return action