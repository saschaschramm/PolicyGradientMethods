from utilities import *

class Model:
    def __init__(self, observation_space, action_space, learning_rate):
        self.action = None
        self.input = None
        self.policy = None
        self.value = None
        self.train_policy = None
        self.train_value = None
        self.td_error = None
        self.I = None
        self.session = tf.Session()
        self.build_graph(observation_space, action_space, learning_rate)

    def build_graph(self, observation_space, action_space, learning_rate):
        self.action = tf.placeholder(tf.uint8)
        self.I = tf.placeholder(tf.float32)
        self.td_error = tf.placeholder(tf.float32)

        width = observation_space[0]
        height = observation_space[1]

        self.input = tf.placeholder(tf.float32, [width, height])
        inputs_reshaped = tf.reshape(self.input, [1, width * height])
        self.value = tf.squeeze(fully_connected(inputs_reshaped, "value", 1))

        # policy
        logits = fully_connected(inputs_reshaped, "policy", action_space)
        self.policy = tf.nn.softmax(logits)
        action_mask = tf.one_hot(self.action, action_space)
        eligibility_vector = action_mask * tf.log(self.policy + 1e-13)
        parameters_policy = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
        gradients_policy = tf.gradients(eligibility_vector, parameters_policy)
        delta_policy = (parameters_policy[0] + learning_rate * self.I * self.td_error * gradients_policy[0])
        self.train_policy = parameters_policy[0].assign(delta_policy)

        # value
        parameters_value = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value')
        gradients_value = tf.gradients(self.value, parameters_value)
        delta_value = (parameters_value[0] + learning_rate * self.td_error * gradients_value[0])
        self.train_value = parameters_value[0].assign(delta_value)
        self.session.run(tf.global_variables_initializer())

    def train(self, input, td_error, action, I):
        self.session.run([self.train_policy, self.train_value], feed_dict={
            self.input: input,
            self.action: action,
            self.I: I,
            self.td_error: td_error
        })

    def predict_action(self, input):
        policy = self.session.run(self.policy, feed_dict={self.input: input})
        action = action_with_policy(policy)
        return action

    def predict_value(self, input):
        value = self.session.run(self.value, feed_dict={self.input: input})
        return value