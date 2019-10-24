__author__ = 'AD'

from actionflow.util import DLogger
from actionflow.util.helper import Helper
from actionflow.qrl.rl import RL
from consts import Const
import tensorflow as tf
import numpy as np


def logit(x):
    return tf.log(x) - tf.log(tf.constant(1.0, Const.FLOAT) - x)


class Config:
    PERSV = 'persv'


class QL(RL):
    def __init__(self, n_actions, params, options=None):

        # input data to the model
        if options is None:
            options = {}
        self.options = options

        self.actions = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, n_actions, dtype=tf.float32, axis=-1)
        self.prev_actions = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.prev_actions_onehot = tf.one_hot(self.prev_actions, n_actions, dtype=tf.float32, axis=-1)
        self.n_actions = n_actions
        self.prev_reward = tf.placeholder(shape=[None, None, 1], dtype=Const.FLOAT)

        # initial condition of the model
        self.init_values = tf.placeholder(shape=[None, self.n_actions], dtype=Const.FLOAT)
        self.init_action = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.init_trial = tf.constant(0, dtype=tf.int32)

        # model parameters
        if self.is_persev():
            self.n_params = 3
        else:
            self.n_params = 2

        self.logit_alpha = None
        self.log_gamma = None
        self.persev = None

        # for internal use
        self.init_ll = tf.constant(0.0, dtype=Const.FLOAT)
        self.init_policies = tf.zeros_like(self.actions_onehot)[:, 0:1, :]

        self._simulate = None
        self._ell = None
        self.set_params(params)

    def set_params(self, params):
        if len(params) != self.n_params:
            raise Exception('wrong number of parameters')

        if self.is_persev():
            self.logit_alpha = params[0]
            self.log_gamma = params[1]
            self.persev = params[2]
        else:
            self.logit_alpha = params[0]
            self.log_gamma = params[1]
            self.persev = tf.constant(0.0, Const.FLOAT)

        self._simulate = self._simulate_()
        self._ell = self._simulate[2]

    def get_params(self):
        if self.is_persev():
            return [self.logit_alpha, self.log_gamma, self.persev]
        else:
            return [self.logit_alpha, self.log_gamma]

    def ell(self):
        return self._ell

    def get_obj(self):
        return -self._ell

    def beh_feed(self, actions, rewards, init_values=None):
        """
        Created a dict for TensorFlow by adding a dummy action and reward to the beginning of
        actions and rewards and a dummy state to the end of states
        """

        if init_values is None:
            init_values = np.zeros((actions.shape[0], self.n_actions))

        prev_rewards = np.hstack((np.zeros((rewards.shape[0], 1)), rewards))
        prev_actions = np.hstack((-1 * np.ones((actions.shape[0], 1)), actions))
        feed_dict = {self.prev_reward: prev_rewards[:, :, np.newaxis],
                     self.prev_actions: prev_actions,
                     self.actions: actions,
                     self.init_values: init_values
                     }

        return feed_dict

    def simulate(self, sess, reward, choices, init_values=None):

        beh_feed = self.beh_feed(choices, reward, init_values)
        _, values, ell, policies = sess.run(self._simulate, feed_dict=beh_feed)
        return ell, policies, values

    def get_transformed_params(self):
        if self.is_persev():
            return [tf.sigmoid(self.logit_alpha), tf.exp(self.log_gamma), self.persev]
        else:
            return [tf.sigmoid(self.logit_alpha), tf.exp(self.log_gamma)]

    def get_trans_func(self):
        if self.is_persev():
            return [tf.sigmoid, tf.exp, tf.identity]
        else:
            return [tf.sigmoid, tf.exp]

    def _simulate_(self):
        alpha = tf.sigmoid(self.logit_alpha)
        gamma = tf.exp(self.log_gamma)

        # adding a dummy element to the beginning
        choices = self.prev_actions_onehot

        def learn(curr_trial, old_value, policies):
            action = choices[:, curr_trial, ]
            error_signal = self.prev_reward[:, curr_trial, :] * action - old_value
            new_value = old_value + action * alpha * error_signal
            policy = tf.nn.log_softmax(gamma * new_value + self.persev * action)
            return tf.add(curr_trial, 1), new_value, tf.concat([policies, policy[:, np.newaxis, :]], 1)

        def cond(curr_trial, old_value, policies):
            return tf.less(curr_trial, tf.shape(self.prev_reward)[1])

        while_otput = tf.while_loop(cond, learn, [self.init_trial, self.init_values, self.init_policies],
                                    shape_invariants=[self.init_trial.get_shape(),
                                                      self.init_values.get_shape(),
                                                      tf.TensorShape([None, None, self.n_actions])]
                                    )

        ll = tf.reduce_sum(while_otput[2][:, 1:-1, ] * self.actions_onehot)

        return [while_otput[0], while_otput[1], ll, while_otput[2][:, 1:]]

    def get_policy(self):
        return self._simulate[3]

    def is_persev(self):
        if Config.PERSV in self.options:
            if not self.options[Config.PERSV]:
                return False
        return True

    def sim_output(self, sess, rewards, actions, pre_step_vars):
        values = pre_step_vars[0]
        feed_dict = {self.prev_reward: rewards[:, :, np.newaxis],
                     self.prev_actions: actions,
                     self.actions: np.array([[-1]]),
                     self.init_values: np.array([values])
                     }

        _, values, _, a_dist = sess.run(self._simulate, feed_dict=feed_dict)
        return a_dist, values

    def init_sim_vars(self):
        values = np.zeros(self.n_actions)
        pre_step_vars = [values]
        return pre_step_vars

    def get_param_names(self):
        if self.is_persev():
            return ['alpha', 'gamma', 'persv']
        else:
            return ['alpha', 'gamma']

    @staticmethod
    def get_instance_with_pser(n_actions, alpha, gamma, persev):
        options = {'persv': True}
        DLogger.logger().debug("options: " + str(Helper.dicstr(options)))

        return QL(n_actions, [
            logit(tf.constant(alpha, dtype=Const.FLOAT)),
            tf.log(tf.constant(gamma, dtype=Const.FLOAT)),
            tf.constant(persev, dtype=Const.FLOAT)
        ], options)

    @staticmethod
    def get_instance_without_pser(n_actions, alpha, gamma):
        options = {'persv': False}
        DLogger.logger().debug("options: " + str(Helper.dicstr(options)))
        return QL(n_actions, [
            logit(tf.constant(alpha, dtype=Const.FLOAT)),
            tf.log(tf.constant(gamma, dtype=Const.FLOAT))
        ], options)


# numpy implementation
def numpy_impl(choices, reward):
    v0 = 0
    v1 = 0
    alpha = 0.3
    gamma = 0.4
    persv = 0.4
    vs = [[v0, v1]]
    policy = []
    ell = 0
    for j in range(len(reward)):

        if j > 0:
            p1 = np.exp(v0 * gamma + (1 - choices[j - 1]) * persv)
            p2 = np.exp(v1 * gamma + (choices[j - 1]) * persv)
        else:
            p1 = np.exp(v0 * gamma)
            p2 = np.exp(v1 * gamma)

        policy.append([p1 / (p1 + p2), p2 / (p1 + p2)])
        if choices[j] == 0:
            d = reward[j] - v0
            v0 += alpha * d
            ell += np.log(p1) - np.log(p1 + p2)
        else:
            d = reward[j] - v1
            v1 += alpha * d
            ell += np.log(p2) - np.log(p1 + p2)
        vs.append([v0, v1])
    return np.array(vs), np.array(policy), ell
