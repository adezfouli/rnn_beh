__author__ = 'AD'

from actionflow.qrl.consts import Const
import tensorflow as tf
from actionflow.qrl.ql import logit
import numpy as np
from actionflow.qrl.rl import RL


class GQL(RL):
    def __init__(self, n_actions, params, options=None):

        if options is None:
            options = {}

        self.n_actions = n_actions
        self.options = options

        # input data
        self.actions = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, n_actions, dtype=tf.float32, axis=-1)
        self.prev_actions = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.prev_actions_onehot = tf.one_hot(self.prev_actions, n_actions, dtype=tf.float32, axis=-1)
        self.prev_reward = tf.placeholder(shape=[None, None, 1], dtype=Const.FLOAT)

        # model variables
        self.logit_alpha = None
        self.logit_choice_alpha = None
        self.coef_value = None
        self.coef_choice = None
        self.coef_interaction = None

        # for internal use
        self.init_action = tf.placeholder(shape=[None, None], dtype=tf.int32)
        self.init_trial = tf.constant(0, dtype=tf.int32)
        self.init_ll = tf.constant(0.0, dtype=Const.FLOAT)
        self.init_policies = self.init_policies = tf.zeros_like(self.actions_onehot)[:, 0:1, :]

        self.n_v = params[0].shape[0]
        self.n_c = params[1].shape[0]

        # initial state of the model
        self.init_values = tf.placeholder(shape=[None, self.n_actions, self.n_v], dtype=Const.FLOAT)
        self.init_choice_kernel = tf.placeholder(shape=[None, self.n_actions, self.n_c], dtype=Const.FLOAT)

        if self.with_nointeraction():
            self.n_params = 4
        else:
            self.n_params = 5

        self._simulate = None
        self._ell = None
        self.set_params(params)

    def beh_feed(self, actions, rewards, init_values=None):

        prev_rewards = np.hstack((np.zeros((rewards.shape[0], 1)), rewards))
        prev_actions = np.hstack((-1 * np.ones((actions.shape[0], 1)), actions))
        feed_dict = {self.prev_reward: prev_rewards[:, :, np.newaxis],
                     self.prev_actions: prev_actions,
                     self.actions: actions,
                     self.init_values: np.zeros([actions.shape[0], self.init_values.shape[1], self.init_values.shape[2]]),
                     self.init_choice_kernel: np.zeros([actions.shape[0], self.init_choice_kernel.shape[1], self.init_choice_kernel.shape[2]])
                     }

        return feed_dict

    def simulate(self, sess, reward, choices, init_values=None):

        beh_feed = self.beh_feed(choices, reward, init_values)
        _, values, ell, policies, choice_kernel = sess.run(self._simulate, feed_dict=beh_feed)
        return ell, policies, [values, choice_kernel]

    def get_transformed_params(self):
        if not self.with_nointeraction():
            return [tf.sigmoid(self.logit_alpha),
                    tf.sigmoid(self.logit_choice_alpha),
                    self.coef_value,
                    self.coef_choice,
                    self.coef_interaction]
        else:
            return [tf.sigmoid(self.logit_alpha),
                    tf.sigmoid(self.logit_choice_alpha),
                    self.coef_value,
                    self.coef_choice]

    def get_trans_func(self):
        if not self.with_nointeraction():
            return [tf.sigmoid,
                    tf.sigmoid,
                    tf.identity,
                    tf.identity,
                    tf.identity]
        else:
            return [tf.sigmoid,
                    tf.sigmoid,
                    tf.identity,
                    tf.identity]

    def with_nointeraction(self):
        return 'no-interac' in self.options and self.options['no-interac']

    def set_params(self, params):

        if self.with_nointeraction():
            self.logit_alpha = params[0]
            self.logit_choice_alpha = params[1]
            self.coef_value = params[2]
            self.coef_choice = params[3]
            self.coef_interaction = tf.zeros(params[4].shape, dtype=Const.FLOAT)

        else:
            self.logit_alpha = params[0]
            self.logit_choice_alpha = params[1]
            self.coef_value = params[2]
            self.coef_choice = params[3]
            self.coef_interaction = params[4]

        self._simulate = self._simulate_()
        self._ell = self._simulate[2]

    def get_params(self):
        if self.with_nointeraction():
            return [
                self.logit_alpha,
                self.logit_choice_alpha,
                self.coef_value,
                self.coef_choice,
                ]
        else:
            return [
                self.logit_alpha,
                self.logit_choice_alpha,
                self.coef_value,
                self.coef_choice,
                self.coef_interaction
                ]

    def ell(self):
        return self._ell

    def get_obj(self):
        return -self._ell

    def get_policy(self):
        return self._simulate[3]

    def get_param_names(self):
        if not self.with_nointeraction():
            return ['alpha', 'choice_alpha', 'coef_value', 'coef_choice', 'coef_interaction']
        else:
            return ['alpha', 'choice_alpha', 'coef_value', 'coef_choice']

    def _simulate_(self):
        alpha = tf.sigmoid(self.logit_alpha)
        choice_alpha = tf.sigmoid(self.logit_choice_alpha)

        choices = self.prev_actions_onehot

        def learn(curr_trial, old_value, old_choice_kernel, policies):
            action = choices[:, curr_trial,]

            r = self.prev_reward[:, curr_trial, :]
            error_signal = r[:, :, np.newaxis] * action[:, :, np.newaxis] - old_value

            new_value = old_value + action[:, :, np.newaxis] * (error_signal * alpha)

            choice_kernel_error_signal = action[:, :, np.newaxis] - old_choice_kernel
            new_choice_kernel = old_choice_kernel + choice_kernel_error_signal * choice_alpha

            interac = tf.matmul(new_choice_kernel[:, :, :, np.newaxis], new_value[:, :, np.newaxis])

            x = tf.reduce_sum(interac * self.coef_interaction, [2, 3]) + \
                tf.reduce_sum(new_value * self.coef_value, [2]) + \
                tf.reduce_sum(new_choice_kernel * self.coef_choice, [2])

            policy = tf.nn.log_softmax(x)

            # new_choice_kernel = new_choice_kernel * (1.0 - r)

            return tf.add(curr_trial, 1), new_value, new_choice_kernel, tf.concat([policies, policy[:, np.newaxis, :]], 1)

        def cond(curr_trial, old_value, old_choice_kernel, policies):
            return tf.less(curr_trial, tf.shape(self.prev_reward)[1])

        while_otput = tf.while_loop(cond, learn, [self.init_trial,
                                                  self.init_values,
                                                  self.init_choice_kernel,
                                                  self.init_policies],
                                    shape_invariants=[self.init_trial.get_shape(),
                                                      self.init_values.get_shape(),
                                                      self.init_choice_kernel.get_shape(),
                                                      tf.TensorShape([None, None, self.n_actions])]
                                    )

        ll = tf.reduce_sum(while_otput[3][:, 1:-1, ] * self.actions_onehot)

        return [while_otput[0], while_otput[1], ll, while_otput[3][:, 1:, ], while_otput[2]]

    def sim_output(self, sess, rewards, actions, pre_step_vars):
        values = pre_step_vars[0]
        choice_kernel = pre_step_vars[1]
        feed_dict = {self.prev_reward: rewards[:, :, np.newaxis],
                     self.prev_actions: actions,
                     self.actions: np.array([[-1]]),
                     self.init_values: values,
                     self.init_choice_kernel: choice_kernel
                     }

        _, values, _, a_dist, choice_kernel = sess.run(self._simulate, feed_dict=feed_dict)
        return a_dist, [values, choice_kernel]

    def init_sim_vars(self):
        values = np.zeros([1, self.init_values.shape[1], self.init_values.shape[2]])
        choice_kernel = np.zeros([1, self.init_choice_kernel.shape[1], self.init_choice_kernel.shape[2]])
        pre_step_vars = [values, choice_kernel]
        return pre_step_vars

    @staticmethod
    def get_instance(n_actions, degree, options):
        return GQL(
            n_actions,
            [logit(tf.constant(np.random.uniform(0, 1, degree), dtype=Const.FLOAT)),
             logit(tf.constant(np.random.uniform(0, 1, degree), dtype=Const.FLOAT)),
             tf.constant(np.random.normal(0, 3, degree), dtype=Const.FLOAT),
             tf.constant(np.random.normal(0, 3, degree), dtype=Const.FLOAT),
             tf.constant(0.0 * np.random.normal(0, 3, [degree, degree]), dtype=Const.FLOAT),
             ],
            options=options)
