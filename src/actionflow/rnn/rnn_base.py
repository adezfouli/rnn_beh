from abc import abstractmethod

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from numpy.random import choice

from actionflow.util import DLogger
from ..rnn.consts import Const
from ..util.helper import normalized_columns_initializer


class RNNBase():
    def __init__(self, a_size, s_size, n_cells):

        DLogger.logger().debug("model created with ncells: " + str(n_cells))
        DLogger.logger().debug("number of actions: " + str(a_size))
        DLogger.logger().debug("number of states: " + str(s_size))

        # placeholders
        self.prev_rewards = tf.placeholder(shape=[None, None, 1], dtype=Const.FLOAT)
        # DIM: nBatches x (nChoices + 1) x 1

        self.prev_actions = tf.placeholder(shape=[None, None], dtype=tf.int32)
        # DIM: nBatches x (nChoices + 1)

        self.timestep = tf.placeholder(shape=[None, None], dtype=tf.int32)
        # DIM: nBatches x nChoices

        self.prev_actions_onehot = tf.one_hot(self.prev_actions, a_size, dtype=Const.FLOAT, axis=-1)
        # DIM: nBatches x (nChoices + 1) x nActionTypes

        self.actions = tf.placeholder(shape=[None, None], dtype=tf.int32)
        # DIM: nBatches x nChoices

        self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=Const.FLOAT, axis=-1)
        # DIM: nBatches x nChoices x nActionTypes

        self.n_batches = tf.shape(self.prev_rewards)[0]

        self.n_cells = n_cells

        if s_size != 0:
            self.prev_states = tf.placeholder(shape=[None, None, s_size], dtype=Const.FLOAT)
            # self.prev_states_onehot = tf.one_hot(self.prev_states, s_size, dtype=Const.FLOAT)
            rnn_in = tf.concat(values=[self.prev_rewards, self.prev_actions_onehot, self.prev_states], axis=2)
            # DIM: nBatches x (nChoices + 1 ) x ( 1 + nActionTypes + nStateTypes)

        else:
            rnn_in = tf.concat(values=[self.prev_rewards, self.prev_actions_onehot], axis=2)
            # DIM: nBatches x (nChoices + 1 ) x ( 1 + nActionTypes)

        self.state_init, \
        self.state_in, \
        self.rnn_out, \
        self.state_out, \
        self.state_track = self.rnn_cells(n_cells, rnn_in)

        # Output layers for policy and value estimations
        self.policy = slim.fully_connected(self.rnn_out, a_size,
                                           activation_fn=tf.nn.softmax,
                                           weights_initializer=normalized_columns_initializer(0.01),
                                           biases_initializer=None,
                                           scope='softmax')
        # DIM: nBatches x (nChoices + 1) x (nActionTypes)

        # self.value = slim.fully_connected(self.rnn_out, 1,
        #                                   activation_fn=None,
        #                                   weights_initializer=normalized_columns_initializer(1.0),
        #                                   biases_initializer=None,
        #                                   sope='softmax-v')

        self.beh_loss = self._get_beh_loss()

    def get_ncells(self):
        return self.n_cells

    def get_nbatches(self):
        return self.n_batches

    def rnn_cells(self, n_cells, rnn_in):
        raise Exception("method undefined")

    def get_rnn_out(self):
        # note we ignore the first element since there is one extra action added at the beginning
        return tf.transpose(self.rnn_out[:, 1:, :], [0, 2, 1])

    def _get_beh_loss(self):

        # this is correction for the missing actions for which the one hot is [0,0, ... 0]
        actions_onehot_corrected = (1 - tf.reduce_sum(self.actions_onehot, axis=2))[:, :, np.newaxis] + \
                                   self.actions_onehot
        # note that we ignore the last element of the policy, since there is no observation after
        # the last reward received.
        action_logs = tf.reduce_sum(tf.log(tf.reduce_sum(self.policy[:, :-1, :] *
                                                         actions_onehot_corrected, axis=2)), axis=[1])
        # uncomment this to calcualte the mean over each datapoint
        # action_logs = tf.div(action_logs, tf.reduce_sum(self.actions_onehot, axis=[1, 2]))

        return -tf.reduce_sum(action_logs)

    def simulate(self, sess, rewards, actions, states):
        """
        N:  number of batches
        T:  number of actions (trials)
        n_a:number of different actions
        n_c:number of cells


        Args:
            sess: TensorFlow session
            rewards:    dim: N x T
            actions:    dim: N x T
            states:     dim: N x T
            timesteps:  dim: N x T

        Returns:
            policy: dim: N x (T+1) x n_a

                (rewards t-1, actions t-1, states t) => model => (policy t)

                policy[:,t]: policy after receiving rewards and actions 0...(t-1) at state s_t

                For example, policy[:,0] corresponds to the probability that each actions will be selected in
                the first trial, i.e., policy[:, 0] corresponds to actions[:,0]. For calculating this
                policy at the very first trial, since there is no observation before that,
                a dummy trials will be added to the data before the first trial. policy[:,T] is at the end of
                learning and does not correspond to any action.

            c_track: dim: N x (T+1) x n_c

                c_track[:,t]: c state track after receiving actions, rewards 0...t and s_t+1

                For example, c_track[:,0, i] corresponds to the ``c'' output of cell ``i'' after receiving the
                first action/reward,
                i.e., it corresponds to the output of the cell at the beginning of the second trials after observing
                the state in the second trial.

            h_track: dim: N x (T+1) x n_c
                similar to ``c_track''
        """

        feed_dict = self.beh_feed(actions, rewards, states)

        policy, c_track, h_track, beh_loss = sess.run([self.policy,
                                                       self.get_c_track(),
                                                       self.get_h_track(),
                                                       self.beh_loss
                                                       ], feed_dict=feed_dict)

        return policy, c_track, h_track, beh_loss

    def dH_dReward(self, H_trials):
        """
        Calculates the gradients of rnn_out at rnnout_trials wrt to the rewards received (for all trials)
        :param H_trials: the array/list containing index of the action for which the gradinet of the cell
                output is calculated.
        :return: an array of size nCells x len(rnnout_trials) x nTrials
                    for example output [c, t1, t2] is the gradient of output of
                    cell c at trial t1 (after receiving reward) wrt to the reward received at
                    trial t2.
        """
        track = self.rnn_out
        nCells = self.get_ncells()
        grads = []
        for c in range(nCells):
            cell_grads = []
            for a in H_trials:
                cell_grads.append(tf.gradients(track[0, a, c], self.get_prev_rewards())[0][0, :, 0])
            grads.append(cell_grads)
        grads = tf.convert_to_tensor(grads)
        return grads

    def dPolicy_dReward(self, policy_trials):
        """
        Calculates the gradient of policies wrt to rewards
        :param policy_trials: the array/list containing index of the action for which the gradinet of the cell
                output is calculated.
        :return: an array of size len(trial_list) x nTrials
                    for example output [t1, t2] is the gradient of policy at trial t1 wrt to reward received at trial t2
        """
        DLogger.logger().debug("adding gradients for each cell-action pair...")
        grads = []
        for a in policy_trials:
            grads.append(tf.gradients(self.policy[0, a, 0], self.prev_rewards)[0][0, :, 0])
        grads = tf.convert_to_tensor(grads)
        DLogger.logger().debug("finished adding gradients.")
        return grads

    @abstractmethod
    def get_prev_rewards(self):
        return self.prev_rewards

    @abstractmethod
    def beh_feed(self, actions, rewards, states):
        return NotImplemented

    @abstractmethod
    def simulate_env(self, sess, max_time, env_model, greedy=False, cell_mask=None):
        return NotImplemented

    @abstractmethod
    def get_h_track(self):
        # DIM: nBatches x nCells x nChoices
        return NotImplemented

    @abstractmethod
    def get_c_track(self):
        # DIM: nBatches x nCells x nChoices
        return NotImplemented