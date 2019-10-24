import tensorflow as tf
import numpy as np
from numpy.random import choice
from actionflow.rnn.rnn_base import RNNBase
from ..rnn.consts import Const


class LSTMBase(RNNBase):
    def __init__(self, a_size, s_size, n_cells):
        RNNBase.__init__(self, a_size, s_size, n_cells)

    def rnn_cells(self, n_cells, rnn_in):
        # RNN component
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_cells, state_is_tuple=True)

        # init input to the model
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        state_init = [c_init, h_init]

        # init state of the RNN
        c_in = tf.placeholder(Const.FLOAT, [None, lstm_cell.state_size.c])
        h_in = tf.placeholder(Const.FLOAT, [None, lstm_cell.state_size.h])
        state_in = (c_in, h_in)
        step_size = tf.ones([tf.shape(self.prev_actions)[0]], dtype=tf.int32) * tf.shape(self.prev_actions)[1]

        rnn_out, state_out, lstm_state_track = self.run_rnn(c_in, h_in, lstm_cell, n_cells, rnn_in, step_size)
        # DIM rnn_out: nBatches x (nChoices + 1) x nCells
        # DIM state_out: 2 * nBatches x nCells
        # DIM lstm_output_track: nBatches x (nChoices + 1) x nCells  - lstm_output_track = rnn_out
        # DIM lstm_state_track: nBatches x 2 x (nChoices + 1) x nCells - [:,0,:,:]=>c , [:,1,:,:]=>h

        return state_init, state_in, rnn_out, state_out, lstm_state_track

    def run_rnn(self, c_in, h_in, lstm_cell, n_cells, rnn_in, step_size):
        from tensorflow.contrib.rnn import LSTMStateTuple

        def learn(curr_trial, state_track, output_track):
            state_in = state_track[:, :, -1, :]
            input_in = rnn_in[:, curr_trial, :]
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, input_in[:, np.newaxis, :], initial_state=LSTMStateTuple(state_in[:, 0], state_in[:, 1]),
                time_major=False)

            return tf.add(curr_trial, 1), \
                   tf.concat((state_track, tf.stack(lstm_state, axis=1)[:, :, np.newaxis, :]), axis=2), \
                   tf.concat((output_track, lstm_outputs), axis=1)

        def cond(curr_trial, _1, _2):
            return tf.less(curr_trial, step_size[0])

        while_output = tf.while_loop(cond, learn,
                                     loop_vars=[tf.constant(0),
                                                tf.stack(LSTMStateTuple(c_in, h_in), axis=1)[:, :, np.newaxis, :],
                                                tf.zeros((self.n_batches, 1, n_cells), dtype=Const.FLOAT)
                                                ],
                                     shape_invariants=[tf.constant(0).get_shape(),
                                                       tf.TensorShape([None, 2, None, n_cells]),
                                                       tf.TensorShape([None, None, n_cells])
                                                       ]
                                     )
        _, state_track, output_track = while_output

        # remove the first dummy variables
        rnn_out = output_track[:, 1:]
        lstm_c, lstm_h = state_track[:, 0, -1, :], state_track[:, 1, -1, :]
        state_out = (lstm_c, lstm_h)

        # remove the first variable
        lstm_state_track = state_track[:, :, 1:, :]

        return rnn_out, state_out, lstm_state_track

    # this one runs using TF builtin RNN runner, however, it does not
    # exposre track of c and h - so it is reimplemented
    # def run_rnn(self, c_in, h_in, lstm_cell, n_cells, rnn_in, step_size):
    #     state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
    #     lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
    #         lstm_cell, rnn_in, initial_state=state_in,
    #         time_major=False, sequence_length=step_size)
    #     lstm_c, lstm_h = lstm_state
    #     state_out = (lstm_c[:1, :], lstm_h[:1, :])
    #     rnn_out = lstm_outputs
    #     return rnn_out, state_out

    def beh_feed(self, actions, rewards, states):
        """
        Created a dict for TensorFlow by adding a dummy action and reward to the beginning of
        actions and rewards and a dummy state to the end of states
        """

        prev_rewards = np.hstack((np.zeros((rewards.shape[0], 1)), rewards))
        prev_actions = np.hstack((-1 * np.ones((actions.shape[0], 1)), actions))
        rnn_state = self.state_init
        feed_dict = {self.prev_rewards: prev_rewards[:, :, np.newaxis],
                     self.prev_actions: prev_actions,
                     self.actions: actions,
                     self.state_in[0]: np.repeat(rnn_state[0], rewards.shape[0], axis=0),
                     self.state_in[1]: np.repeat(rnn_state[1], rewards.shape[0], axis=0)}
        if states is not None:
            prev_states = np.hstack((states, np.zeros(states[:, 0:1].shape)))
            feed_dict[self.prev_states] = prev_states
        return feed_dict

    def simulate_env(self, sess, max_time, env_model, greedy=False, cell_mask=None):
        rnn_state_lists = []
        policy_lists = []
        r_lists = []
        state_lists = []
        actions_list = []

        if cell_mask is None:
            cell_mask = [np.ones((1, self.state_in[0].shape[1])), np.ones((1, self.state_in[1].shape[1]))]

        rnn_state = self.state_init
        s, r, a = env_model(None, None, -1)
        for t in range(max_time + 1):
            s, r, next_a = env_model(s, a, t)
            feed_dict = {
                self.prev_rewards: [[[r]]],
                self.timestep: [[t]],
                self.prev_actions: [[a]],
                self.state_in[0]: rnn_state[0],
                self.state_in[1]: rnn_state[1]}

            if s is not None:
                feed_dict[self.prev_states] = [[s]]

            a_dist, rnn_state_new = sess.run([self.policy, self.state_out], feed_dict=feed_dict)
            rnn_state_new = [rnn_state_new[0] * cell_mask[0], rnn_state_new[1] * cell_mask[1]]
            rnn_state = rnn_state_new
            rnn_state_lists.append(rnn_state_new)
            policy_lists.append(a_dist)
            r_lists.append(r)
            state_lists.append(s)
            actions_list.append(a)

            if next_a is not None:
                a = next_a
            elif greedy:
                a = np.argmax(a_dist)
            else:
                a = choice(np.arange(a_dist.shape[2]), p=a_dist[0, 0])

        # note that policy is for the next trial
        return np.hstack(state_lists)[:-1], \
               np.hstack(policy_lists)[0, :-1], \
               np.hstack(r_lists)[1:], \
               np.hstack(actions_list)[1:], \
               np.hstack(rnn_state_lists)[:, 1:]

    def get_c_track(self):
        # note we ignore the first element since there is one extra action added at the beginning
        return tf.transpose(self.state_track[:, 0, 1:, :], [0, 2, 1])

    def get_h_track(self):
        # note we ignore the first element since there is one extra action added at the beginning
        return tf.transpose(self.state_track[:, 1, 1:, :], [0, 2, 1])
