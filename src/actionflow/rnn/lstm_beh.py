import tensorflow as tf

from actionflow.util import DLogger
from ..rnn.lstm_base import LSTMBase


class LSTMBeh(LSTMBase):
    def __init__(self, a_size, s_size, n_cells):
        LSTMBase.__init__(self, a_size, s_size, n_cells)

    def get_obj(self):
        return self.beh_loss
