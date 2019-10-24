from actionflow.rnn.gru_base import GRUBase
from actionflow.util import DLogger


class GRUBeh(GRUBase):
    def __init__(self, a_size, s_size, n_cells):
        GRUBase.__init__(self, a_size, s_size, n_cells)

        DLogger.logger().debug("model created with ncells: " + str(n_cells))
        DLogger.logger().debug("number of actions: " + str(a_size))
        DLogger.logger().debug("number of states: " + str(s_size))

    def get_obj(self):
        return self.beh_loss
