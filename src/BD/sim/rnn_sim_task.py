# on-policy simulations of RNNs

import os
import tensorflow as tf
from actionflow.rnn.lstm_beh import LSTMBeh
from actionflow.rnn.simulate import Simulator
from actionflow.sim.envs import bandit_evn
from actionflow.util.helper import id_generator
import pandas as pd
from BD.sim.sims import get_BD_confs
from BD.util.paths import Paths


def simulate_BD_real_task(greedy, output):

    n_cells = {'Healthy': 10, 'Depression': 10, 'Bipolar': 20}
    model_iter = 'model-final'

    confs, _, _ = get_BD_confs()

    for c in confs:
        group = c['group']
        choices = c['choices']
        output_path = Paths.local_path + output + 'sim_' + id_generator(size=9) + '/'
        input_path = Paths.rest_path + 'archive/beh/rnn-opt-from-init/' + str(n_cells[group]) + \
                     'cells/' + group + '/' + model_iter + '/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        c2 = c.copy()
        c2['N'] = c['id']
        c2['cells'] = str(n_cells[group])
        pdc = pd.DataFrame(c2, index=[0])
        pdc.to_csv(output_path + 'config.csv', index=False)

        tf.reset_default_graph()
        worker = LSTMBeh(2, 0, n_cells[group])
        Simulator.simulate_env(worker,
                               input_path,
                               output_path,
                               choices,
                               bandit_evn(c['prop0'], c['prop1'], init_state=None, init_action=-1, init_reward=0),
                               greedy=greedy,
                               export_state_track=False
                               )


if __name__ == '__main__':
    simulate_BD_real_task(False, 'BD/sims/rnn_onpolicy/')
