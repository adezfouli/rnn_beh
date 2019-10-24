# This file optimises RNN using fixed initialisations.

import sys
from multiprocessing.pool import Pool
import pandas as pd
from actionflow.data.data_process import DataProcess
from actionflow.rnn.lstm_beh import LSTMBeh
from actionflow.rnn.opt_beh import OptBEH
from actionflow.util.helper import get_total_pionts
from actionflow.util.logger import LogFile, DLogger
from BD.data.data_reader import DataReader
from BD.util.paths import Paths
import tensorflow as tf

configs = []

configs.append({
    'g': 'Healthy',
    'lr': 1e-2,
    'cells': 10,
    'model_path': '../inits/rnn-init/10cells/model-final/',
    # 'model_path': None,
    'iters': 1100})

configs.append({
    'g': 'Bipolar',
    'lr': 1e-2,
    'cells': 20,
    'model_path': '../inits/rnn-init/20cells/model-final/',
    # 'model_path': None,
    'iters': 400})

configs.append({
    'g': 'Depression',
    'lr': 1e-2,
    'cells': 10,
    'model_path': '../inits/rnn-init/10cells/model-final/',
    # 'model_path': None,
    'iters': 1200})


def run_BD(i):
    tf.reset_default_graph()
    data = DataReader.read_BD()
    ncells = configs[i]['cells']
    learning_rate = configs[i]['lr']
    group = configs[i]['g']
    iters = configs[i]['iters']
    model_path = configs[i]['model_path']
    output_path = Paths.local_path + 'BD/rnn-opt-from-init/' + str(ncells) + 'cells/' + group + '/'
    with LogFile(output_path, 'run.log'):
        DLogger.logger().debug("group: " + str(group))
        gdata = data.loc[data.diag == group]
        ids = gdata['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': ids, 'train': 'test'})
        train, test = DataProcess.train_test_between_subject(gdata, pd.concat((dftr, tdftr)),
                                                             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        train = DataProcess.merge_data(train)
        DLogger.logger().debug("total points: " + str(get_total_pionts(train)))

        worker = LSTMBeh(2, 0, n_cells=ncells)
        OptBEH.optimise(worker,
                        output_path, train, None,
                        learning_rate=learning_rate, global_iters=iters,
                        load_model_path=model_path
                        )


if __name__ == '__main__':

    if len(sys.argv) == 2:
        n_proc = int(sys.argv[1])
    elif len(sys.argv) == 1:
        n_proc = 1
    else:
        raise Exception('invalid argument')

    p = Pool(n_proc)
    p.map(run_BD, range(len(configs)))
    p.close()  # no more tasks
    p.join()  # wrap up current tasks
