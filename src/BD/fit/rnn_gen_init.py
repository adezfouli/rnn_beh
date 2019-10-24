# This file generates several initialisations for RNNs.

import sys
from multiprocessing.pool import Pool
import pandas as pd
import actionflow.rnn.opt_beh as lrh
from actionflow.data.data_process import DataProcess
from actionflow.rnn.lstm_beh import LSTMBeh
from actionflow.util.helper import get_total_pionts
from actionflow.util.logger import LogFile, DLogger
from BD.data.data_reader import DataReader
from BD.util.paths import Paths
import tensorflow as tf

data = DataReader.read_BD()
configs = []

for lr in [1e-2]:
    for cells in [5, 10, 20, 30]:
        configs.append({'lr': lr, 'cells': cells})


def run_BD_RNN(i):

    tf.reset_default_graph()
    ncells = configs[i]['cells']
    lr = configs[i]['lr']
    output_path = Paths.local_path + 'BD/rnn-init/' + str(ncells) + 'cells/'
    with LogFile(output_path, 'run.log'):

        ids = data['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        train, test = DataProcess.train_test_between_subject(data, dftr,
                                                             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        train = DataProcess.merge_data(train)

        DLogger.logger().debug("total points: " + str(get_total_pionts(train)))

        worker = LSTMBeh(2, 0, n_cells=ncells)
        lrh.OptBEH.optimise(worker, output_path, train, None,
                            learning_rate=lr, global_iters=0)


def run_cv(f, n_proc):
    p = Pool(n_proc)
    p.map(f, range(len(configs)))
    p.close() # no more tasks
    p.join()  # wrap up current tasks


if __name__ == '__main__':

    if len(sys.argv) == 2:
        n_proc = int(sys.argv[1])
    elif len(sys.argv) == 1:
        n_proc = 1
    else:
        raise Exception('invalid argument')

    run_cv(run_BD_RNN, n_proc)
