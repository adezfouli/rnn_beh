# This file optimises QL and QLP using all the data in each group using maximum likelihood method (ML).

import sys
from multiprocessing.pool import Pool
import pandas as pd
from actionflow.data.data_process import DataProcess
from actionflow.qrl.opt_ml import OptML
from actionflow.qrl.ql import QL
from actionflow.util.helper import get_total_pionts
from actionflow.util.logger import LogFile, DLogger
from BD.data.data_reader import DataReader
from BD.util.paths import Paths
import tensorflow as tf

configs = []
for group in ['Healthy', 'Bipolar', 'Depression']:
    for lr in [1e-1]:
        for option in [{'persv': False}, {'persv': True}]:
            configs.append({'g': group, 'lr': lr, 'option': option})


def run_BD(i):
    tf.reset_default_graph()

    data = DataReader.read_BD()
    learning_rate = configs[i]['lr']
    group = configs[i]['g']
    option = configs[i]['option']

    if option['persv']:
        output_path = Paths.local_path + 'BD/qlp-ml-opt/' + group + '/'
        worker = QL.get_instance_with_pser(2, 0.1, 0.2, 0.3)
    else:
        output_path = Paths.local_path + 'BD/ql-ml-opt/' + group + '/'
        worker = QL.get_instance_without_pser(2, 0.1, 0.2)

    with LogFile(output_path, 'run.log'):
        DLogger.logger().debug("group: " + str(group))
        gdata = data.loc[data.diag == group]
        ids = gdata['id'].unique().tolist()
        dftr = pd.DataFrame({'id': ids, 'train': 'train'})
        tdftr = pd.DataFrame({'id': ids, 'train': 'test'})
        train, test = DataProcess.train_test_between_subject(gdata, pd.concat((dftr, tdftr)),
                                                             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        DLogger.logger().debug("total points: " + str(get_total_pionts(train)))
        train = DataProcess.merge_data(train)

        OptML.optimise(worker, output_path, train, test, global_iters=1000, learning_rate=learning_rate)


if __name__ == '__main__':

    if len(sys.argv) == 2:
        n_proc = int(sys.argv[1])
    elif len(sys.argv) == 1:
        n_proc = 1
    else:
        raise Exception('invalid argument')

    p = Pool(n_proc)
    p.map(run_BD, range(len(configs)))
    p.close() # no more tasks
    p.join()  # wrap up current tasks
