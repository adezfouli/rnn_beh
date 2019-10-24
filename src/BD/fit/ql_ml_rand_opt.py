# This file optimises QL, but fits a separate model to each subject (instead of fitting a single model to
# the whole group).

import sys
from multiprocessing.pool import Pool
from actionflow.data.data_process import DataProcess
from actionflow.qrl.opt_ml import OptML
from actionflow.qrl.ql import QL
from actionflow.util.helper import get_total_pionts
from actionflow.util.logger import LogFile, DLogger
from BD.data.data_reader import DataReader
from BD.util.paths import Paths
import pandas as pd
import tensorflow as tf

cv_lists_group = {}
data = DataReader.read_BD()
for group in data['diag'].unique().tolist():
    gdata = data.loc[data.diag == group]
    ids = gdata['id'].unique().tolist()

    cv_lists = []
    for i in range(len(ids)):
        dftr = pd.DataFrame({'id': [ids[i]], 'train': 'train'})
        dfte = pd.DataFrame({'id': [ids[i]], 'train': 'test'})
        cv_lists.append(pd.concat((dftr, dfte)))

    cv_lists_group[group] = cv_lists

configs = []
for i in range(len(cv_lists_group['Healthy'])):
    configs.append({'g': 'Healthy', 'lr': 1e-1, 'cv_index': i, 'iters': 300})


for i in range(len(cv_lists_group['Depression'])):
    configs.append({'g': 'Depression', 'lr': 1e-1, 'cv_index': i, 'iters': 300})


for i in range(len(cv_lists_group['Bipolar'])):
    configs.append({'g': 'Bipolar', 'lr': 1e-1, 'cv_index': i, 'iters': 300})


def run_BD_QL(i):
    tf.reset_default_graph()
    learning_rate = configs[i]['lr']
    group = configs[i]['g']
    cv_index = configs[i]['cv_index']
    iters = configs[i]['iters']

    output_path = Paths.local_path + 'BD/ql-ml-rand-opt/' + group + '/' + 'fold' + str(cv_index) + '/'
    with LogFile(output_path, 'run.log'):
        indx_data = cv_lists_group[group][cv_index]
        gdata = data.loc[data.diag == group]
        train, test = DataProcess.train_test_between_subject(gdata, indx_data,
                                                             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

        indx_data.to_csv(output_path + 'train_test.csv')
        DLogger.logger().debug("total points: " + str(get_total_pionts(train)))

        worker = QL.get_instance_without_pser(2, 0.1, 0.2)
        train = DataProcess.merge_data(train)

        OptML.optimise(worker, output_path, train, None, learning_rate=learning_rate, global_iters=iters)


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

    run_cv(run_BD_QL, n_proc)
