# This runs cross-validation for GQL (d=10). Note that the output of this file is only the saved models (and not the
# actual performance of the model on the test data). For evaluating how the saved models perform
# on the test dataset, file 'sim/gql_ml_cv.py' should be
# used on the output of this file.


import sys
from multiprocessing.pool import Pool
from actionflow.data.data_process import DataProcess
from actionflow.qrl.gql import GQL
from actionflow.qrl.opt_ml import OptML
from actionflow.util.helper import cv_list
from actionflow.util.logger import LogFile
from BD.data.data_reader import DataReader
from BD.util.paths import Paths
import tensorflow as tf


cv_lists_group = {}
data = DataReader.read_BD()
for group in data['diag'].unique().tolist():
    gdata = data.loc[data.diag == group]
    ids = gdata['id'].unique().tolist()
    cv_lists_group[group] = cv_list(ids, len(ids))

configs = []

for group in ['Healthy', 'Depression', 'Bipolar']:
    gdata = data.loc[data.diag == group]
    ids = gdata['id'].unique().tolist()
    for lr in [1e-1]:
        for i in range(len(ids)):
            configs.append({'g': group, 'lr': lr, 'cv_index': i})


def run_BD_Q(i):
    tf.reset_default_graph()

    learning_rate = configs[i]['lr']
    group = configs[i]['g']
    cv_index = configs[i]['cv_index']

    indx_data = cv_lists_group[group][cv_index]
    gdata = data.loc[data.diag == group]
    train, test = DataProcess.train_test_between_subject(gdata, indx_data,
                                                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    output_path = Paths.local_path + 'BD/gql10d-ml-cv/' + group + '/' + 'fold' + str(cv_index) + '/'
    with LogFile(output_path, 'run.log'):
        indx_data.to_csv(output_path + 'train_test.csv')

        worker = GQL.get_instance(2, 10, {})
        train = DataProcess.merge_data(train)

        OptML.optimise(worker, output_path, train, test, learning_rate=learning_rate, global_iters=1000)


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

    run_cv(run_BD_Q, n_proc)
