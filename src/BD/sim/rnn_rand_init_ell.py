# Calculates ELL for RNNs optimised from random initialisations.

from actionflow.data.data_process import DataProcess
from actionflow.rnn.lstm_beh import LSTMBeh
from actionflow.util import DLogger
from actionflow.util.helper import ensure_dir
from BD.data.data_reader import DataReader
from BD.util.paths import Paths
import pandas as pd
import tensorflow as tf

configs = []

for i in range(15):
    configs.append({
        'g': 'Healthy',
        'cells': 10,
        's': i
        },
    )

    configs.append({
        'g': 'Bipolar',
        'cells': 20,
        's': i
    })

    configs.append({
        'g': 'Depression',
        'iters': 1200,
        'cells': 10,
        's': i
    })


def run_BD(i):
    data = DataReader.read_BD()
    ncells = configs[i]['cells']
    group = configs[i]['g']
    input_path = Paths.rest_path + 'archive/beh/rnn-opt-rand-init/' + 'run_' + \
                 str(configs[i]['s']) + '/' + str(ncells) + 'cells/' + group + '/model-final/'
    output_path = Paths.local_path + 'BD/rnn-opt-rand-init-evals/' + 'run_' + \
                  str(configs[i]['s']) + '/' + str(ncells) + 'cells/' + group + '/'

    gdata = data.loc[data.diag == group]
    ids = gdata['id'].unique().tolist()
    dftr = pd.DataFrame({'id': ids, 'train': 'train'})
    tdftr = pd.DataFrame({'id': ids, 'train': 'test'})
    train, test = DataProcess.train_test_between_subject(gdata, pd.concat((dftr, tdftr)),
                                                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    test = DataProcess.merge_data(test)
    tf.reset_default_graph()
    worker = LSTMBeh(2, 0, ncells)
    saver = tf.train.Saver(max_to_keep=None)

    with tf.Session() as sess:
        DLogger.logger().debug("loading model from: " + str(input_path))
        ckpt = tf.train.get_checkpoint_state(input_path)
        tf.train.import_meta_graph(input_path + 'model.cptk.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)

        for k, tr in test.iteritems():
            for v in tr:
                _, _, _, ell = worker.simulate(sess,
                                        v['reward'],
                                        v['action'],
                                        v['state'])

    DLogger.logger().debug("input path: " + input_path)
    DLogger.logger().debug("output path: " + output_path)
    DLogger.logger().debug("total nlp: {} ".format(str(ell)))

    return pd.DataFrame({
        'total nlp': [ell],
        'group': group,
        'cell': ncells,
        'fold': None,
        'model_iter': 'model-final',
        's': configs[i]['s']
    })


if __name__ == '__main__':
    df = pd.DataFrame()
    for i in range(len(configs)):
        df = df.append(run_BD(i))

    ensure_dir(Paths.local_path + 'BD/rnn-opt-rand-init-evals/')
    df.to_csv(Paths.local_path + 'BD/rnn-opt-rand-init-evals/' + 'accu.csv')
