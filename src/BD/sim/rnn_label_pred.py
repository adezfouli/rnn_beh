# Making predictions about diagnostic labels of the subjects. Note that this file needs
# the output of 'fit/rnn_pred.py'.

import tensorflow as tf
from os.path import isdir
from actionflow.data.data_process import DataProcess
from actionflow.rnn.lstm_beh import LSTMBeh
from actionflow.util import DLogger
from BD.data.data_reader import DataReader
import pandas as pd
from BD.util.paths import Paths


def finding_CV(base_input_folder):
    from os import listdir
    from os.path import join

    group_address = {'Healthy': base_input_folder + 'Healthy/fold0/',
                     'Bipolar': base_input_folder + 'Bipolar/fold0/',
                     'Depression': base_input_folder + 'Depression/fold0/'}

    subj_address = {}

    for group in ['Healthy', 'Depression', 'Bipolar']:

        input_folder = base_input_folder + group + '/'
        sims_dirs = [f for f in listdir(input_folder) if isdir(join(input_folder, f))]
        for d in sims_dirs:
            DLogger.logger().debug(d)
            full_path = input_folder + d + '/'

            tr_tst = pd.read_csv(full_path + 'train_test.csv')

            if not ('id' in tr_tst):
                tr_tst['id'] = tr_tst['ID']

            tst_ids = tr_tst.loc[tr_tst.train == 'test']['id']

            if len(tst_ids) > 1:
                raise Exception('tests should contain one data-point.')
            if len(tst_ids) > 0:
                _id = tst_ids.iloc[0]
                subj_address[_id] = {}
                for group2 in ['Healthy', 'Depression', 'Bipolar']:
                    if group == group2:
                        subj_address[_id][group] = full_path
                    else:
                        subj_address[_id][group2] = group_address[group2]

    return subj_address


def RNN_classify_subjects():
    data = DataReader.read_BD()
    ids = data['id'].unique().tolist()
    dftr = pd.DataFrame({'id': ids, 'train': 'train'})
    train, test = DataProcess.train_test_between_subject(data, dftr,
                                                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    n_cells = {'Healthy': 10, 'Depression': 10, 'Bipolar': 20}
    model_iter = 'model-final'

    df = pd.DataFrame(columns=('model', 'id', 'loss'))
    config = tf.ConfigProto(device_count={'GPU': 0})

    subj_paths = finding_CV(Paths.rest_path + 'archive/beh/rnn-pred-diag/')

    for k, tr in train.iteritems():
        for g, p in subj_paths[k].iteritems():
            tf.reset_default_graph()
            worker = LSTMBeh(2, 0, n_cells[g])
            saver = tf.train.Saver(max_to_keep=5)

            DLogger.logger().debug('subject ' + k + ' group ' + g + ' path ' + p)
            model_path = p + model_iter + '/'
            ckpt = tf.train.get_checkpoint_state(model_path)
            tf.train.import_meta_graph(model_path + 'model.cptk.meta')
            with tf.Session(config=config) as sess:
                saver.restore(sess, ckpt.model_checkpoint_path)

                total_loss = 0
                for v in tr:
                    policies, c_track, h_track, loss = worker.simulate(sess,
                                                                       v['reward'],
                                                                       v['action'],
                                                                       v['state'])
                    total_loss += loss

                df.loc[len(df)] = [g, k, total_loss]

    df.to_csv(Paths.local_path + 'BD/rnn_diag.csv')


if __name__ == '__main__':
    RNN_classify_subjects()
