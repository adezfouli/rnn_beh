# Evaluates RNN in terms of cross-validation. Note that this file only evaluates the model and it needs
# the output of file 'fit/rnn_cv.py' for trained models.

from actionflow.rnn.lstm_beh import LSTMBeh
from actionflow.rnn.simulate import Simulator
from BD.data.data_reader import DataReader
from BD.util.paths import Paths


def evaluate_BD_CV():
    data = DataReader.read_BD()
    base_input_folder = Paths.rest_path + 'archive/beh/rnn-cv/'
    base_output_folder = Paths.local_path + 'BD/evals/rnn-cv-evals/'
    model_iters = ['model-0', 'model-100',
                   'model-200', 'model-300',
                   'model-400', 'model-500',
                   'model-600', 'model-700',
                   'model-800', 'model-900',
                   'model-1000', 'model-1100',
                   'model-1200', 'model-1300',
                   'model-1400', 'model-1500',
                   'model-1600', 'model-1700',
                   'model-1800', 'model-1900',
                   'model-2000', 'model-2100',
                   'model-2200', 'model-2300',
                   'model-2400', 'model-2500',
                   'model-2600', 'model-2700',
                   'model-2800', 'model-2900',
                   'model-final'
                   ]

    cells = [5, 10, 20]
    folds = {'Healthy': ['fold' + str(x) for x in range(0, 34)],
             'Depression': ['fold' + str(x) for x in range(0, 34)],
             'Bipolar': ['fold' + str(x) for x in range(0, 33)]
             }

    def worker_gen(n_actions, n_states, n_cells):
        return LSTMBeh(2, 0, n_cells)

    Simulator.evaluate_CV(worker_gen, base_input_folder, base_output_folder,
                          cells, 2, 0,
                          data, folds, model_iters,
                          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


if __name__ == '__main__':
    evaluate_BD_CV()
