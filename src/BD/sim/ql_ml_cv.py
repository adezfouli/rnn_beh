# Evaluates QL and QLP in terms of cross-validation. Note that this file only evaluates the models and it needs
# the output of file 'fit/ql_ml_cv.py' for trained models.

from actionflow.qrl.opt_ml import OptML
from actionflow.qrl.ql import QL
from actionflow.qrl.simulate import Simulator
from BD.data.data_reader import DataReader
import tensorflow as tf
from BD.util.paths import Paths


def evaluate_qlp_cv():
    data = DataReader.read_BD()
    base_input_folder = Paths.rest_path + 'archive/beh/qlp-ml-cv/'
    base_output_folder = Paths.local_path + 'BD/evals/qlp-ml-cv-evals/'
    model_iters = ['model-final']
    folds = {'Healthy': ['fold' + str(x) for x in range(0, 34)],
             'Bipolar': ['fold' + str(x) for x in range(0, 33)],
             'Depression': ['fold' + str(x) for x in range(0, 34)]
             }

    tf.reset_default_graph()
    worker = QL.get_instance_with_pser(2, 0.1, 0.1, 0.1)
    worker.set_params(OptML.get_variables(worker.get_params()))

    def test_and_save(sess, test, output_folder):
        return OptML.test_and_save("", output_folder, None, sess, test, worker)

    Simulator.evaluate_CV(base_input_folder, base_output_folder, test_and_save, data, folds,
                                    model_iters,
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], random_tie=True)


def evaluate_ql_cv():
    data = DataReader.read_BD()
    base_input_folder = Paths.rest_path + 'archive/beh/ql-ml-cv/'
    base_output_folder = Paths.local_path + 'BD/evals/ql-ml-cv-evals/'
    model_iters = ['model-final']
    folds = {'Healthy': ['fold' + str(x) for x in range(0, 34)],
             'Bipolar': ['fold' + str(x) for x in range(0, 33)],
             'Depression': ['fold' + str(x) for x in range(0, 34)]
             }

    tf.reset_default_graph()
    worker = QL.get_instance_without_pser(2, 0.1, 0.1)
    worker.set_params(OptML.get_variables(worker.get_params()))

    def test_and_save(sess, test, output_folder):
        return OptML.test_and_save("", output_folder, None, sess, test, worker)

    Simulator.evaluate_CV(base_input_folder, base_output_folder, test_and_save, data, folds,
                                    model_iters,
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], random_tie=True)

if __name__ == '__main__':
    evaluate_qlp_cv()
    evaluate_ql_cv()
