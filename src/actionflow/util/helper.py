import tensorflow as tf
import string
import random
import numpy as np
import pandas as pd

from ..rnn.consts import Const
import os


class Helper:
    def __init__(self):
        pass

    @staticmethod
    def dicstr(adict):
        return ''.join('{}_{}-'.format(key, val) for key, val in adict.items())

    @staticmethod
    def get_num_trainables():
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print(shape)
            # print(len(shape))
            variable_parametes = 1
            for dim in shape:
                # print(dim)
                variable_parametes *= dim.value
            # print(variable_parametes)
            total_parameters += variable_parametes
        return total_parameters

    @staticmethod
    def id_generator(size=4, chars=string.ascii_uppercase + string.digits):
        """generates a random sequence of character of length ``size``"""

        return ''.join(random.choice(chars) for _ in range(size))

    @staticmethod
    def get_git():
        """
        If the current directory is a git repository, this function extracts the hash code, and current branch

        Returns
        -------
        hash : string
         hash code of current commit

        branch : string
         current branch
        """
        try:
            from subprocess import Popen, PIPE

            gitproc = Popen(['git', 'show-ref'], stdout=PIPE)
            (stdout, stderr) = gitproc.communicate()

            gitproc = Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=PIPE)
            (branch, stderr) = gitproc.communicate()
            branch = branch.split('\n')[0]
            for row in stdout.split('\n'):
                if row.find(branch) != -1:
                    hash = row.split()[0]
                    break
        except:
            hash = None
            branch = None
        return hash, branch


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out, dtype=Const.FLOAT)

    return _initializer


def fix_init_all_vars(sess):
    agn = []
    for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        agn.append(tf.assign(i, tf.random_uniform(tf.shape(i), seed=1, dtype=i.dtype.base_dtype)))
    sess.run(agn)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def cv_list(ids, n_folds):
    ls = np.array_split(np.array(ids), n_folds)
    cv_lists = []
    for k in range(len(ls)):
        dl = ls[:k] + ls[(k + 1):]
        tid = [item for sublist in dl for item in sublist]
        dftr = pd.DataFrame({'id': tid, 'train': 'train'})
        dfte = pd.DataFrame({'id': ls[k], 'train': 'test'})
        cv_lists.append(pd.concat((dftr, dfte)))

    return cv_lists


def get_git():
    """
    If the current directory is a git repository, this function extracts the hash code, and current branch

    Returns
    -------
    hash : string
     hash code of current commit

    branch : string
     current branch
    """
    try:
        from subprocess import Popen, PIPE

        gitproc = Popen(['git', 'show-ref'], stdout=PIPE)
        (stdout, stderr) = gitproc.communicate()

        gitproc = Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stdout=PIPE)
        (branch, stderr) = gitproc.communicate()
        branch = branch.split('\n')[0]
        for row in stdout.split('\n'):
            if row.find(branch) != -1:
                hash = row.split()[0]
                break
    except:
        hash = None
        branch = None
    return hash, branch


def id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    """generates a random sequence of character of length ``size``"""

    return ''.join(random.choice(chars) for _ in range(size))


def one_hot(a, n=None):
    if n is None:
        n = a.max() + 1
    b = np.zeros((a.shape[0], n))
    b[np.arange(a.shape[0]), a] = 1
    return b


def one_hot_batch(a, n=None):
    if n is None:
        n = a.max() + 1
    b = (np.arange(n) == a[..., None]).astype(int)
    return b


def get_files_starting_with(path, starts_with):
    import os
    names = []
    for file in os.listdir(path):
        if file.startswith(starts_with):
            names.append(os.path.join(path, file))
    names.sort()
    return names


def get_total_pionts(train):
    s = 0
    for v in train.values():
        for t in v:
            s += t['reward'].shape[0]
    return s


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def format_to_training_data(rewards, choices, states):
    times = np.zeros((rewards.shape[0]))
    return {'id1': [{
        'reward': rewards[np.newaxis,],
        'action': choices[np.newaxis,],
        'timesteps': times[np.newaxis,],
        'state': states[np.newaxis, :, np.newaxis],
        'block': 1,
        'id': 'id1'
    }]}


def get_KL_normal(mu, sigma2, prior_mu, prior_sigma2):
    _half = tf.constant(0.5, dtype=Const.FLOAT)
    kl = tf.add(tf.div(tf.add(tf.square(tf.subtract(mu, prior_mu)), sigma2),
                       tf.multiply(tf.constant(2., dtype=Const.FLOAT), prior_sigma2))
                , -_half + _half * tf.log(prior_sigma2) - _half * tf.log(sigma2))

    return tf.reduce_sum(kl)


def load_model(sess, model_path, load_meta_graph=False):
    saver = tf.train.Saver(max_to_keep=5)
    ckpt = tf.train.get_checkpoint_state(model_path)
    if load_meta_graph:
        tf.train.import_meta_graph(model_path + 'model.cptk.meta')
    saver.restore(sess, ckpt.model_checkpoint_path)
    return saver


def flatten(arr):
    return np.concatenate([p.flatten() for p in arr])


def tf_round(x, decimals=0):
    multiplier = tf.constant(10 ** decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier


def state_to_onehot(train, n_states):
    for k in train.keys():
        for v in train[k]:
            v['state'] = one_hot_batch(v['state'], n_states)


def obj_to_str(obj):
    str_out = ''
    if type(obj) is dict:
        for k in sorted(obj.keys()):
            str_out += str(k)
            str_out += obj_to_str(obj[k])
        return str_out

    if type(obj) is list:
        for o in obj:
            str_out += obj_to_str(o)
        return str_out

    if type(obj) is tuple:
        for o in obj:
            str_out += obj_to_str(o)
        return str_out

    return str(obj)
