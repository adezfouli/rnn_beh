import tensorflow as tf


class Opt:

    @staticmethod
    def _get_apply_grads(obj, trainables, learning_rate, _optimizer):

        optimizer = _optimizer(learning_rate=learning_rate)
        max_global_norm = 50.0
        grads = tf.gradients(obj, trainables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
        grad_var_pairs = zip(grads, trainables)
        apply_grads = optimizer.apply_gradients(grad_var_pairs)
        return apply_grads

    @staticmethod
    def get_apply_grads(obj, trainables, learning_rate):
        return Opt._get_apply_grads(obj, trainables, learning_rate, tf.train.AdamOptimizer)

    @staticmethod
    def get_apply_grads_RMSProp(obj, trainables, learning_rate):
        return Opt._get_apply_grads(obj, trainables, learning_rate, tf.train.RMSPropOptimizer)
