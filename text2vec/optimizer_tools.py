import tensorflow as tf
import math


class RampUpDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, embedding_size, warmup_steps=4000):
        super(RampUpDecaySchedule, self).__init__()
        self.decay_rate = embedding_size ** -0.5
        self.dims = tf.cast(embedding_size, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step, **kwargs):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dims) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "initial_learning_rate": 0,
            "warmup_steps": self.warmup_steps,
            "decay_rate": self.decay_rate,
            "name": "ramp-up-decay-lr"
        }

    def callback(self, step):
        arg1 = math.sqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return (1 / math.sqrt(self.dims)) * min(1 / arg1, arg2)
