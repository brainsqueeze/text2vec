import tensorflow as tf
import math


class RampUpDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Defines a linear ramp up and then geometric decline of the learning rate as defined in
    https://arxiv.org/abs/1706.03762.

    Parameters
    ----------
    embedding_size : int
        Dimensionality of the word-embedding.
    warmup_steps : int, optional
        How many steps should support the linear ramp up, by default 4000
    """

    def __init__(self, embedding_size, warmup_steps=4000):
        super().__init__()
        self.decay_rate = embedding_size ** -0.5
        self.dims = tf.cast(embedding_size, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dims) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        """Get the class config settings.

        Returns
        -------
        dict
        """

        return {
            "initial_learning_rate": 0,
            "warmup_steps": self.warmup_steps,
            "decay_rate": self.decay_rate,
            "name": "ramp-up-decay-lr"
        }

    def callback(self, step):
        """Get the learning rate as a callback for the sake of Tensorboard logging.

        Parameters
        ----------
        step : int
            Training step.

        Returns
        -------
        float
            Learning rate at a given step
        """

        arg1 = math.sqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return (1 / math.sqrt(self.dims)) * min(1 / arg1, arg2)
