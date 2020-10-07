import tensorflow as tf
from TCN import TCN
class DKT(tf.keras.models.Model):
    def __init__(self, total_skills_correctness, embedding_size):
        super(DKT, self).__init__(name="DKTModel")

        self.mask = tf.keras.layers.Masking(mask_value=-1)

        # 两个嵌入层
        self.skill_embedding = tf.keras.layers.Embedding(total_skills_correctness, embedding_size)
        # RNN
        self.rnn = TCN(nb_filters=[128, 128, 64, 32],
                       dropout_rate=0.3,
                       kernel_size=(6, 3, 3, 3),
                       dilations=(1, 2, 4, 8),
                       nb_stacks=1, return_sequences=True)

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(total_skills_correctness / 2, activation='sigmoid')])

        self.distribute = tf.keras.layers.TimeDistributed(self.dense)


    def call(self, skillid, local, training=None):
        skillid = tf.expand_dims(skillid, axis=-1)

        skillid = self.mask(skillid)

        x = self.skill_embedding(skillid)

        x = tf.squeeze(x, axis=-2)

        x = self.rnn(x, training=training)

        y = self.distribute(x)

        return y
