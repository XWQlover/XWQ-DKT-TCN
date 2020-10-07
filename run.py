import tensorflow as tf
from model import  DKT
from  dataUtil import  AssismentData

ass = AssismentData()
train_data,test_data = ass.datasetReturn(ass.train),ass.datasetReturn(ass.test)
val_log = 'log/val'
train_loss_log = 'log/train'
summary_writer = tf.summary.create_file_writer(val_log)

dkt = DKT(int(ass.data["skills_correctness"].max() + 1), 32)
skill_num = int((ass.data["skills_correctness"].max() + 1) / 2)
print(skill_num)
AUC = tf.keras.metrics.AUC()
VAUC = tf.keras.metrics.AUC()
SCC = tf.keras.metrics.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


def test_one_step(skills_correctness, skillid, correctness, local):
    loss = dkt(skills_correctness, local, training=False)

    label = tf.expand_dims(correctness, axis=-1)

    mask = 1. - tf.cast(tf.equal(label, -1), tf.float32)

    mask = tf.squeeze(mask)

    squenceid = tf.boolean_mask(skillid, mask=mask)
    squenceid = tf.one_hot(squenceid, depth=skill_num, axis=-1)
    label = tf.boolean_mask(label, mask=mask)
    loss = tf.boolean_mask(loss, mask=mask)

    loss = tf.expand_dims(tf.reduce_sum(tf.multiply(squenceid, loss), axis=-1), axis=-1)

    VAUC.update_state(label, loss)


def train_one_step(skills_correctness, skillid, correctness, local):
    with tf.GradientTape() as tape:
        loss = dkt(skills_correctness, local, training=True)

        label = tf.expand_dims(correctness, axis=-1)

        mask = 1. - tf.cast(tf.equal(label, -1), tf.float32)
        mask = tf.squeeze(mask)

        sequenceid = tf.boolean_mask(skillid, mask=mask)
        sequenceid = tf.one_hot(sequenceid, depth=skill_num, axis=-1)
        label = tf.boolean_mask(label, mask=mask)
        loss = tf.boolean_mask(loss, mask=mask)

        loss = tf.expand_dims(tf.reduce_sum(tf.multiply(sequenceid, loss), axis=-1), axis=-1)

        loss_real = tf.reduce_sum(tf.keras.losses.binary_crossentropy(label, loss))

        SCC.update_state(label, loss)

        AUC.update_state(label, loss)

        gradients = tape.gradient(loss_real, dkt.trainable_variables)
        # 反向传播，自动微分计算
        optimizer.apply_gradients(zip(gradients, dkt.trainable_variables))


for epoch in range(50):
    train_data = train_data.shuffle(50)
    AUC.reset_states()
    VAUC.reset_states()
    SCC.reset_states()
    for skills_correctness, skillid, correctness, w in train_data.as_numpy_iterator():
        train_one_step(skills_correctness, skillid, correctness, 1)

    for skills_correctness, skillid, correctness, w in test_data.as_numpy_iterator():
        test_one_step(skills_correctness, skillid, correctness, 1)

    with summary_writer.as_default():
        tf.summary.scalar('train_auc', AUC.result(), step=epoch)
        tf.summary.scalar('val_auc', VAUC.result(), step=epoch)

    print(SCC.result(), AUC.result(), VAUC.result())