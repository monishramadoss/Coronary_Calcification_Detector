import datetime
import glob
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import subprocess
import numpy as np
import tensorflow as tf
from jarvis.train import custom
from tensorflow import optimizers, losses
from tensorflow.keras import Input, layers, Model, callbacks
from sklearn.model_selection import train_test_split
from tqdm import trange
try:
    from jarvis.utils.general import gpus
    gpus.autoselect(1)
except:
    pass

os.makedirs('./plaque_seg', exist_ok=True)
os.makedirs('./plaque_seg/log_dir', exist_ok=True)
os.makedirs('./plaque_seg/ckp', exist_ok=True)


def dense_unet(inputs, filters=32):
    '''Model Creation'''
    # Define kwargs dictionary
    kwargs = {
        'kernel_size': (1, 3, 3),
        'padding': 'same',
        'bias_initializer': 'zeros'
    }
    # Define lambda functions#
    conv = lambda x, filters, strides: layers.Conv3D(filters=int(filters), strides=(1, strides, strides), **kwargs)(x)
    norm = lambda x: layers.BatchNormalization()(x)
    relu = lambda x: layers.LeakyReLU()(x)
    # Define stride-1, stride-2 blocks#
    conv1 = lambda filters, x: relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x: relu(norm(conv(x, filters, strides=2)))
    # Define single transpose#
    tran = lambda x, filters, strides: layers.Conv3DTranspose(filters=int(filters), strides=(1, strides, strides),
                                                              **kwargs)(x)
    # Define transpose block#
    tran2 = lambda filters, x: relu(norm(tran(x, filters, strides=2)))
    concat = lambda a, b: layers.Concatenate()([a, b])

    # Define Dense Block#
    def dense_block(filters, input, DB_depth):
        ext = 2 + DB_depth
        outside_layer = input
        for _ in range(0, int(ext)):
            inside_layer = conv1(filters, outside_layer)
            outside_layer = concat(outside_layer, inside_layer)
        return outside_layer

    def td_block(conv1_filters, conv2_filters, input, DB_depth):
        TD = conv1(conv1_filters, conv2(conv2_filters, input))
        DB = dense_block(conv1_filters, TD, DB_depth)
        return DB

    def tu_block(conv1_filters, tran2_filters, input, td_input, DB_depth):
        TU = conv1(conv1_filters, tran2(tran2_filters, input))
        C = concat(TU, td_input)
        DB = dense_block(conv1_filters, C, DB_depth)
        return DB

    TD1 = td_block(filters * 1, filters * 1, inputs, 0)
    TD2 = td_block(filters * 1.5, filters * 1, TD1, 1)
    TD3 = td_block(filters * 2, filters * 1.5, TD2, 2)
    TD4 = td_block(filters * 2.5, filters * 2, TD3, 3)
    TD5 = td_block(filters * 3, filters * 2.5, TD4, 4)

    TU1 = tu_block(filters * 2.5, filters * 3, TD5, TD4, 4)
    TU2 = tu_block(filters * 2, filters * 2.5, TU1, TD3, 3)
    TU3 = tu_block(filters * 1.5, filters * 2, TU2, TD2, 2)
    TU4 = tu_block(filters * 1, filters * 1.5, TU3, TD1, 1)
    TU5 = tran2(filters * 1, TU4)
    logits = {}
    logits = layers.Conv3D(filters=2, **kwargs)(TU5)
    model = Model(inputs=inputs, outputs=logits)
    return model


class sce_dsc(layers.Layer):
    def __init__(self, scale_sce=1.0, scale_dsc=1.0, epsilon=0.01, name=None):
        super(sce_dsc, self).__init__(name=name)
        self.sce = losses.SparseCategoricalCrossentropy(from_logits=True)
        self.epsilon = epsilon
        self.scale_a = scale_sce
        self.scale_b = scale_dsc
        self.cls = 1

    def dsc(self, y_true, y_pred, sample_weight=None):
        true = tf.cast(y_true[..., 0] == self.cls, tf.float32)
        pred = tf.nn.softmax(y_pred, axis=-1)[..., self.cls]
        if sample_weight is not None:
            true = true * (sample_weight[...])
            pred = pred * (sample_weight[...])
        A = tf.math.reduce_sum(true * pred) * 2
        B = tf.math.reduce_sum(true) + tf.math.reduce_sum(pred) + self.epsilon
        return (1.0 - A / B) * self.scale_b

    def call(self, y_true, y_pred, weights=None):
        sce_loss = self.sce(y_true=y_true, y_pred=y_pred, sample_weight=weights) * self.scale_a
        dsc_loss = self.dsc(y_true=y_true, y_pred=y_pred, sample_weight=weights)
        loss = sce_loss + dsc_loss      
        self.add_loss(loss)
        return loss


data = np.expand_dims(np.load('./data/orca_data.npy'), (1, -1)).astype(np.float32)
data = np.clip(data, 30, 150)
label = np.expand_dims(np.load('./data/orca_label.npy'), (1, -1))

if not os.path.isfile('./data/heart_msk_data.npy'):
    file_out = open('./msk_gen_stdout', 'w')
    subprocess.Popen(['python', 'heart_mask_gen.py'], stdout=file_out, shell=False)
    file_out.flush()
    file_out.close()

msk = np.expand_dims(np.load('./data/heart_msk_data.npy'), (1, -1)).astype(np.float32)
assert msk.shape == data.shape

data *= msk

train_x, valid_x, train_y, valid_y, train_msk, valid_msk = train_test_split(data, label, msk, test_size=0.33, random_state=42)

gen_train = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(4).shuffle(100)
gen_valid = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(1)


model_checkpoint_callback = callbacks.ModelCheckpoint(filepath='./plaque_seg/ckp/', monitor='val_dsc_1',
                                                          mode='max', save_best_only=True)
tensorboard_callback = callbacks.TensorBoard('./plaque_seg/log_dir', profile_batch=0)

reduce_lr_callback = callbacks.ReduceLROnPlateau(monitor='dsc_1', factor=0.8, patience=2, mode="max", verbose=1)
early_stop_callback = callbacks.EarlyStopping(monitor='val_dsc_1', patience=5, verbose=0, mode='max',
                                                  restore_best_weights=False)

model = dense_unet(Input(shape=(1, 512, 512, 1)), 32)


model.compile(optimizer=optimizers.Adam(learning_rate=8e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  loss=sce_dsc(1.0, 0.3), metrics=[custom.dsc(cls=1)])                  

model.fit(x=gen_train, epochs=200, validation_data=gen_valid, validation_freq=1,
              callbacks=[tensorboard_callback, model_checkpoint_callback, reduce_lr_callback, early_stop_callback])

model.save('./plaque_seg/model.h5', include_optimizer=False, overwrite=True)