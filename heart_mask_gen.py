import os
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers, Model, callbacks

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

try:
    from jarvis.utils.general import gpus
    gpus.autoselect(1)
except:
    pass

if not os.path.isfile('./struct_seg/model.h5'):
    file_out = open('./mask_seg', 'w')
    subprocess.Popen(['python', 'struct_seg.py'], stdout=file_out, shell=False)
    file_out.flush()
    file_out.close()

model = tf.keras.models.load_model('./struct_seg/model.h5', compile=False)
data = np.expand_dims(np.load('./data/plaque_data.npy'), (1, -1)).astype(np.float32)
data = np.clip(data, -100, 100) / 100
label = np.expand_dims(np.load('./data/plaque_label.npy'), (1, -1))
logit = model.predict(data, batch_size=4, workers=12)
logit = np.argmax(logit, axis=-1)
logit = np.squeeze(logit)
print(logit.shape)
np.save('./data/heart_msk_data.npy', logit)
