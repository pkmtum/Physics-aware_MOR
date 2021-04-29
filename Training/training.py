
"""
September 2020

@author: Sebastian Kaltenbach
"""

import model as m
import tensorflow as tf
import time
import scipy.io
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt




mat = np.load('Data_AD.npy')
md=mat

train_images= md[:,:40,:].astype('float32')

BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE)


epochs = 200000
latent_dim = 5
model = m.MODEL(latent_dim)
optimizer = tf.keras.optimizers.Adam(8e-4)



# Used to assemble Covariance matrix later. Corresponds to a d-dimensional bidiagonal complex covariance.
# In case the interaction between real and imaginary components should be neglected the 2d-dimeniosnal real-valued covariance matrix is also bidiagonal.
n=80
r = tf.range(n)
ii = tf.concat([r, r[:-1], r[:-2],r[:-3:2]], axis=0)
jj = tf.concat([r, r[1:], r[2:],r[3::2]], axis=0)
index=np.zeros((64,int(n+(n-1)+(n-2)+(n-2)/2),3),dtype=np.int32)
for samples in range(64):
  tester=np.ones((int(n+(n-1)+(n-2)+(n-2)/2),))*(samples)
  idxx = tf.stack([tester,ii, jj], axis=1)
  index[samples,:,:]=idxx

index_tf=tf.constant(index)



for epoch in range(1, epochs + 1):
  for train_x in train_dataset:
      m.compute_apply_gradients(model, train_x, optimizer,index_tf)

  if epoch % 1000 == 0:
    elbo = -m.compute_ELBO(model,train_x,index_tf)
    print('Epoch: {}, Current ELBO: {} '.format(epoch,elbo))


    
model.save_weights('weights_AD', save_format='tf')


