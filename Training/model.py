
"""
January 2021

@author: Sebastian Kaltenbach
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np


class MODEL(tf.keras.Model):
  def __init__(self, latent_dim):
    super(MODEL, self).__init__()
    self.hN = latent_dim
    
    self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(self.hN*2,)),
          tf.keras.layers.Dense(25*2)
      ]
    )
    self.amortized_net= tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=(40,25)),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(units=1000, activation=tf.nn.relu),
          tf.keras.layers.Dense(self.hN*80+self.hN*80+79*self.hN+78*self.hN+39*self.hN),
      ]
    )



    
    self.Theta_r = tf.Variable(1.25*np.ones((self.hN,), dtype=np.float32), name="Theta_r",trainable=True)
    self.Theta_i = tf.Variable(0.25*np.ones((self.hN,), dtype=np.float32), name="Theta_i",trainable=True)
 
    self.X = tf.Variable(0.5*np.ones((64,40,25),dtype=np.float32), name="z_r",trainable=True)
    self.X_v = tf.Variable(-1.*np.ones((64,40,25), dtype=np.float32), name="zs_r",trainable=True)
   


  def encode(self, x):
    mean = self.inference_net(x)
    return mean

  def amortized(self, x):
    z1_m,z1_d,z1_u = tf.split(self.amortized_net(x), num_or_size_splits=[self.hN*80,self.hN*80,79*self.hN+78*self.hN+39*self.hN], axis=1)
    return z1_m,z1_d,z1_u


  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def reparameterize_full_fast(self, mean, cov_L):
    eps = tf.random.normal(shape=[64,80,1])
    return (tf.linalg.triangular_solve(cov_L,eps,lower=False))[..., 0] + mean


#Log Normal PDF
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(tf.math.minimum(-logvar,20.)) + logvar + log2pi),
      axis=raxis)

#Log Complex Normal PDF
def log_complex_normal_pdf(r, i, logvar):
  logpi = tf.math.log( np.pi)
  return  - ((r + i)  * tf.exp(tf.math.minimum(-logvar,20.)) + logvar + logpi)


# Log Multinomial PDF
def log_multinomial_pdf(sample, m):
  sample_s=sample-tf.tile(tf.reshape(tf.math.reduce_max(sample,axis=1),((64,1))),[1,25])
  sum=tf.reduce_sum(tf.exp(sample_s),axis=1)
  #Calculate up to a constant
  return tf.reduce_sum(tf.transpose(m)*(tf.math.log(tf.math.exp(tf.transpose(sample_s)-tf.math.log(sum)))),axis=0)



# Log Normal
def normal_pdf(sample, mean, std, raxis=1):
  return tf.reduce_sum(tf.exp(-.5 * ((sample - mean) ** 2. /std**2))/(2*np.pi*std**2)**0.5, axis=raxis)


@tf.function
def compute_ELBO(model, x,index):
  abc=64
  log_pcf=[]
  logpz=[]
  logpz0=[]
  logqz_x=[]
  n = 80
  
  X_am=model.reparameterize(model.X,5.*model.X_v) # Reparametrize X
  z_m,z_d,z_u=model.amortized(X_am)
  z_rt=[]
  z_it=[]
  diag = 1.e-1+tf.math.exp(z_d)
  for tt in range(model.hN):
      values = tf.concat([diag[:,(tt)*80:(tt+1)*80], z_u[:,(tt)*(79+78+39):(tt+1)*(79+78+39)]], axis=1)
      Cov=tf.scatter_nd(index, values, [64,n, n]) # Assemble Covariance Matrix
      z1 = model.reparameterize_full_fast(z_m[:,(tt)*80:(tt+1)*80], Cov) # Reparametrize z
      z_r_tmp=z1[:,::2]
      z_i_tmp=z1[:,1::2]
      z_rt.append(z_r_tmp)
      z_it.append(z_i_tmp)
      
  logqz_x.append(tf.reduce_sum(tf.math.log(diag))) #Entropy
  z_r_stacked=tf.stack(z_rt)
  z_i_stacked=tf.stack(z_it)
  z_r=tf.transpose(z_r_stacked,[1,2,0])
  z_i=tf.transpose(z_i_stacked,[1,2,0])
  for t in range(40):
       z=tf.concat([z_r[:,t,:],z_i[:,t,:]],axis=1)
       X_pred=model.encode(z)
       meanX, logvarX = tf.split(X_pred,num_or_size_splits=2,axis=1) 
       XX = model.reparameterize(meanX, logvarX)
       
       log_pcf.append(log_normal_pdf(X_am[:,t,:],meanX,logvarX))
       log_pcf.append(log_multinomial_pdf(X_am[:,t,:],x[:,t,:]))
       
  
  logpz0.append(log_complex_normal_pdf((z_r[:,0,:]**2.+z_i[:,0,:]**2.), tf.zeros((abc,1)), tf.math.log(1.0)*tf.ones((1,)))) #Prior on z_0
  for i in range(39):
      l_r=tf.exp(-1e-9-(i+1)*model.Theta_r**2.)*tf.cos(model.Theta_i*(i+1))
      l_i=tf.exp(-1e-9-(i+1)*model.Theta_r**2.)*tf.sin(model.Theta_i*(i+1))
      logpz.append(log_complex_normal_pdf((z_r[:,i+1,:]-z_r[:,0,:]*l_r+z_i[:,0,:]*l_i)**2.,(z_i[:,i+1,:]-z_i[:,0,:]*l_r-z_r[:,0,:]*l_i)**2.,tf.math.log(1.-tf.exp(-1e-9-model.Theta_r**2)**(2*(i+1))) )) #p(z_{t+i} | p(z_t)) or p(z_{t+1} | p(z_t))


  logpzstacked=tf.stack(logpz)
  logpz0stacked=tf.stack(logpz0)
  logpcfstacked=tf.stack(log_pcf)
  logqz_xs = tf.stack(logqz_x)
  beta=0.5 #Scaling on prior leads to more robust solutions if data is very small

  return -tf.reduce_sum(logpcfstacked)-beta*tf.reduce_sum(logpzstacked)-beta*tf.reduce_sum(logpz0stacked)+ tf.reduce_sum(logqz_xs)-0.5*tf.reduce_sum(5.*model.X_v)

@tf.function
def compute_apply_gradients(model, x, optimizer,index):
  with tf.GradientTape() as tape:
    loss = compute_ELBO(model, x,index)
  gradients = tape.gradient(loss, model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, model.trainable_variables))






