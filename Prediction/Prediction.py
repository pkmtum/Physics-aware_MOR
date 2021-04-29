
"""
January 2021

@author: Sebastian Kaltenbach
"""

import numpy as np
import tensorflow as tf

# PReconstructed densities in the training data
def prediction(sa,ti,model):
    
    density=np.zeros((100,25*1))

    for mc in range(100):
        X_am=model.reparameterize(model.X[sa,ti,:],5*model.X_v[sa,ti,:])
        d2=tf.nn.softmax(X_am)
        density[mc,:]=np.random.multinomial(250000,d2[:])

    density_m=np.mean(density/250000,axis=0)
    density_s=np.std(density/250000,axis=0)
    
    return density_m,density_s

#Fast reparametrization for triangular covariance matrix (complex covariance amtrix is bidiagonal)
def reparameterize_full_fast( mean, cov_L):
    eps = tf.random.normal(shape=[1,80,1])
    return (tf.linalg.triangular_solve(cov_L,eps,lower=False))[..., 0] + mean

#Prediction for time-steps not contained in the training data
def prediction_extrapolative(sa, timer, model,lamb_r,lamb_i):
    ti=39
    density=np.zeros((100,25*1))
    z_new_r=np.zeros((5,1))
    z_new_i=np.zeros((5,1))
    
    #Index to build Covariance
    n=80
    r = tf.range(n)
    ii = tf.concat([r, r[:-1], r[:-2],r[:-3:2]], axis=0)
    jj = tf.concat([r, r[1:], r[2:],r[3::2]], axis=0)
    index=np.zeros((1,int(n+(n-1)+(n-2)+(n-2)/2),3),dtype=np.int32)
    for looper in range(1):
        tester=np.ones((int(n+(n-1)+(n-2)+(n-2)/2),))*(looper)
        idxx = tf.stack([tester,ii, jj], axis=1)
        index[looper,:,:]=idxx

    index_tf3=tf.constant(index)
    
    
    
    
    
    

    for mc in range(100):
        X_am=model.reparameterize(model.X[sa,:,:],5.*model.X_v[sa,:,:])
        z_m,z_d,z_u=model.amortized(tf.reshape(X_am,[1,40,25]))
        z_rt=[]
        z_it=[]
        diag = 1.e-1+tf.math.exp(z_d)
        for tt in range(model.hN):
          values = tf.concat([diag[:,(tt)*80:(tt+1)*80], z_u[:,(tt)*(79+78+39):(tt+1)*(79+78+39)]], axis=1)
          Cov=tf.scatter_nd(index_tf3, values, [1,n, n])
          z1 = reparameterize_full_fast(z_m[:,(tt)*80:(tt+1)*80], Cov)
          z_r_tmp=z1[:,::2]
          z_i_tmp=z1[:,1::2]
          z_rt.append(z_r_tmp)
          z_it.append(z_i_tmp)

        z_r_stacked=tf.stack(z_rt)
        z_i_stacked=tf.stack(z_it)
        z_r_stacked2=tf.transpose(z_r_stacked,[1,2,0])
        z_i_stacked2=tf.transpose(z_i_stacked,[1,2,0])
        z_r=z_r_stacked2[0,ti,:]
        z_i=z_i_stacked2[0,ti,:]
  
  
   
        for i in range(5):
            z_new_r[i,0]=np.random.normal(np.exp(-timer*lamb_r[i]**2)*np.cos(lamb_i[i]*timer)*z_r[i]-np.exp(-timer*lamb_r[i]**2)*np.sin(timer*lamb_i[i])*z_i[i],(1./np.sqrt(2.))*(1-np.exp(-lamb_r[i]**2)**(2*timer))**0.5)
            z_new_i[i,0]=np.random.normal(np.exp(-timer*lamb_r[i]**2)*np.cos(lamb_i[i]*timer)*z_i[i]+np.exp(-timer*lamb_r[i]**2)*np.sin(timer*lamb_i[i])*z_r[i],(1./np.sqrt(2.))*(1-np.exp(-lamb_r[i]**2)**(2*timer))**0.5)
    
        z_new=tf.concat([z_new_r,z_new_i],axis=0)
        X=model.encode(tf.reshape(z_new,[1,5*2]))
    
    

        meanX, varX = tf.split(X,num_or_size_splits=2, axis=1) 
        XX = model.reparameterize(meanX, varX)
        d2=tf.nn.softmax(XX)
        density[mc,:]=np.random.multinomial(250000,d2[0,:])

    density_m=np.mean(density/250000,axis=0)
    density_s=np.std(density/250000,axis=0)
    
    return density_m,density_s


def prediction_z(sa,model,lamb_r,lamb_i):

    ti=39
    timert=1000
    z_new_r=np.zeros((5,timert))
    z_new_i=np.zeros((5,timert))
    z_rmc=np.zeros((100,40,5))
    z_imc=np.zeros((100,40,5))
    
    n=80
    r = tf.range(n)
    ii = tf.concat([r, r[:-1], r[:-2],r[:-3:2]], axis=0)
    jj = tf.concat([r, r[1:], r[2:],r[3::2]], axis=0)
    index=np.zeros((1,int(n+(n-1)+(n-2)+(n-2)/2),3),dtype=np.int32)
    for looper in range(1):
        tester=np.ones((int(n+(n-1)+(n-2)+(n-2)/2),))*(looper)
        idxx = tf.stack([tester,ii, jj], axis=1)
        index[looper,:,:]=idxx

    index_tf3=tf.constant(index)
    
    


    for mc in range(100):
        X_am=model.reparameterize(model.X[sa,:,:],5.*model.X_v[sa,:,:])
        z_m,z_d,z_u=model.amortized(tf.reshape(X_am,[1,40,25]))
        z_rt=[]
        z_it=[]
        diag = 1.e-1+tf.math.exp(z_d)
        for tt in range(model.hN):
          values = tf.concat([diag[:,(tt)*80:(tt+1)*80], z_u[:,(tt)*(79+78+39):(tt+1)*(79+78+39)]], axis=1)
          Cov=tf.scatter_nd(index_tf3, values, [1,n, n])
          z1 = reparameterize_full_fast(z_m[:,(tt)*80:(tt+1)*80], Cov)
          z_r_tmp=z1[:,::2]
          z_i_tmp=z1[:,1::2]
          z_rt.append(z_r_tmp)
          z_it.append(z_i_tmp)

        z_r_stacked=tf.stack(z_rt)
        z_i_stacked=tf.stack(z_it)
        z_r_stacked2=tf.transpose(z_r_stacked,[1,2,0])
        z_i_stacked2=tf.transpose(z_i_stacked,[1,2,0])
        z_rmc[mc,:,:]=z_r_stacked2[0,:,:]
        z_imc[mc,:,:]=z_i_stacked2[0,:,:]

    z_r=np.mean(z_rmc[:,ti,:,],axis=0)
    z_i=np.mean(z_imc[:,ti,:,],axis=0)
    for timer in range(timert):
        for i in range(5):
            z_new_r[i,timer]=np.exp(-timer*lamb_r[i]**2)*np.cos(lamb_i[i]*timer)*z_r[i]-np.exp(-timer*lamb_r[i]**2)*np.sin(timer*lamb_i[i])*z_i[i]
            z_new_i[i,timer]=np.exp(-timer*lamb_r[i]**2)*np.cos(lamb_i[i]*timer)*z_i[i]+np.exp(-timer*lamb_r[i]**2)*np.sin(timer*lamb_i[i])*z_r[i]
    
    z_r=np.mean(z_rmc[:,:,:,],axis=0)
    z_i=np.mean(z_imc[:,:,:,],axis=0)
    
    return z_r,z_i,z_new_r,z_new_i
    
