# (c) 2024 Patrick Nieman and Varun Sahay
# Custom loss functions and metrics

import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from utilLoss import *
import numpy as np

#Model to predict response specrtra of predicted records
responsePath="/Applications/CS230 Data/responseModel.keras"
responseModel=tf.keras.models.load_model(responsePath)
responseModel.trainable=False

#Normalized MSE of predicted response spectra
@register_keras_serializable()
def spectrum(y,yhat):
    ySpectrum=responseModel(y)
    yhatSpectrum=responseModel(yhat)
    return tf.divide(tf.divide(tf.reduce_mean(tf.square(tf.subtract(ySpectrum,yhatSpectrum))),tf.reduce_mean(tf.square(ySpectrum))+1e-8),0.01)

#Non-tensorflow version of the above
def spectrumNP(y,yhat):
    ySpectrum=responseModel(y)
    yhatSpectrum=responseModel(yhat)
    return np.mean(np.abs(ySpectrum-yhatSpectrum)/ySpectrum,axis=1)

#Normalized MSE of predicted response spectra
@register_keras_serializable()
def spectrumNormalized(y,yhat):
    ySpectrum=responseModel(y)
    ySpectrum=tf.divide(ySpectrum,tf.reduce_mean(ySpectrum,axis=1,keepdims=True))
    yhatSpectrum=responseModel(yhat)
    yhatSpectrum=tf.divide(yhatSpectrum,tf.reduce_mean(yhatSpectrum,axis=1,keepdims=True))
    return tf.reduce_mean(tf.square(tf.subtract(ySpectrum,yhatSpectrum)))

#Relative difference in Arias intensities
@register_keras_serializable()
def arias(y,yhat):
    aly=tf.reduce_sum(tf.square(y))
    alyhat=tf.reduce_sum(tf.square(yhat))
    return tf.divide(tf.abs(tf.subtract(aly,alyhat)),aly+1e-8)

#Non-tensorflow version of the above
def ariasNP(y,yhat):
    aly=np.sum(np.square(y),axis=1)
    alyhat=np.sum(np.square(yhat),axis=1)
    return np.abs(aly-alyhat)/(aly+1e-8)

#Direct MSE comparison of records, normalized
@register_keras_serializable()
def motion(y,yhat):
    return tf.divide(tf.reduce_mean(tf.square(tf.subtract(tf.abs(y),tf.abs(yhat)))),tf.reduce_mean(tf.square(y))+1e-8)

#Direct MSE comparison of records, averaged in 16 bins, normalized, weighted to record start
@register_keras_serializable()
def smearedMotion(y,yhat):
    n=64
    ySmeared=tf.reduce_mean(tf.abs(tf.reshape(y,(n,-1))),axis=1)
    yhatSmeared=tf.reduce_mean(tf.abs(tf.reshape(yhat,(n,-1))),axis=1)
    return tf.divide(tf.reduce_mean(tf.square(tf.subtract(ySmeared,yhatSmeared))),tf.reduce_mean(tf.square(y))+1e-8)

#Non-tensorflow version of the above
def smearedNP(y,yhat):
    n=64
    ySmeared=np.mean(np.abs(np.reshape(y,(n,-1))),axis=1)
    yhatSmeared=np.mean(np.abs(np.reshape(yhat,(n,-1))),axis=1)
    return np.mean(np.abs(ySmeared-yhatSmeared)/(ySmeared+1e-4))

#Direct MSE comparison of records, averaged in 16 bins, normalized, weighted to record start
@register_keras_serializable()
def biSmearedMotion(y,yhat):
    n=64
    ySmeared=tf.reduce_mean(tf.abs(tf.reshape(y,(n,-1))),axis=1)
    yhatSmeared=tf.reduce_mean(tf.abs(tf.reshape(yhat,(n,-1))),axis=1)
    error=tf.reduce_mean(tf.abs(tf.subtract(ySmeared,yhatSmeared)))
    return tf.reduce_mean([tf.divide(error,tf.reduce_mean(tf.abs(y))+1e-8),tf.divide(error,tf.reduce_mean(tf.abs(yhat))+1e-8)])

#Time of peak acceleration
@register_keras_serializable()
def recordPeak(y,yhat):
    return tf.reduce_max(tf.divide(tf.abs(tf.subtract(tf.argmax(tf.abs(y),axis=1),tf.argmax(tf.abs(yhat),axis=1))),300))

#Custom loss model
@register_keras_serializable()
def responseLoss(y,yhat):
    spectrumLoss=spectrum(y,yhat)
    ariasLoss=arias(y,yhat)
    smearedMotionLoss=smearedMotion(y,yhat)
    return tf.divide(tf.reduce_sum([tf.cast(spectrumLoss,tf.float64),tf.cast(ariasLoss,tf.float64),tf.cast(smearedMotionLoss,tf.float64)]),3.0)

#Full-record Fourier transform MSE loss
@register_keras_serializable()
def fourier(y,yhat):
    yfft = tf.abs(tf.signal.fft(tf.cast(y,tf.complex64)))
    yhatfft = tf.abs(tf.signal.fft(tf.cast(yhat,tf.complex64)))
    return tf.reduce_mean(tf.square(tf.subtract(yfft,yhatfft)))

#Low-frequency loss for the oscillation model
@register_keras_serializable()
def convLossLF(y,yhat):
    avg=meanLoss(y,yhat)
    stdev=varianceLoss(y,yhat)
    fourierLoss=fourier(y,yhat)
    return avg+stdev+0.03*fourierLoss

#High-frequency loss for the oscillation model
@register_keras_serializable()
def convLossHF(y,yhat):
    mseLoss=tf.reduce_mean(tf.square(tf.subtract(y,yhat)))
    avg=meanLoss(y,yhat)
    stdev=varianceLoss(y,yhat)
    return avg+stdev+mseLoss
