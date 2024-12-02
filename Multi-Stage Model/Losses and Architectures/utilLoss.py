# (c) 2024 Patrick Nieman and Varun Sahay
# Custom low-level loss functions and metrics

import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

#Compares the error as referred to both the original and predicted values
@register_keras_serializable()
def elementAverageRelativeAbsolute(y,yhat):
    y=tf.squeeze(y)
    yhat=tf.squeeze(yhat)
    return tf.reduce_mean(tf.divide(tf.abs(tf.subtract(y,yhat)),tf.abs(y)+1e-8))

#A variant of the above
@register_keras_serializable()
def relativeBiabsolute(y,yhat):
    absDiff=tf.abs(tf.subtract(y,yhat))
    return tf.divide(tf.reduce_sum([tf.abs(tf.reduce_mean(tf.divide(absDiff,y+1e-8))),tf.abs(tf.reduce_mean(tf.divide(absDiff,yhat+1e-8)))]),2)

#Compares in 16 buckets the standard deviation in a MSE fashion
def varianceLoss(y,yhat):
    n=16
    yVar=tf.math.reduce_std(tf.reshape(y,(n,-1)),axis=1)
    yhatVar=tf.math.reduce_std(tf.reshape(yhat,(n,-1)),axis=1)
    return tf.reduce_mean(tf.square(tf.subtract(yVar,yhatVar)))

#Compares in 16 buckets the mean in a MSE fashion
def meanLoss(y,yhat):
    n=16
    yVar=tf.math.reduce_mean(tf.reshape(y,(n,-1)),axis=1)
    yhatVar=tf.math.reduce_mean(tf.reshape(yhat,(n,-1)),axis=1)
    return tf.reduce_mean(tf.square(tf.subtract(yVar,yhatVar)))

#Composite loss used for training the interpolation model
@register_keras_serializable()
def expansion(y,yhat):
    mseLoss=tf.reduce_mean(tf.square(tf.subtract(y,yhat)))
    return mseLoss+varianceLoss(y,yhat)+meanLoss(y,yhat)