# (c) 2024 Patrick Nieman and Varun Sahay
# The coarse time dependency stage model, predicting reduced timesteps in a linear model from metadata

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as p
from loss import *
import architectures

reduceInput=False
normalize=True
tf.compat.v1.enable_eager_execution()

testSplit=0.05
examples=53960

#Load X and Y
x=np.loadtxt("/Applications/CS230 Data/Export/inputExpanded.csv",delimiter=",")
y=np.load("/Applications/CS230 Data/Export/condensedOutput4096.npy")

if reduceInput:
    x=x[:,74:]
x=x[:examples,:]
y=y[:examples,:]

#Normalize except for modified one-hot rock path classification
if reduceInput:
    x=x-np.mean(x,axis=0)
    x=x/np.std(x,axis=0)
else:
    x[:,74:]=x[:,74:]-np.mean(x[:,74:],axis=0)
    x[:,74:]=x[:,74:]/np.std(x[:,74:],axis=0)
if normalize:
    y/=np.max(np.abs(y),axis=1,keepdims=True)

#Split data (pre-shuffled)
m=x.shape[0]
index=int(np.floor(m*(1-testSplit)))
xTrain=x[1:index,:]
yTrain=y[1:index,:]
xTest=x[index:,:]
yTest=y[index:,:]

#Instantiate the model
if reduceInput:
    model=architectures.recurrentCondensedAlternativeReducedInput()
else:
    model=architectures.recurrentCondensedAlternative()

#Complile model
decayRate=-tf.math.log(0.25)/100
learningRate=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,decay_steps=100,decay_rate=tf.exp(decayRate),staircase=False)
learningRate=0.001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate,clipnorm=1.0),loss='mae')

#Train model with checkpointing
checkpointPath = 'Applications/CS230 Data/Checkpoints/modelConv-{epoch:02d}-{val_loss:.5f}.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPath,save_weights_only=False,save_best_only=True,monitor='val_loss',mode='min')
history=model.fit(xTrain, yTrain, epochs=8, batch_size=64, validation_split=testSplit,callbacks=[checkpoint])
model.evaluate(xTest,yTest)
print(model.summary())
#Save model
model.save("/Applications/CS230 Data/condensedModel.keras")
p.figure(1)
p.plot(history.history['val_loss'])
p.show()

