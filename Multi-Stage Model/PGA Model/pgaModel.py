# (c) 2024 Patrick Nieman and Varun Sahay
# Trains a linear network to predict peak ground acceleration from metadata

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as p
from utilLoss import *
import architectures

reduceInput=True

tf.compat.v1.enable_eager_execution()
xpath="/Applications/CS230 Data/Export/inputExpanded.csv"
ypath="/Applications/CS230 Data/PGA/pgaOutput.npy"

#Load X and Y and expand dimensions
x=np.loadtxt(xpath,delimiter=",")
y=np.load(ypath)

#Normalize except for modified one-hot rock path classification
metaStart=74
x[:,metaStart:]=x[:,metaStart:]-np.mean(x[:,metaStart:],axis=0)
x[:,metaStart:]=x[:,metaStart:]/np.std(x[:,metaStart:],axis=0)

if reduceInput:
    x=x[:,metaStart:]

#Split data (pre-shuffled)
m=x.shape[0]
testSplit=0.05
index=int(np.floor(m*(1-testSplit)))
xTrain=x[0:index,:]
yTrain=y[0:index,:]
xTest=x[index:,:]
yTest=y[index:,:]

model=architectures.pga2()

#Complile model
decayRate=-tf.math.log(0.25)/172
learningRate=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,decay_steps=100,decay_rate=tf.exp(decayRate),staircase=False)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss=relativeBiabsolute)

#Train model with checkpointing
checkpointPath = 'Applications/CS230 Data/epoch-{epoch:02d}-val_loss-{val_loss:.2f}.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPath,save_weights_only=False,save_best_only=True,monitor='val_loss',mode='min')
history=model.fit(xTrain, yTrain, epochs=216, batch_size=64, validation_split=testSplit,callbacks=[checkpoint])
print(model.evaluate(xTest,yTest))

#Save model
model.save('/Applications/CS230 Data/pgaModel.keras')
p.figure(1)
p.plot(history.history['val_loss'])
p.show()