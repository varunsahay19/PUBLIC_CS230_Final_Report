# (c) 2024 Patrick Nieman and Varun Sahay
# Trains a linear model to interpolate magnitude timesteps in a variable manner

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as p
from utilLoss import *
import architectures

tf.compat.v1.enable_eager_execution()

length=4096
testSplit=0.05

examples=53960

#Load X and Y
x=np.load("/Applications/CS230 Data/Export/condensedOutput%s.npy"%length)
y=np.load("/Applications/CS230 Data/Export/condensedOutputIntermediate%s.npy"%length)
x=x[:examples,:]
y=y[:examples,:]

#Split data (pre-shuffled)
m=x.shape[0]
index=int(np.floor(m*(1-testSplit)))
xTrain=x[1:index,:]
yTrain=y[1:index,:]
xTest=x[index:,:]
yTest=y[index:,:]

#Instantiate the model
model=architectures.expansion()

#Complile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=expansion,metrics=['mse'])

#Train model with checkpointing
checkpointPath = 'Applications/CS230 Data/Checkpoints/modelConv-{epoch:02d}-{val_loss:.5f}.keras'
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPath,save_weights_only=False,save_best_only=True,monitor='val_loss',mode='min')
history=model.fit(xTrain, yTrain, epochs=96, batch_size=64, validation_split=testSplit,callbacks=[checkpoint])
model.evaluate(xTest,yTest)
print(model.summary())
#Save model
model.save("/Applications/CS230 Data/expansionModel.keras")
p.figure(1)
p.plot(history.history['val_loss'])
p.show()

