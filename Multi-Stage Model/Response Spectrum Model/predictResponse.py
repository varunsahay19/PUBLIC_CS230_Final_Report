# (c) 2024 Patrick Nieman and Varun Sahay
# Plots example predictions for the response spectrum model

import numpy as np
import tensorflow.keras.models as tf
from matplotlib import pyplot as p

t=np.load("/Applications/CS230 Data/Export/spectraPeriods.npy")
print(t.shape)
xpath="/Applications/CS230 Data/Export/spectraInput.npy"
ypath="/Applications/CS230 Data/Export/spectraOutput.npy"

plots=8
x=np.load(xpath)
y=np.load(ypath)
print(x.shape)
print(y.shape)
f,ax=p.subplots(plots)

model=tf.load_model('/Applications/CS230 Data/responseModel.keras')
for i in range(plots):
    ax[i].plot(t,y[i,:],label="Truth")
    spectrum=model.predict(np.atleast_2d(x[i,:]))
    ax[i].plot(t,np.squeeze(spectrum),label="Prediction")
    ax[i].legend()
p.show()
