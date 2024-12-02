# (c) 2024 Patrick Nieman and Varun Sahay
# Calculates PGA for dataset records to create training data for PGA model

import numpy as np

kmax=5396

rsns=np.load("/Applications/CS230 Data/order.npy")
k=0
ri=0
#For each rotated record
for rsn in rsns:
    #Load records
    records=np.load(f"/Applications/CS230 Data/Rotated Records/{rsn}.npy")
    na=records.shape[0]
    
    #Calculate pga
    pga=np.max(np.abs(records),axis=1)

    if k==0: #Initialize array
        output=np.zeros((na*kmax,1))
    output[ri:ri+na,0]=pga

    ri+=na
    k+=1
    if k%100==0:
        print(k)
    if k==kmax:
        break

#Save output data for model training
np.save("/Applications/CS230 Data/PGA/pgaOutput.npy",output)