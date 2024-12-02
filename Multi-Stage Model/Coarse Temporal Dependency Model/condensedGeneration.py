# (c) 2024 Patrick Nieman and Varun Sahay
# Generate intermediate training data for coarse temporal dependency model

import numpy as np
import os

length=4096
subdivisions=16
metadataPath="/Applications/CS230 Data/timeSeriesMetadata.json"
recordPath="/Applications/CS230 Data/Rotated Records 4096"
rsns=np.load("/Applications/CS230 Data/order.npy")
na=10
condensed=np.zeros((len(rsns)*na,subdivisions))
count=0

for rsn in rsns:
    rdata=np.atleast_2d(np.load(os.path.join(recordPath,f'{rsn}.npy')))
    rdata=rdata[:,:length]
    rdata=np.reshape(rdata,newshape=(subdivisions*na,-1))
    rdata=np.mean(np.abs(rdata),axis=1,keepdims=True)
    rdata=np.reshape(rdata,newshape=(na,-1))
    rdata/=np.max(np.abs(rdata),axis=1,keepdims=True)
    condensed[count*na:(count+1)*na,:]=rdata
    count+=1
    if count%100==0:
        print(count)
np.save("/Applications/CS230 Data/Export/condensedOutput%s.npy"%length,condensed)
