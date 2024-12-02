# (c) 2024 Patrick Nieman and Varun Sahay
# Utility function to assess distribution of 5-95% or 5-75% significant durations

import json
import numpy as np
import os
from matplotlib import pyplot as p

metadataPath="/Applications/CS230 Data/timeSeriesMetadata.json"
recordPath="/Applications/CS230 Data/Records"

with open(metadataPath,"r") as f:
    database=json.loads(f.read())

ds595=[]
count=0
for k in database:
    r=database[k]
    for i in range(2):
        di=np.atleast_2d(np.loadtxt(os.path.join(recordPath,f'{k}-{r["d"][i]}.csv')))
        ai=np.sum(np.square(di))
        dt=float(r["dt"][i])
        
        j=0
        total=0
        while j<di.shape[1]:
            total+=di[0,j]**2
            if total>=0.75*ai:
                ds595.append((j+1)*dt)
                break
            j+=1
    count+=1
    if count%100==0:
        print(count)

p.figure(1)
p.hist(ds595)
p.xlabel("Duration (s)")
p.ylabel("Number of records")

p.figure(2)
p.ecdf(ds595)
p.xlabel("Duration (s)")
p.ylabel("Proportion of records at or below duration")

p.show()
