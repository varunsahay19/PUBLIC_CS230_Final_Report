# (c) 2024 Patrick Nieman and Varun Sahay
# Utility function to determine the distribution of record durations

import json
import numpy as np
import os
from matplotlib import pyplot as p
metadataPath="/Applications/CS230 Data/timeSeriesMetadata.json"
recordPath="/Applications/CS230 Data/Records"
with open(metadataPath,"r") as f:
    database=json.loads(f.read())
duration=[]
count=0
for k in database:
    r=database[k]
    for i in range(2):
        di=np.atleast_2d(np.loadtxt(os.path.join(recordPath,f'{k}-{r["d"][i]}.csv')))
        dt=max(di.shape)*float(r["dt"][i])
        duration.append(dt)
    count+=1
    if count%100==0:
        print(count)
p.plot(duration)
p.show()
