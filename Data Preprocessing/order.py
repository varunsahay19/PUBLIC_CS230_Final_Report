# (c) 2024 Patrick Nieman and Varun Sahay
# Establishes consistent randomized order for data, for convenience during preprocessing

import numpy as np
import json
n=5396
rsns=[]
database=json.loads(open("/Applications/CS230 Data/timeSeriesMetadata.json","r").read())
for i in database:
    rsns.append(i)

rsns=np.array(rsns)
np.random.seed(12)
shuffle=np.arange(n)
np.random.shuffle(shuffle)
rsns=rsns[shuffle]
np.save("/Applications/CS230 Data/order.npy",rsns)