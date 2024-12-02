# (c) 2024 Patrick Nieman and Varun Sahay
# Compiles X and Y datasets for training linear ground motion model, from a subset of the full dataset if specified

import numpy as np
import shutil
import json

subinterpolation=" 4096"
length=1024
enforceLength=False

#Utility
def pad(a,n):
    out=""
    for i in range(n-len(a)):
        out=out+"0"
    return out+a

def padfr(a,n):
    a=str(int(round(float(a))))
    return str(pad(a,n))

n=7477 #Total dataset size
max=5396#5396 #desired records; number of X, Y entries will be max*angles
angles=10 #Per record file
indicesI=np.arange(n)

mapPath="Rock Query/requests.csv"
pathPath="/Applications/CS230 Data/pathData201h.csv"
metadataPath="/Applications/CS230 Data/metadata.csv"
tsmdPath="/Applications/CS230 Data/timeSeriesMetadata.json"

#Get ordered record IDs for available records
with open(mapPath,"r") as f:
    list=np.array(f.read().replace("\n",",").split(","))
rsnsI=list[indicesI]
indices=[]

rsns=np.load("/Applications/CS230 Data/order.npy")

#Load rock paths for each record
inputPath=[]
with open(pathPath,"r") as f:
    inputPath=f.read().split("\n")[:-1]

fullOrder=np.load("/Applications/CS230 Data/orderFull.npy")
inputOrdered={}
k=0
for i in inputPath:
    #print(padfr(fullOrder[k],5))
    inputOrdered[padfr(fullOrder[k],5)]=i
    k+=1


#Load event and station metadata for each record
metadata=[]
input=[]
with open(metadataPath,"r") as f:
    mdata={}
    data=f.read().split("\n")
    for d in data:
        mdata[d.split("||")[0]]=d
    for i in rsns:
        input.append(inputOrdered[i]+","+mdata[str(float(i))].replace("||",",").split(",",1)[-1])


#Expand inputs for each rotated angle
with open("/Applications/CS230 Data/Export/inputExpanded.csv","w") as o:
    for i in input:
        for j in range(angles):
            o.write(i)
            o.write(",%s"%j)
            o.write("\n")

#Save for reference the time intervals of each record
with open("/Applications/CS230 Data/Export/timeSeriesDTsExpanded.csv","w") as o2:
    with open("/Applications/CS230 Data/Export/timeSeriesDTs.csv","w") as o:
        with open(tsmdPath,"r") as f:
            tsMetadata=json.loads(f.read())
            for i in rsns:
                text="%s\n"%tsMetadata[str(pad(i,5))]["dt"][0]
                o.write(text)
                for j in range(angles):
                    o2.write(text)

#Concatenate all rotated records into one array
k=0
for i in range(len(rsns)):
    records=np.load("/Applications/CS230 Data/Rotated Records%s/%s.npy"%(subinterpolation,rsns[i]))
    if i==0:
        output=np.zeros((angles*len(rsns),length if enforceLength else records.shape[1]))
    output[k:k+angles,:]=records[:,0:(length if enforceLength else records.shape[1])]
    k+=angles

#Save Y and copy over useful files
np.save("/Applications/CS230 Data/Export/output%s.npy"%(length if enforceLength else ""),output)
shutil.copy("/Applications/CS230 Data/metadataHeaders.csv","/Applications/CS230 Data/Export/metadataHeaders.csv")
shutil.copy("/Applications/CS230 Data/rocks.csv","/Applications/CS230 Data/Export/rocks.csv")