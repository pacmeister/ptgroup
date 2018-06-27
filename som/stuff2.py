from root_numpy import root2array, tree2array, array2root
from root_numpy import testdata
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime
from som import SOM

def cluster(numpyndarray, dimx, dimy):
    count = np.empty([dimx,dimy])
    for i in range(dimx):
        for j in range(dimy):
            count[i,j] = numpyndarray[np.where((numpyndarray[:,8] == float(i)) & (numpyndarray[:,9] == float(j))),1].size
    return count

def sigcount(numpyndarray, numpyndarray2, dimx, dimy):
    count_sq = np.empty([dimx,dimy])
    for i in range(dimx):
        for j in range(dimy):
            count_sq[i,j] = numpyndarray[np.where((numpyndarray[:,8] == float(i)) & (numpyndarray[:,9] == float(j))),10].sum().astype(int)/(numpyndarray2[i,j]+1)
    return count_sq

def bgcount(numpyndarray, numpyndarray2,dimx, dimy):
    count_sq_bg = np.empty([dimx,dimy])
    for i in range(dimx):
        for j in range(dimy):
            count_sq_bg[i,j] = (count[i,j] - numpyndarray[np.where((numpyndarray[:,8] == float(i)) & (numpyndarray[:,9] == float(j))),10].sum().astype(int))/(numpyndarray2[i,j]+1)
    return count_sq_bg

# Print time, useful for determining how long it takes to train the self-organizing map
print(str(datetime.datetime.now()))
dimx = 10
dimy = 10
sig = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeS')
bg = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeB')
#sig = np.concatenate(sig[24],sig[29],sig[30],sig[31],sig[34], axis=1)
sig = np.array(sig.tolist())
sig = sig[:10000,:]
sasquatch_sig = sig[:,35]
sig = sig[:,[24,25,26,29,30,31,32,34]]
#sig = sig[:,:37]
bg = np.array(bg.tolist())
bg = bg[:10000,:]
sasquatch_bg = bg[:,35]
bg = bg[:,[24,25,26,29,30,31,32,34]]
#bg = bg[,:37]
data = np.concatenate((sig,bg),axis=0)
sasquatch = np.concatenate((sasquatch_sig, sasquatch_bg), axis=0)
mapped = np.array(som.map_vects(data))
data = np.append(data,mapped,axis=1)
data = np.column_stack((data,sasquatch))
count = cluster(data,dimx,dimy)
count_sq = sigcount(data,dimx,dimy)
count_sq_bg = bgcount(data,dimx,dimy)
#Train a 10x10 SOM with 400 iterations, 8 is the number of variables in the data array
som = SOM(dimx, dimy, 8, 400)
som.train(data)
print(str(datetime.datetime.now()))

plt.imshow(count, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

sig = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeS')
bg = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeB')
#sig = np.concatenate(sig[24],sig[29],sig[30],sig[31],sig[34], axis=1)
sig = np.array(sig.tolist())
sig = sig[10000:,:]
sasquatch_sig = sig[:,35]
sig = sig[:,[24,25,26,29,30,31,32,34]]
#sig = sig[:,:37]
bg = np.array(bg.tolist())
bg = bg[10000:,:]
sasquatch_bg = bg[:,35]
bg = bg[:,[24,25,26,29,30,31,32,34]]
#bg = bg[,:37]
data20k = np.concatenate((sig,bg),axis=0)
sasquatch20k = np.concatenate((sasquatch_sig, sasquatch_bg), axis=0)
mapped = np.array(som.map_vects(data20k))
data20k = np.append(data20k,mapped,axis=1)
data20k = np.column_stack((data20k,sasquatch20k))
count20k = cluster(data20k,dimx,dimy)
count_sq20k = sigcount(data20k,count20k,dimx,dimy)
count_sq_bg20k = bgcount(data20k,count20k,dimx,dimy)

sig = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeS')
bg = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeB')
#sig = np.concatenate(sig[24],sig[29],sig[30],sig[31],sig[34], axis=1)
sig = np.array(sig.tolist())
sig = sig[15000:,:]
sasquatch_sig = sig[:,35]
sig = sig[:,[24,25,26,29,30,31,32,34]]
#sig = sig[:,:37]
bg = np.array(bg.tolist())
bg = bg[15000:,:]
sasquatch_bg = bg[:,35]
bg = bg[:,[24,25,26,29,30,31,32,34]]
#bg = bg[,:37]
data15k = np.concatenate((sig,bg),axis=0)
sasquatch15k = np.concatenate((sasquatch_sig, sasquatch_bg), axis=0)
mapped = np.array(som.map_vects(data15k))
data15k = np.append(data15k,mapped,axis=1)
data15k = np.column_stack((data15k,sasquatch15k))
count15k = cluster(data15k,dimx,dimy)
count_sq15k = sigcount(data15k,count15k,dimx,dimy)
count_sq_bg15k = bgcount(data15k,count15k,dimx,dimy)

sig = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeS')
bg = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeB')
#sig = np.concatenate(sig[24],sig[29],sig[30],sig[31],sig[34], axis=1)
sig = np.array(sig.tolist())
sig = sig[20000:,:]
sasquatch_sig = sig[:,35]
sig = sig[:,[24,25,26,29,30,31,32,34]]
#sig = sig[:,:37]
bg = np.array(bg.tolist())
bg = bg[20000:,:]
sasquatch_bg = bg[:,35]
bg = bg[:,[24,25,26,29,30,31,32,34]]
#bg = bg[,:37]
data10k = np.concatenate((sig,bg),axis=0)
sasquatch10k = np.concatenate((sasquatch_sig, sasquatch_bg), axis=0)
mapped = np.array(som.map_vects(data10k))
data10k = np.append(data10k,mapped,axis=1)
data10k = np.column_stack((data10k,sasquatch10k))
count10k = cluster(data10k,dimx,dimy)
count10k = cluster(data10k,dimx,dimy)
count_sq10k = sigcount(data10k,count10k,dimx,dimy)
count_sq_b10k = bgcount(data10k,count10k,dimx,dimy)

sig = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeS')
bg = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeB')
#sig = np.concatenate(sig[24],sig[29],sig[30],sig[31],sig[34], axis=1)
sig = np.array(sig.tolist())
sig = sig[25000:,:]
sasquatch_sig = sig[:,35]
sig = sig[:,[24,25,26,29,30,31,32,34]]
#sig = sig[:,:37]
bg = np.array(bg.tolist())
bg diff = np.empty([5])
diff[0] = abs(count_sq - count_sq2k).mean()
diff[1] = abs(count_sq - count_sq5k).mean()
diff[2] = abs(count_sq - count_sq10k).mean()
diff[3] = abs(count_sq - count_sq15k).mean()
diff[4] = abs(count_sq - count_sq20k).mean()= bg[25000:,:]
sasquatch_bg = bg[:,35]
bg = bg[:,[24,25,26,29,30,31,32,34]]
#bg = bg[,:37]
data5k = np.concatenate((sig,bg),axis=0)
sasquatch5k = np.concatenate((sasquatch_sig, sasquatch_bg), axis=0)
mapped = np.array(som.map_vects(data5k))
data5k = np.append(data5k,mapped,axis=1)
data5k = np.column_stack((data5k,sasquatch5k))
count5k = cluster(data5k,dimx,dimy)
count_sq5k = sigcount(data5k, count5k, dimx,dimy)
count_sq_b5k = bgcount(data5k,count5k, dimx,dimy)

sig = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeS')
bg = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeB')
#sig = np.concatenate(sig[24],sig[29],sig[30],sig[31],sig[34], axis=1)
sig = np.array(sig.tolist())
sig = sig[28000:,:]
sasquatch_sig = sig[:,35]
sig = sig[:,[24,25,26,29,30,31,32,34]]
#sig = sig[:,:37]
bg = np.array(bg.tolist())
bg = bg[28000:,:]
sasquatch_bg = bg[:,35]
bg = bg[:,[24,25,26,29,30,31,32,34]]
#bg = bg[,:37]
data2k = np.concatenate((sig,bg),axis=0)
sasquatch2k = np.concatenate((sasquatch_sig, sasquatch_bg), axis=0)
mapped = np.array(som.map_vects(data2k))
data2k = np.append(data2k,mapped,axis=1)
data2k = np.column_stack((data2k,sasquatch2k))
count2k = cluster(data2k,dimx,dimy)
count_sq2k = sigcount(data2k, count2k, dimx,dimy)
count_sq_b2k = bgcount(data2k,count2k, dimx,dimy)

diff = np.empty([5])
diff[0] = abs(count_sq - count_sq2k).mean()
diff[1] = abs(count_sq - count_sq5k).mean()
diff[2] = abs(count_sq - count_sq10k).mean()
diff[3] = abs(count_sq - count_sq15k).mean()
diff[4] = abs(count_sq - count_sq20k).mean()
