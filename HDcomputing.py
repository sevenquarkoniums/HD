#!/usr/bin/env python
"""



run by:


### TODO ###
get nonConstantMetrics from all data.

### warning ###
1 need to use a fixed random seed.
2 be careful with printing without newline.
3 only using input X.

"""
#=========================


#=========================
import datetime
now = datetime.datetime.now()

import pandas as pd
import numpy as np
import math
import os.path
import timeit
import matplotlib.pyplot as plt
plt.switch_backend('agg') # for running in linux batch job.

def main():
#    drawTPDS()
    #readResults()
    #apps()
#    generateFold()
#    updateFold()
#    normal()
    #drawFromResult()
    scc()
    #gridSearch()
    #windowSize()
    #dimension()
    #downSample()
    #seed()
    #metricNum()
    #noise()
    #reshape('C:/Programming/monitoring/data/bt_X/work_search_allbt.csv')

def normal():
    hd = HD(env='dell', mode='work', outputEvery=False, trainSliding=False, windowSize=1, downSample=100, 
            dimension=10000, trainMethod='closest', seed=0, selectApp='all', 
            selectIntensity=[100], metadata=None, anomalyTrain='bt', noPreparedData=True,
            storeEncoding=False, simVecThres=0.5, onlyOneFold=False, invertTrainTest=False)#['mg','kripke','lu']
    if hd.noPreparedData:
        hd.genMetricVecs()
        hd.normalize()
    #    hd.diagnose()
    #    hd.draw()
        hd.slidingWindow()
    #    hd.baseline()
        hd.encoding()
    #    hd.checkCorrelation()
    else:
        hd.readEncoded()
    hd.trainTest()
    hd.confusionMatrix(suffix='')

def scc():
    import sys
    hd = HD(env='scc', mode='work', outputEvery=False, trainSliding=False, windowSize=int(sys.argv[1]), downSample=int(sys.argv[2]), 
                dimension=int(sys.argv[4]), trainMethod=sys.argv[3], seed=int(sys.argv[5]), selectApp='all', 
                selectIntensity=[100], metadata=None, anomalyTrain=sys.argv[6], 
                simVecThres=0.5, onlyOneFold=False, invertTrainTest=False)
    hd.genMetricVecs()
    hd.normalize()
    hd.slidingWindow()
    hd.encoding()
    hd.trainTest()
    hd.confusionMatrix()
    
class HD:
    def __init__(self, env='dell', mode='work', outFile=None, outputEvery=False, windowSize=5, trainMethod='closest', downSample=1, 
                 dimension=10000, seed=0, fakeMetricNum=8, fakeNoiseScale=0.3, trainSliding=False, noPreparedData=True, selectApp='all', 
                 selectIntensity=[20,50,100], anomalyTrain='all', metadata=None, storeEncoding=False, simVecThres=0.01, onlyOneFold=False,
                 invertTrainTest=False):
        print('======================')
        self.env = env
        self.mode = mode# work, check.
        self.fout = outFile
        self.outputEvery = outputEvery
        self.trainSliding = trainSliding
        self.trainMethod = trainMethod# HDadd, closest, addFilter. Closest method is the best but with large overhead. addFilter method can be good.
        self.windowSize = windowSize
        self.dimension = dimension # seems larger dimension won't help much.
        self.downSample = downSample
        self.seed = seed
        self.selectApp = selectApp
        self.selectIntensity = selectIntensity
        self.anomalyTrain = anomalyTrain
        self.metadata = metadata
        self.noPreparedData = noPreparedData
        self.storeEncoding = storeEncoding
        self.simVecThres = simVecThres
        self.onlyOneFold = onlyOneFold
        self.invertTrainTest = invertTrainTest
        np.random.seed(self.seed) # not sensitive to this.
        
        if self.env == 'dell':
            self.dataFolder = 'C:/Programming/monitoring/data'
            self.metricFileName = 'C:/Programming/monitoring/HD/nonConstantMetrics.txt'
        elif self.env == 'scc':
            self.dataFolder = '/projectnb/peaclab-mon/yijia/data/volta'
            self.metricFileName = '/projectnb/peaclab-mon/yijia/HD/nonConstantMetrics.txt'
        self.numFold = 5
        self.closeCriterion = 0.05 # for d=10000, 0.01 is too small.
        self.sequenceMode = 'shift'# shift, permutate. Random permutate is 100x slow.
        self.permutSeed = 0
        self.rawData = {}
        self.length, self.start = {}, {}
        print('window size: %d, dimension: %d, seed: %d, downsampling: %d, train method: %s' % 
                    (self.windowSize, self.dimension, self.seed, self.downSample, self.trainMethod))
        if self.mode == 'work':
            self.marginCut = 60 # TPDS is using 30.
            self.types = ['dcopy','leak','linkclog','dial','memeater','none']# # this order will show in the matrix.
            print(self.types)
            if self.metadata != None:
                if self.env == 'dell':
                    self.metafile = pd.read_csv('C:/Programming/monitoring/run_metadata_%d.csv' % self.metadata)
                elif self.env == 'scc':
                    self.metafile = pd.read_csv('/projectnb/peaclab-mon/cache/run_metadata_%d.csv' % self.metadata)
            if noPreparedData:
                self.readData()
        elif self.mode == 'check':
            self.metricNum = fakeMetricNum # larger value makes the result better.
            self.fakeNoiseScale = fakeNoiseScale
            if noPreparedData:
                self.genFakeSeries()
        # the following file is only for no-computing-drawing.
        self.resultFile = 'C:/Programming/monitoring/HDcomputing/results/results_window%d_downsample%d_trainWith%s_dim%d_seed%d_trainSliding.csv' % (self.windowSize,
                                                            self.downSample, self.trainMethod, self.dimension, self.seed)

    def readData(self):
        cumStart = 0
        metricFileExist = False
        if os.path.isfile(self.metricFileName):
            metricFileExist = True
            self.metrics = []
            with open(self.metricFileName, 'r') as metricFile:
                for line in metricFile:
                    self.metrics.append(line[:-1])
            print('Only use %d metrics from nonConstantMetrics.txt.' % len(self.metrics))
        if self.selectApp == 'all':
            self.apps = ['bt','cg','CoMD','ft','kripke','lu','mg','miniAMR','miniGhost','miniMD','sp']
        else:
            self.apps = self.selectApp
        for itype in self.types:
            print('reading type %s data..' % itype)
            self.rawData[itype] = {}
            # self.rawData:
            # {'dcopy':{(runID1, 1):DF1, (runID2, 1):DF2, (runID3, 1):DF3, ...}
            #  'none': ...
            # }
            self.length[itype], self.start[itype] = {}, {}
                        
            files = [] # [ ('%s/%s_X/%s_%d/%s', 'bt'), ...]
            for intensity in self.selectIntensity:
                if itype != 'none':
                    for iapp in self.apps:
                        temp = getfiles('%s/%s_X/%s_%d' % (self.dataFolder, iapp, itype, intensity))
                        files += [(x, iapp) for x in temp if int(x[-5])==1]# only anomalous.
                else:
                    for iapp in self.apps:
                        for jtype in self.types:# need to go into jtype folders to read none-type files.
                            if self.metadata == None:
                                temp = getfiles('%s/%s_X/%s_%d' % (self.dataFolder, iapp, jtype, intensity))
                                if jtype != 'none':
                                    temp = [(x, iapp) for x in temp if int(x[-5])!=1]# only healthy.
                                else:
                                    temp = [(x, iapp) for x in temp]
                                files += temp
                            else:
                                if jtype != 'none':
                                    temp = getfiles('%s/%s_X/%s_%d' % (self.dataFolder, iapp, jtype, intensity))
                                    temp = [(x, iapp) for x in temp if int(x[-5])!=1]# only healthy.
                                    files += temp
                                elif jtype == 'none':
                                    for jintensity in [20,50,100]:# need to go to all intensities because of the discrepency between dell and scc files.
                                        temp = getfiles('%s/%s_X/%s_%d' % (self.dataFolder, iapp, jtype, jintensity))
                                        for ifile in temp:
                                            if self.env == 'dell':
                                                runID = ifile.split('\\')[-1].split('_')[0]
                                            elif self.env == 'scc':
                                                runID = ifile.split('/')[-1].split('_')[0]
                                            if runID in list(self.metafile['runID']):
                                                files.append((ifile, iapp))
            fileIdx = 0
            for ifile in files:
                appName = ifile[1]
                if self.env == 'dell':
                    runID = ifile[0].split('\\')[-1].split('_')[0]
                elif self.env == 'scc':
                    runID = ifile[0].split('/')[-1].split('_')[0]
                node = int(ifile[0][-5])
                df = pd.read_csv(ifile[0]).iloc[self.marginCut:-self.marginCut]
                if metricFileExist:
                    df = df[['#Time']+self.metrics]
                if self.downSample > 1:
                    dfAvg = df.rolling(window=self.downSample).mean()
                    df = dfAvg[(self.downSample-1)::self.downSample]
                self.length[itype][(runID, node, appName)] = len(df)
                self.start[itype][(runID, node, appName)] = cumStart
                cumStart += len(df)
                self.rawData[itype][(runID, node, appName)] = df
                fileIdx += 1
        if not metricFileExist:
            self.metrics = list(self.rawData['none'][list(self.rawData['none'].keys())[0]].columns)
            self.metrics.remove('#Time')
        self.metricNum = len(self.metrics)
        print('Total trace number: %d' % sum([len(self.rawData[itype]) for itype in self.types]))

    def genFakeSeries(self):
        '''
        have indexing problem because of the index (runID, nodeNum, appName).
        '''
        self.metrics = list(range(self.metricNum))
        self.types = ['a','b','c','d','e','f']
        fakeLength = 200
        T, phase, self.length, self.start = {}, {}, {}, {}
        cumStart = 0
        for idx, itype in enumerate(self.types):
            T[itype] = np.random.uniform(low=1, high=400, size=self.metricNum)
            phase[itype] = np.random.uniform(low=-math.pi, high=math.pi, size=self.metricNum)
        for itype in self.types:
            self.rawData[itype] = {}
            self.length[itype] = {}
            self.start[itype] = {}
            for itrace in range(2):
                df = pd.DataFrame(index=range(fakeLength), columns=['#Time'])
                for imetric in range(self.metricNum):
                    df[imetric] = [math.sin(2 * math.pi / T[itype][imetric] * x + phase[itype][imetric]) + 
                        np.random.normal(loc=0, scale=self.fakeNoiseScale) for x in range(fakeLength)]
#                    df[imetric] = [math.sin(2 * math.pi / T[itype][imetric] * x + np.random.uniform(low=-math.pi, high=math.pi)) + 
#                        np.random.normal(loc=0, scale=self.fakeNoiseScale) for x in range(fakeLength)]
                if self.downSample > 1:
                    dfAvg = df.rolling(window=self.downSample).mean()
                    df = dfAvg[(self.downSample-1)::self.downSample]
                self.rawData[itype][itrace] = df
                self.length[itype][itrace] = len(df)
                self.start[itype][itrace] = cumStart
                cumStart += len(df)

    def genRandVec(self):
        return 2 * np.random.randint(0, 2, self.dimension) - 1
    
    def genMetricVecs(self):
        self.metricVecs = {}
        for metric in self.metrics:
            oneVec = self.genRandVec()
            self.metricVecs[metric] = oneVec
        self.metricMatrix = np.stack([self.metricVecs[imetric] for imetric in self.metrics])
        for imetric in self.metrics:
            for jmetric in self.metrics:
                if imetric != jmetric:
#                    print(self.cos(self.metricVecs[imetric], self.metricVecs[jmetric]))
                    if self.cos(self.metricVecs[imetric], self.metricVecs[jmetric]) > self.closeCriterion:
                        print('===Too similar HD vectors for metrics found.===')
        print('Orthogonality of HD vectors for metrics are checked.')
        
    def normalize(self):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        self.normalized = {}
#        scaler = StandardScaler()
        scaler = MinMaxScaler()
        dfs = []
        for itype in self.types:
            self.normalized[itype] = {}
            for ifile in self.rawData[itype].keys():
                dfs.append(self.rawData[itype][ifile])
#        print('start concatenation..')
        allData = pd.concat(dfs, ignore_index=True)
        allNormalized = allData[['#Time']].copy()
        print('start normalization..')
        if not os.path.isfile(self.metricFileName):
            with open(self.metricFileName, 'w') as metricFile:
                for imetric, metric in enumerate(self.metrics):
                    if allData[metric].min() != allData[metric].max():
                        metricFile.write('%s\n' % metric)
            print('metricFile generated.')
        transformArray = scaler.fit_transform(allData[self.metrics])
        allNormalized[self.metrics] = pd.DataFrame(transformArray, index=allNormalized.index)
#        plt.plot(allNormalized.iloc[:100][self.metrics])
        for itype in self.types:
            for ifile in self.rawData[itype].keys():
                self.normalized[itype][ifile] = allNormalized.iloc[self.start[itype][ifile]:(self.start[itype][ifile]+self.length[itype][ifile])]
    
    def slidingWindow(self):
        self.windows = {}
        for itype in self.types:
            self.windows[itype] = {}
            for ifile in self.rawData[itype].keys():
                self.windows[itype][ifile] = []
                for istart in range(self.length[itype][ifile]-self.windowSize+1):
                    thisWindow = self.normalized[itype][ifile].iloc[istart:(istart+self.windowSize)]
                    self.windows[itype][ifile].append(thisWindow)
#        print('sliding windows generated.')
                    
    def baseline(self):
        '''
        The baseline which use nearest neighbor (in terms of matrix Frobenius norm) to predict anomaly.
        The index may should be updated to work.
        '''
        from sklearn.metrics import f1_score, accuracy_score
        self.represent = {}
        for itype in self.types:
            self.represent[itype] = []
            for i in [0]:
                for iwindow in range(0, len(self.windows[itype][i]), self.windowSize):# only use periodic windows.
                    self.represent[itype].append(self.windows[itype][i][iwindow][self.metrics])
        self.truth, self.predict = [], []
        testLength = 0
        start = timeit.default_timer()
        print('start predicting..')
        for itype in self.types:
            print(itype)
            for i in [1]:
                for iwindow in range(len(self.windows[itype][i])):
                    testLength += len(self.windows[itype][i])
                    self.truth.append(itype)
                    distance = {}
                    for testType in self.types:
                        thisTypeDistances = []
                        for itrain in self.represent[testType]:
                            thisTypeDistances.append(np.linalg.norm(np.subtract(self.windows[itype][i][iwindow][self.metrics], itrain), 'fro'))
                        distance[testType] = min(thisTypeDistances)
                    thisPredict = min(distance, key=distance.get)
                    self.predict.append(thisPredict)
        stop = timeit.default_timer()
        print('testing time per sliding window: %.3f us' % ((stop - start) * 1000000 / testLength))
        f1 = f1_score(self.truth, self.predict, average='weighted')
        accuracy = accuracy_score(self.truth, self.predict)
        print('f1: %.3f, accuracy: %.3f' % (f1, accuracy))
    
    def encoding(self):
        import pickle
        print('start encoding timeseries into HD vectors..')
        self.encoded = {}
        encodeLen = 0
        start = timeit.default_timer()
        for itype in self.types:
            print(itype, end='')
            self.encoded[itype] = {}
            for i, ifile in enumerate(self.rawData[itype].keys()):
                for k in range(1, 20):
                    if i == int(len(self.rawData[itype]) * k/20):
                        print('#', end='')
                        break
                encodeLen += len(self.windows[itype][ifile])
                self.encoded[itype][ifile] = []
                thisTraceValues = self.normalized[itype][ifile][self.metrics].values
                matmul = np.matmul(thisTraceValues, self.metricMatrix)# each line is an unfiltered HD vector for a timestamp.
                matmul[matmul >=0] = 1
                matmul[matmul < 0] = -1
                for iwindow in range( len(self.windows[itype][ifile]) ):
                    windowHD = matmul[iwindow+self.windowSize-1, :]
                    for ivec in range(2, self.windowSize+1):
                        if self.sequenceMode == 'shift':
                            permutated = np.roll(matmul[iwindow+self.windowSize-ivec, :], shift=ivec-1)
                        elif self.sequenceMode == 'permutate':
                            permutated = matmul[iwindow+self.windowSize-ivec]
                            for ipermut in range(ivec-1):
                                permutated = np.random.RandomState(seed=self.permutSeed).permutation(permutated)
                        windowHD = self.multiply(permutated, windowHD)
                    self.encoded[itype][ifile].append(windowHD)
            print()
                    
        stop = timeit.default_timer()
        print('encoding time per sliding window: %.3f ms' % ((stop - start) * 1000 / encodeLen))
        print('encoding finished.')
        if self.storeEncoding:
            print('Storing encoding..')
            with open('%s/encoded.obj' % self.dataFolder, 'bw') as encodedDump:
                pickle.dump(self.encoded, encodedDump)

    def genHDVec(self, dfTimepoint, metricVecs):
        HDVec = np.zeros(self.dimension)
        for metric in self.metrics:
            HDVec += dfTimepoint[metric] * metricVecs[metric]
        HDVec[HDVec >= 0] = 1
        HDVec[HDVec < 0] = -1
        return HDVec

    def multiply(self, a, b):
        return np.multiply(a, b)
        
    def cos(self, a, b):
        if a.shape != b.shape:
            print('===Dimensions should be the same for calculating cosine distance.===')
        else:
            return np.dot(a, b) / self.dimension
        
    def add(self, a, b):
        if a.shape != b.shape:
            print('===Dimensions should be the same for addition.===')
        else:
            idxNotEqual = np.where(np.not_equal(a, b))
            c = np.copy(a)
            c[idxNotEqual] = np.random.choice([-1, 1], len(idxNotEqual))
        return c
        
    def trainTest(self):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.cluster import DBSCAN
        print('start training..')
        print('train method: %s' % self.trainMethod)
        self.combIdx = []
        # self.combIdx = [ ('dcopy', ('59e5059657f3f44ead641ab5', 0)),
        #                 ...
        #                 ]
        # trainTestSets = [ ([1,2,3,4], [0]), ...
        #                  ] (indices for self.combIdx)
        labels = []
        for itype in self.types:
            for ifile in self.encoded[itype].keys():
                self.combIdx.append((itype, ifile))
                labels.append(itype)
        if self.metadata == None and self.anomalyTrain == 'all':# This is regular cross-validation. Not one-shot version.
            skf = StratifiedKFold(n_splits=self.numFold)
            trainTestSets = skf.split(self.combIdx, labels)
        elif self.metadata != None and self.anomalyTrain == 'all': # use metadata.
            trainTestSets = []
            for ifold in range(5):
                trainSet, testSet = [], []
                for idx, row in self.metafile.iterrows():
                    thisFold = row['fold_0']
                    thisRunID = row['runID']
                    thisRunApp = row['app']
                    for nodeNum in range(4):
                        if nodeNum == 1:
                            thisType = row['anomaly']
                        else:
                            thisType = 'none'
                        thisIdx = self.combIdx.index((thisType, (thisRunID, nodeNum, thisRunApp)))
                        if thisFold == ifold:# here is adapted to one-shot learning.
                            trainSet.append(thisIdx)
                        else:
                            testSet.append(thisIdx)
                trainTestSets.append((trainSet, testSet))
                print('trainSetLen, testSetLen: %d, %d' % (len(trainSet), len(testSet)))
        else:# use only app from self.anomalyTrain.
            print('Train only with selected app.')
            trainTestSets = []
            trainSet = []
            testSet = []
            for iicombIdx, icombIdx in enumerate(self.combIdx):
                if icombIdx[1][2] == self.anomalyTrain:
                    trainSet.append(iicombIdx)
                else:
                    testSet.append(iicombIdx)
            trainTestSets.append((trainSet, testSet))
        self.truth, self.predict = [], []
        fold = 0
        for trainIdx, self.testIdx in trainTestSets:
            if self.invertTrainTest:# for training-set-reduced case.
                temp = trainIdx
                trainIdx = self.testIdx
                self.testIdx = temp
            print('fold: %d..' % fold)
            selectedTrainIdx = trainIdx
            print('training..', end='')
            self.represent = {}
            for itype in self.types:
                if self.trainMethod in ['HDadd','addFilter']:
                    self.represent[itype] = []
                elif self.trainMethod in ['closest','HDaddSimilar','DBSCAN']:
                    self.represent[itype] = []
            sampleNum = 0
            lenTrain = len(selectedTrainIdx)
            for iitrainIdx, itrainIdx in enumerate(selectedTrainIdx):
                
                for k in range(1, 20):
                    if iitrainIdx == int(lenTrain * k/20):
                        print('#', end='')
                        break
                
                itype, ifile = self.combIdx[itrainIdx]
                if self.trainSliding:
                    iwindowList = range(0, len(self.encoded[itype][ifile]))
                else:
                    iwindowList = range(0, len(self.encoded[itype][ifile]), self.windowSize)
                for iiwindow, iwindow in enumerate(iwindowList):
                    if self.trainMethod == 'HDadd':
                        if len(self.represent[itype]) != self.dimension:# for the first one.
                            self.represent[itype] = self.encoded[itype][ifile][iwindow]
                            break
                        if self.cos(self.represent[itype], self.encoded[itype][ifile][iwindow]) < 0.05:# only add unsimilar vectors.
                            self.represent[itype] = self.add(self.represent[itype], self.encoded[itype][ifile][iwindow])
                    elif self.trainMethod in ['closest','DBSCAN']:
                        self.represent[itype].append(self.encoded[itype][ifile][iwindow])
                    elif self.trainMethod == 'addFilter':
                        if len(self.represent[itype]) != self.dimension:# for the first one.
                            self.represent[itype] = self.encoded[itype][ifile][iwindow]
                            break
                        self.represent[itype] = np.add(self.represent[itype], self.encoded[itype][ifile][iwindow])
                    elif self.trainMethod == 'HDaddSimilar':
                        sampleNum += 1
                        noSimilar = True
                        for ivector, vector in enumerate(self.represent[itype]):
                            if self.cos(vector, self.encoded[itype][ifile][iwindow]) < self.simVecThres:
                                noSimilar = False
                                self.represent[itype][ivector] = self.add(vector, self.encoded[itype][ifile][iwindow])
                                break
                                #self.represent[itype] = self.add(self.represent[itype], self.encoded[itype][ifile][iwindow])
                        if noSimilar:
                            self.represent[itype].append(self.encoded[itype][ifile][iwindow])
                if self.trainMethod == 'addFilter':
                    self.represent[itype][ self.represent[itype] >= 0 ] = 1
                    self.represent[itype][ self.represent[itype] < 0 ] = -1
            if self.trainMethod == 'DBSCAN':
                print('doing DBSCAN..')
                for itype in self.types:
                    clustering = DBSCAN(eps=self.dimension/100, metric='manhattan', n_jobs=-1).fit(self.represent[itype])
                    DBSCANclass = len(set(clustering.labels_))-1
                    print('DBSCAN classes (excluding -1): %d' % (DBSCANclass))
                    print(clustering.labels_)
                    for ilabel in range(DBSCANclass):
                        print(len(self.represent[itype][ clustering.labels_ == ilabel ]))
                    # not finished.
            if self.trainMethod == 'HDaddSimilar':
                print('At threshold %.3f, collected representative vectors ratio: %.3f' % 
                        (self.simVecThres, sum([len(self.represent[x]) for x in self.types])/sampleNum))
                    
            testLength = 0
            start = timeit.default_timer()
            lenTest = len(self.testIdx)
            print('testing..', end='')
            for iitestIdx, itestIdx in enumerate(self.testIdx):
                
                for k in range(1, 20):
                    if iitestIdx == int(lenTest * k/20):
                        print('#', end='')
                        break
                
                itype, ifile = self.combIdx[itestIdx]
                for iwindow in range(len(self.encoded[itype][ifile])):
                    testLength += len(self.encoded[itype][ifile])
                    self.truth.append(itype)
                    product = {}
                    for testType in self.types:
                        if self.trainMethod in ['HDadd', 'addFilter']:
                            product[testType] = np.dot(self.encoded[itype][ifile][iwindow], self.represent[testType])
                        elif self.trainMethod in ['closest','HDaddSimilar','DBSCAN']:
                            thisTypeProducts = []
                            for itrain in self.represent[testType]:
                                thisTypeProducts.append(np.dot(self.encoded[itype][ifile][iwindow], itrain))
                                if self.cos(self.encoded[itype][ifile][iwindow], itrain) > self.closeCriterion:
                                    pass
                                    #print('Close vector detected: %s, %s' % (itype, testType))
                            product[testType] = max(thisTypeProducts)
                    thisPredict = max(product, key=product.get)
                    self.predict.append(thisPredict)
    
            stop = timeit.default_timer()
            print('f1 for this fold: %.3f' % (f1_score(self.truth, self.predict, average='weighted')))
            correct, incorrect = 0, 0
            for ijtruth, itruth in enumerate(self.truth):
                if itruth == 'linkclog':
                    if self.predict[ijtruth] == 'linkclog':
                        correct += 1
                    else:
                        incorrect += 1
            print('linkclog accuracy this fold: %.3f' % (correct/(correct + incorrect)))
            print('testing time per sliding window: %.3f us' % ((stop - start) * 1000000 / testLength))
            fold += 1
            if self.onlyOneFold:
                print('Only used the first fold.')
                break
            
        f1 = f1_score(self.truth, self.predict, average='weighted')
        self.f1 = f1
        accuracy = accuracy_score(self.truth, self.predict)
        if self.outputEvery:
            results = pd.DataFrame(columns=['truth','predict'])
            results['truth'] = self.truth
            results['predict'] = self.predict
            if self.trainSliding:
                results.to_csv('/projectnb/peaclab-mon/yijia/HDcomputing/results_window%d_downsample%d_trainWith%s_dim%d_seed%d_trainSliding.csv' % (self.windowSize, 
                                                                self.downSample, self.trainMethod, self.dimension, self.seed))
            else:
                results.to_csv('/projectnb/peaclab-mon/yijia/HDcomputing/results_window%d_downsample%d_trainWith%s_dim%d_seed%d_trainPeriodic.csv' % (self.windowSize, 
                                                                self.downSample, self.trainMethod, self.dimension, self.seed))
        print('f1: %.3f, accuracy: %.3f' % (f1, accuracy))
        if self.fout is not None:
            with open(self.fout, 'a') as f:
                f.write('%d,%d,%d,%d,%s,%d,%.3f\n' % (self.metricNum, self.dimension, self.windowSize, self.downSample, 
                                                      self.trainMethod, self.seed, f1))
#                f.write('%d,%.2f,%d,%d,%d,%s,%d,%.3f\n' % (self.metricNum, self.fakeNoiseScale, self.dimension, self.windowSize, self.downSample, 
#                                                      self.trainMethod, self.seed, f1))

    def readEncoded(self):
        import pickle
        readFile = open('%s/encoded.obj' % self.dataFolder, 'rb')
        self.encoded = pickle.load(readFile)

    def checkCorrelation(self):
        '''
        The close relation percentage should be high if the encoding is useful.
        '''
        length = len(self.encoded[self.types[0]][0])
        close = 0
        for i in range(length):
            for j in range(length):
                if i != j:
                    if self.cos(self.encoded[self.types[0]][0][i], self.encoded[self.types[0]][0][j]) > self.closeCriterion:
                        close += 1
        print('close relation percentage: %.3f' % ( close / (length * (length - 1)) ))

    def confusionMatrix(self, suffix=''):
        from sklearn.metrics import confusion_matrix
#        if self.selectApp != 'all':
#            aboutApp = '-'.join(self.selectApp)
#        else:
#            aboutApp = 'allApp'
        aboutApp = self.anomalyTrain
        if self.trainSliding:
            suffix = '_window%d_downsample%d_trainWith%s_dim%d_seed%d_trainSliding%s' % (self.windowSize, self.downSample, self.trainMethod, 
                                                                      self.dimension, self.seed, aboutApp)
        else:
            suffix = '_window%d_downsample%d_trainWith%s_dim%d_seed%d_%s_%s' % (self.windowSize, self.downSample, self.trainMethod, 
                                                                      self.dimension, self.seed, aboutApp, 'invert' if self.invertTrainTest else '')
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(self.truth, self.predict, labels=self.types)
        cnf_matrix = np.fliplr(cnf_matrix)
        np.set_printoptions(precision=2)
    
        # Plot normalized confusion matrix
        plt.figure(figsize=(10,7))
        self.plot_confusion_matrix(cnf_matrix, classes=self.types, normalize=True, suffix=suffix)    

    def plot_confusion_matrix(self, cm, classes,
                              normalize=True,
                              suffix='',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import itertools
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        fs = 20
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('F1-score: %.3f' % self.f1, fontsize=fs)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        rev = classes.copy()
        rev.reverse()
        plt.xticks(tick_marks, rev, rotation=45, fontsize=fs)
        plt.yticks(tick_marks, classes, fontsize=fs)
    
        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), fontsize=fs-4,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        y = plt.ylabel('True label', fontsize=fs)
        x = plt.xlabel('Predicted label', fontsize=fs)
        if self.env == 'dell':
            plt.savefig('C:/Programming/monitoring/HDcomputing/matrix%s.png' % (suffix), bbox_extra_artists=(y,x,), bbox_inches='tight')
        if self.env == 'scc':
            plt.savefig('/projectnb/peaclab-mon/yijia/HDcomputing/matrix%s.png' % (suffix), bbox_extra_artists=(y,x,), bbox_inches='tight')
    
    def draw(self):
        fs = 25
        plt.rc('xtick', labelsize=fs)
        plt.rc('ytick', labelsize=fs)
        for itype in self.types:
            for ifile in range(1):
                fig, ax = plt.subplots(figsize=(16,4))
                ax.plot(self.normalized[itype][ifile][self.metrics].values)
                ax.set_xlabel('Time', fontsize=fs)
                ax.set_ylabel('Values', fontsize=fs)
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 1)
                ax.text(0.5, 0.5, '%s-%s' % (self.selectApp, itype), transform=ax.transAxes, fontsize=fs)
                plt.tight_layout()
                plt.savefig('C:/Programming/monitoring/HDcomputing/draw-%s-%s-%d.png' % (itype, self.selectApp, ifile))
                plt.close()
        
    def readResult(self):
        df = pd.read_csv(self.resultFile)
        self.truth = df['truth']
        self.predict = df['predict']
        
    def diagnose(self):
        dfs = []
        for itype in self.types:
            for ifile in range(len(self.normalized[itype])):# choose a random timestamp (downsampled) from each run and paste them together
                randrow = np.random.randint(0, len(self.normalized[itype][ifile])-1)
                oneRow = self.normalized[itype][ifile].iloc[[randrow]][self.metrics].copy()
                oneRow['type'] = itype
                dfs.append(oneRow)
        concatenated = pd.concat(dfs, ignore_index=True)
        concatenated[['type']+self.metrics].to_csv('C:/Programming/monitoring/HDcomputing/diagnose_downsample%d_%s.csv'
                                                    % (self.downSample, self.selectApp), index=False)
        
#        for itype in self.types:
#            onefile = self.normalized[itype][0][self.metrics].copy()
#            onefile['type'] = [itype] * len(onefile)
#            dfs.append(onefile)
#        concatenated = pd.concat(dfs, ignore_index=True)
#        concatenated[['type']+self.metrics].to_csv('C:/Programming/monitoring/HDcomputing/diagnose_singleRun_downsample%d_%s.csv' 
#                                                    % (self.downSample, self.selectApp), index=False)

def generateFold():
    apps = ['kripke','lu','mg']
    inputs = ['X']
    intensities = [100]
    classes = ['none','dcopy','leak','linkclog']
    outputName = 'C:/Programming/monitoring/HD/folds_threeApps.csv'
    df = pd.DataFrame(columns=['runID','folds'])
    for app in apps:
        for thisInput in inputs:
            for intensity in intensities:
                for thisClass in classes:
                    path = 'C:/Programming/monitoring/data/%s_%s/%s_%d' % (app, thisInput, thisClass, intensity)
                    files = getfiles(path)
                    thisFold = 0
                    for file in files:
                        if file[-5] == '0':
                            thisID = file.split('\\')[-1].split('_')[0]
                            df.loc[len(df)] = [thisID, thisFold]
                            thisFold += 1
                            if thisFold == 5:
                                thisFold = 0
    df.to_csv(outputName, index=False)
    metadata10 = pd.read_csv('C:/Programming/monitoring/run_metadata_10.csv')
    for idx, row in metadata10.iterrows():
        runID = row['runID']
        print(runID)
        thisFold = df[df['runID']==runID].iloc[0]['folds']# don't work. Because some none_20/50/100 are different from the scc version.
        metadata10.at[idx, 'fold_0'] = thisFold
    metadata10.to_csv('C:/Programming/monitoring/run_metadata_10_updated.csv', index=False)
    
def updateFold():
    metadata10 = pd.read_csv('C:/Programming/monitoring/run_metadata_10.csv')
    thisFold = 0
    for idx, row in metadata10.iterrows():
        metadata10.at[idx, 'fold_0'] = thisFold
        thisFold += 1
        if thisFold == 5:
            thisFold = 0
    metadata10.to_csv('C:/Programming/monitoring/run_metadata_10_updated.csv', index=False)
    
def getfiles(path):
    # get all the files with full path.
    fileList = []
    for root, directories, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            fileList.append(filepath)
    return fileList

def readResults():
    from sklearn.metrics import f1_score
    files = getfiles('C:/Programming/monitoring/HDcomputing/results_lu_4class')
    collected = pd.DataFrame(columns=['Window Size', 'Downsample Rate', 'Train Method', 'Dimension', 'Seed', 'F1-score'])
    for f in files:
        if f[-3:] == 'csv':
            fend = f.split('\\')[-1]
            print(fend)
            fsplit = fend.split('_')
            window = int(fsplit[1][6:])
            downsample = int(fsplit[2][10:])
            trainWith = fsplit[3][9:]
            dim = int(fsplit[4][3:])
            seed = int(fsplit[5][4:])
            df = pd.read_csv(f)
            f1 = f1_score(df['truth'], df['predict'], average='weighted')
            collected.loc[len(collected)] = [window, downsample, trainWith, dim, seed, f1]
    collected.to_csv('C:/Programming/monitoring/HDcomputing/results_lu_4class/collected.csv', index=None)
#    collected = pd.read_csv('C:/Programming/monitoring/HDcomputing/results_lu_4class/collected.csv')
    for itrain in ['closest','HDadd','addFilter']:
        thisTrain = collected[collected['Train Method']==itrain]
        pivoted = thisTrain.pivot(index='Window Size', columns='Downsample Rate', values='F1-score')
        pivoted.to_csv('C:/Programming/monitoring/HDcomputing/results_lu_4class/pivoted_%s.csv' % itrain)

def reshape(csv):
    df = pd.read_csv(csv)
#    outdf = df.pivot(index='downSample', columns='windowSize', values='f1')
#    outdf.to_csv('C:/Programming/monitoring/data/bt_X/pivotGrid.csv')
    outdf = df.pivot(index='windowSize', columns='downSample', values='f1')
    outdf.to_csv('C:/Programming/monitoring/data/bt_X/work_search_allbt_pivot.csv')

def windowSize():
    outFile = 'C:/Programming/monitoring/data/bt_X/check_windowSize_100k_512metric.csv'
    with open(outFile, 'w') as f:
        f.write('metricNum,dimension,windowSize,downSample,trainMethod,seed,f1\n')
    for ws in range(2,21):
        hd = HD(mode='check', outFile=outFile, windowSize=ws, dimension=100000, fakeMetricNum=512)
        hd.genMetricVecs()
        hd.normalize()
        hd.slidingWindow()
        hd.encoding()
        hd.checkCorrelation()
        hd.train()
        hd.test()

def dimension():
    outFile = 'C:/Programming/monitoring/data/bt_X/check_dimension.csv'
    with open(outFile, 'w') as f:
        f.write('metricNum,dimension,windowSize,downSample,trainMethod,seed,f1\n')
    for dim in [10**x for x in range(2, 7)]:
        hd = HD(mode='check', outFile=outFile, dimension=dim)
        hd.genMetricVecs()
        hd.normalize()
        hd.slidingWindow()
        hd.encoding()
        hd.checkCorrelation()
        hd.train()
        hd.test()

def downSample():
    outFile = 'C:/Programming/monitoring/data/bt_X/check_downSample.csv'
    with open(outFile, 'w') as f:
        f.write('metricNum,dimension,windowSize,downSample,trainMethod,seed,f1\n')
    for ds in range(2, 31):
        hd = HD(mode='check', outFile=outFile, downSample=ds, seed=1)
        hd.genMetricVecs()
        hd.normalize()
        hd.slidingWindow()
        hd.encoding()
        hd.checkCorrelation()
        hd.train()
        hd.test()

def seed():
    outFile = 'C:/Programming/monitoring/data/bt_X/check_seed.csv'
    with open(outFile, 'w') as f:
        f.write('metricNum,dimension,windowSize,downSample,trainMethod,seed,f1\n')
    for seed in range(100):
        hd = HD(mode='check', outFile=outFile, seed=seed)
        hd.genMetricVecs()
        hd.normalize()
        hd.slidingWindow()
        hd.encoding()
        hd.checkCorrelation()
        hd.train()
        hd.test()

def metricNum():
    outFile = 'C:/Programming/monitoring/data/bt_X/check_metricNum.csv'
    with open(outFile, 'w') as f:
        f.write('metricNum,dimension,windowSize,downSample,trainMethod,seed,f1\n')
    for metricNum in [2**x for x in range(11)]:
        print('===metricNum %d===' % metricNum)
        hd = HD(mode='check', outFile=outFile, fakeMetricNum=metricNum)
        hd.genMetricVecs()
        hd.normalize()
        hd.slidingWindow()
        hd.encoding()
        hd.checkCorrelation()
        hd.train()
        hd.test()

def noise():
    outFile = 'C:/Programming/monitoring/data/bt_X/check_noise_100k_256metric.csv'
    with open(outFile, 'w') as f:
        f.write('metricNum,noise,dimension,windowSize,downSample,trainMethod,seed,f1\n')
    for noise in [x/100 for x in range(5, 71, 5)]:
        print('===noise %.2f===' % noise)
        hd = HD(mode='check', outFile=outFile, fakeNoiseScale=noise, dimension=100000, fakeMetricNum=256)
        hd.genMetricVecs()
        hd.normalize()
        hd.slidingWindow()
        hd.encoding()
        hd.checkCorrelation()
        hd.train()
        hd.test()

def apps():
#    for app in ['bt','cg','CoMD','ft','kripke','lu','mg','miniAMR','miniGhost','miniMD','sp']:
    for app in ['bt','cg','CoMD','ft','kripke','lu','mg','miniAMR','miniGhost','miniMD','sp']:
        hd = HD(mode='work', windowSize=5, downSample=64, dimension=10000, trainMethod='closest', seed=0, trainSliding=False, selectApp=[app],
                selectIntensity=[20,50,100])
        hd.genMetricVecs()
        hd.normalize()
#        hd.draw()
        hd.slidingWindow()
        hd.encoding()
        hd.trainTest()
        hd.confusionMatrix(suffix='')

def drawFromResult():
    for ws in [5]:
        for ds in [64]:
            for tm in ['closest','addFilter','HDadd']:
                hd = HD(mode='work', windowSize=ws, downSample=ds, dimension=10000, trainMethod=tm, seed=0, noPreparedData=False)
                if os.path.isfile(hd.resultFile):
                    hd.readResult()
                    hd.confusionMatrix(suffix='')

def gridSearch():
    outFile = 'C:/Programming/monitoring/data/bt_X/work_search_allbt.csv'
    with open(outFile, 'w') as f:
        f.write('metricNum,dimension,windowSize,downSample,trainMethod,seed,f1\n')
    for ds in [2**x for x in range(5, 7)]:
        for ws in range(2,11):
            hd = HD(mode='work', outFile=outFile, dimension=10000, windowSize=ws, downSample=ds, trainMethod='addFilter')
            hd.genMetricVecs()
            hd.normalize()
            hd.slidingWindow()
            hd.encoding()
            hd.checkCorrelation()
            hd.train()
            hd.test()

def drawTPDS():
    file = pd.read_hdf('C:/Programming/monitoring/HDcomputing/tpds_trainbtcg_entire.hdf')
    print(file.columns)
    drawMatrix(['dcopy','leak','linkclog','memeater','dial','none'], file['actual_label'], file['RandomForest'],
               'C:/Programming/monitoring/HDcomputing/tpds_trainbtcg_entire.png')

def drawMatrix(classes, truth, predict, output):
    from sklearn.metrics import confusion_matrix
    import itertools
    from sklearn.metrics import f1_score
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(truth, predict, labels=classes)
    cnf_matrix = np.fliplr(cnf_matrix)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10,7))
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    fs = 20
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    f1 = f1_score(truth, predict, average='weighted')
    plt.title('F1-score: %.3f' % f1, fontsize=fs)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    rev = classes.copy()
    rev.reverse()
    plt.xticks(tick_marks, rev, rotation=45, fontsize=fs)
    plt.yticks(tick_marks, classes, fontsize=fs)

    fmt = '.3f'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), fontsize=fs-4,
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    y = plt.ylabel('True label', fontsize=fs)
    x = plt.xlabel('Predicted label', fontsize=fs)
    plt.savefig(output, bbox_extra_artists=(y,x,), bbox_inches='tight')

#================================
# main function starts.
if __name__ == '__main__':
    main()

print('finished in %d seconds.' % (datetime.datetime.now()-now).seconds)

