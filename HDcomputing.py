#!/usr/bin/env python
"""



run by:


### TODO ###
get nonConstantMetrics from all data.
gradually reduce dataset.
improve addFilter method.

### warning ###
1 need to use a fixed random seed.
2 be careful with printing without newline.

"""
#=========================


#=========================
import datetime
now = datetime.datetime.now()

import pandas as pd
import numpy as np
import os.path
import timeit
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class HD:
    def __init__(self, env='dell', mode='work', outFile=None, outputEvery=False, windowSize=5, trainMethod='closest', downSample=1, dimension=10000, seed=0, 
                 fakeMetricNum=8, fakeNoiseScale=0.3, trainSliding=False, withData=True, selectApp='all', selectIntensity=[20,50,100]):
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
        print('window size: %d, dimension: %d, seed: %d, downsampling: %d' % (self.windowSize, self.dimension, self.seed, self.downSample))
        if self.mode == 'work':
            self.marginCut = 60
            self.types = ['dcopy','leak','linkclog','none']#,'dial','memeater'
            if withData:
                self.readData()
        elif self.mode == 'check':
            self.metricNum = fakeMetricNum # larger value makes the result better.
            self.fakeNoiseScale = fakeNoiseScale
            if withData:
                self.genFakeSeries()
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
            self.apps = [self.selectApp]
        for itype in self.types:
            print('reading type %s data..' % itype)
            self.rawData[itype] = []
            # self.rawData:
            # {'none':[DF1, DF2, DF3, ...]
            #  'dcopy': ...
            # }
            self.length[itype], self.start[itype] = {}, {}
                        
            files = []
            for intensity in self.selectIntensity:
                if itype != 'none':
                    for iapp in self.apps:
                        temp = getfiles('%s/%s_X/%s_%d' % (self.dataFolder, iapp, itype, intensity))
                        files += [x for x in temp if int(x[-5])==1]# only anomalous.
                else:
                    for iapp in self.apps:
                        for jtype in self.types:
                            temp = getfiles('%s/%s_X/%s_%d' % (self.dataFolder, iapp, jtype, intensity))
                            if jtype != 'none':
                                temp = [x for x in temp if int(x[-5])!=1]# only healthy.
                            files += temp
            fileIdx = 0
            for ifile in files:
                df = pd.read_csv(ifile).iloc[self.marginCut:-self.marginCut]
                if metricFileExist:
                    df = df[['#Time']+self.metrics]
                if self.downSample > 1:
                    dfAvg = df.rolling(window=self.downSample).mean()
                    df = dfAvg[(self.downSample-1)::self.downSample]
                self.length[itype][fileIdx] = len(df)
                self.start[itype][fileIdx] = cumStart
                cumStart += len(df)
                self.rawData[itype].append(df)
                fileIdx += 1
        if not metricFileExist:
            self.metrics = list(self.rawData['none'][0].columns)
            self.metrics.remove('#Time')
        self.metricNum = len(self.metrics)
        print('Total trace number: %d' % sum([len(self.rawData[itype]) for itype in self.types]))

    def genFakeSeries(self):
        import math
        self.metrics = list(range(self.metricNum))
        self.types = ['a','b','c','d','e','f']
        fakeLength = 200
        T, phase, self.length, self.start = {}, {}, {}, {}
        cumStart = 0
        for idx, itype in enumerate(self.types):
            T[itype] = np.random.uniform(low=1, high=400, size=self.metricNum)
            phase[itype] = np.random.uniform(low=-math.pi, high=math.pi, size=self.metricNum)
        for itype in self.types:
            self.rawData[itype] = []
            self.length[itype] = []
            self.start[itype] = []
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
                self.rawData[itype].append(df)
                self.length[itype].append(len(df))
                self.start[itype].append(cumStart)
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
            for i in range(len(self.rawData[itype])):
                dfs.append(self.rawData[itype][i])
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
            for i in range(len(self.rawData[itype])):
                self.normalized[itype][i] = allNormalized.iloc[self.start[itype][i]:(self.start[itype][i]+self.length[itype][i])]
    
    def slidingWindow(self):
        self.windows = {}
        for itype in self.types:
            self.windows[itype] = {}
            for i in range(len(self.rawData[itype])):
                self.windows[itype][i] = []
                for istart in range(self.length[itype][i]-self.windowSize+1):
                    thisWindow = self.normalized[itype][i].iloc[istart:(istart+self.windowSize)]
                    self.windows[itype][i].append(thisWindow)
#        print('sliding windows generated.')
                    
    def baseline(self):
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
            for i in range(len(self.rawData[itype])):
                for k in range(1, 20):
                    if i == int(len(self.rawData[itype]) * k/20):
                        print('#', end='')
                        break
                encodeLen += len(self.windows[itype][i])
                self.encoded[itype][i] = []
                thisTraceValues = self.normalized[itype][i][self.metrics].values
                matmul = np.matmul(thisTraceValues, self.metricMatrix)# each line is an unfiltered HD vector for a timestamp.
                matmul[matmul >=0] = 1
                matmul[matmul < 0] = -1
                for iwindow in range( len(self.windows[itype][i]) ):
                    windowHD = matmul[iwindow+self.windowSize-1, :]
                    for ivec in range(2, self.windowSize+1):
                        if self.sequenceMode == 'shift':
                            permutated = np.roll(matmul[iwindow+self.windowSize-ivec, :], shift=ivec-1)
                        elif self.sequenceMode == 'permutate':
                            permutated = matmul[iwindow+self.windowSize-ivec]
                            for ipermut in range(ivec-1):
                                permutated = np.random.RandomState(seed=self.permutSeed).permutation(permutated)
                        windowHD = self.multiply(permutated, windowHD)
                    self.encoded[itype][i].append(windowHD)
            print()
                    
        stop = timeit.default_timer()
        print('encoding time per sliding window: %.3f ms' % ((stop - start) * 1000 / encodeLen))
#        print('encoding finished.')
#        with open('%s/bt_X/encoded.obj' % self.dataFolder, 'bw') as encodedDump:
#            pickle.dump(self.encoded, encodedDump)

    def genHDVec(self, dfTimepoint, metricVecs):
        HDVec = np.zeros(self.dimension)
        for metric in self.metrics:
            HDVec += dfTimepoint[metric] * metricVecs[metric]
        HDVec[HDVec >= 0] = 1
        HDVec[HDVec < 0] = -1
        return HDVec

    def multiply(self, a, b):
        return np.multiply(a, b)
        
    def train(self):
        '''
        obsolete train and test without n-fold cross-validation.
        '''
        from sklearn.model_selection import StratifiedKFold
        print('start training..')
        print('train method: %s' % self.trainMethod)
        self.represent = {}
        for itype in self.types:
            if self.trainMethod == 'HDadd':
                self.represent[itype] = self.encoded[itype][0][0]
            elif self.trainMethod == 'closest':
                self.represent[itype] = []
            elif self.trainMethod == 'addFilter':
                self.represent[itype] = self.encoded[itype][0][0]
        
        self.combIdx = []
        labels = []
        for itype in self.types:
            for ifile in range(len(self.rawData[itype])):
                self.combIdx.append((itype, ifile))
                labels.append(itype)
        skf = StratifiedKFold(n_splits=self.numFold)
        for trainIdx, self.testIdx in skf.split(self.combIdx, labels):
            for iitrainIdx, itrainIdx in enumerate(trainIdx):                
                itype, ifile = self.combIdx[itrainIdx]
                for iwindow in range(0, len(self.encoded[itype][ifile]), self.windowSize):# only use periodic windows.
                    if self.trainMethod == 'HDadd':
                        if self.cos(self.represent[itype], self.encoded[itype][ifile][iwindow]) < 0.05:# only add unsimilar vectors.
                            self.represent[itype] = self.add(self.represent[itype], self.encoded[itype][ifile][iwindow])
                    elif self.trainMethod == 'closest':
                        self.represent[itype].append(self.encoded[itype][ifile][iwindow])
                    elif self.trainMethod == 'addFilter':
                        self.represent[itype] = np.add(self.represent[itype], self.encoded[itype][ifile][iwindow])
                if self.trainMethod == 'addFilter':
                    self.represent[itype][ self.represent[itype] >= 0 ] = 1
                    self.represent[itype][ self.represent[itype] < 0 ] = -1
                    
            break # thus only get the 1st of all 5 iterations.

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
        
    def test(self):
        '''
        obsolete train and test without n-fold cross-validation.
        '''
        from sklearn.metrics import f1_score, accuracy_score
        print('start predicting..')
        self.truth, self.predict = [], []
        testLength = 0
        start = timeit.default_timer()
        for iitestIdx, itestIdx in enumerate(self.testIdx):
            itype, ifile = self.combIdx[itestIdx]
            for iwindow in range(len(self.encoded[itype][ifile])):
                testLength += len(self.encoded[itype][ifile])
                self.truth.append(itype)
                product = {}
                for testType in self.types:
                    if self.trainMethod in ['HDadd', 'addFilter']:
                        product[testType] = np.dot(self.encoded[itype][ifile][iwindow], self.represent[testType])
                    elif self.trainMethod == 'closest':
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
        print('testing time per sliding window: %.3f us' % ((stop - start) * 1000000 / testLength))
        f1 = f1_score(self.truth, self.predict, average='weighted')
        accuracy = accuracy_score(self.truth, self.predict)
        print('f1: %.3f, accuracy: %.3f' % (f1, accuracy))
        if self.fout is not None:
            with open(self.fout, 'a') as f:
                f.write('%d,%d,%d,%d,%s,%d,%.3f\n' % (self.metricNum, self.dimension, self.windowSize, self.downSample, 
                                                      self.trainMethod, self.seed, f1))
#                f.write('%d,%.2f,%d,%d,%d,%s,%d,%.3f\n' % (self.metricNum, self.fakeNoiseScale, self.dimension, self.windowSize, self.downSample, 
#                                                      self.trainMethod, self.seed, f1))

    def trainTest(self):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score, accuracy_score
        print('start training..')
        print('train method: %s' % self.trainMethod)
        self.combIdx = []
        labels = []
        for itype in self.types:
            for ifile in range(len(self.rawData[itype])):
                self.combIdx.append((itype, ifile))
                labels.append(itype)
        skf = StratifiedKFold(n_splits=self.numFold)
        self.truth, self.predict = [], []
        fold = 0
        for trainIdx, self.testIdx in skf.split(self.combIdx, labels):
        
            print('fold: %d..' % fold)
            self.represent = {}
            for itype in self.types:
                if self.trainMethod == 'HDadd':
                    self.represent[itype] = self.encoded[itype][0][0]
                elif self.trainMethod == 'closest':
                    self.represent[itype] = []
                elif self.trainMethod == 'addFilter':
                    self.represent[itype] = self.encoded[itype][0][0]
            lenTrain = len(trainIdx)
            print('training..', end='')
            for iitrainIdx, itrainIdx in enumerate(trainIdx):
                
                for k in range(1, 20):
                    if iitrainIdx == int(lenTrain * k/20):
                        print('#', end='')
                        break
                
                itype, ifile = self.combIdx[itrainIdx]
                if self.trainSliding:
                    iwindowList = range(0, len(self.encoded[itype][ifile]))
                else:
                    iwindowList = range(0, len(self.encoded[itype][ifile]), self.windowSize)
                for iwindow in iwindowList:
                    if self.trainMethod == 'HDadd':
                        if self.cos(self.represent[itype], self.encoded[itype][ifile][iwindow]) < 0.05:# only add unsimilar vectors.
                            self.represent[itype] = self.add(self.represent[itype], self.encoded[itype][ifile][iwindow])
                    elif self.trainMethod == 'closest':
                        self.represent[itype].append(self.encoded[itype][ifile][iwindow])
                    elif self.trainMethod == 'addFilter':
                        self.represent[itype] = np.add(self.represent[itype], self.encoded[itype][ifile][iwindow])
                if self.trainMethod == 'addFilter':
                    self.represent[itype][ self.represent[itype] >= 0 ] = 1
                    self.represent[itype][ self.represent[itype] < 0 ] = -1
                    
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
                        elif self.trainMethod == 'closest':
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
            print('testing time per sliding window: %.3f us' % ((stop - start) * 1000000 / testLength))
            fold += 1
            
        f1 = f1_score(self.truth, self.predict, average='weighted')
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
        readFile = open('%s/bt_X/encoded.obj' % self.dataFolder, 'rb')
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
        if self.selectApp != 'all':
            aboutApp = '_' + self.selectApp
        else:
            aboutApp = ''
        if self.trainSliding:
            suffix = '_window%d_downsample%d_trainWith%s_dim%d_seed%d_trainSliding%s' % (self.windowSize, self.downSample, self.trainMethod, 
                                                                      self.dimension, self.seed, aboutApp)
        else:
            suffix = '_window%d_downsample%d_trainWith%s_dim%d_seed%d%s' % (self.windowSize, self.downSample, self.trainMethod, 
                                                                      self.dimension, self.seed, aboutApp)
        classifier = 'HD'
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(self.truth, self.predict, labels=self.types)
        cnf_matrix = np.fliplr(cnf_matrix)
        np.set_printoptions(precision=2)
    
        # Plot normalized confusion matrix
        plt.figure(figsize=(10,7))
        self.plot_confusion_matrix(cnf_matrix, classes=self.types, normalize=True, suffix=suffix, 
                              title='%s confusion matrix' % (classifier))    

    def plot_confusion_matrix(self, cm, classes,
                              normalize=True,
                              suffix='',
                              title='Confusion matrix',
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
        plt.title(title, fontsize=fs)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        rev = classes.copy()
        rev.reverse()
        plt.xticks(tick_marks, rev, rotation=45, fontsize=fs)
        plt.yticks(tick_marks, classes, fontsize=fs)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), fontsize=fs,
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
        
def getfiles(path):
    # get all the files with full path.
    fileList = []
    for root, directories, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            fileList.append(filepath)
    return fileList

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
        hd = HD(mode='work', windowSize=5, downSample=64, dimension=10000, trainMethod='closest', seed=0, trainSliding=False, selectApp=app,
                selectIntensity=[20,50,100])
        hd.genMetricVecs()
        hd.normalize()
#        hd.draw()
        hd.slidingWindow()
        hd.encoding()
        hd.trainTest()
        hd.confusionMatrix(suffix='')

def normal():
    hd = HD(mode='work', windowSize=1, downSample=64, dimension=10000, trainMethod='addFilter', seed=0, trainSliding=False, selectApp='lu',
            selectIntensity=[20,50,100])
    hd.genMetricVecs()
    hd.normalize()
#    hd.diagnose()
#    hd.draw()
    hd.slidingWindow()
#    hd.baseline()
    hd.encoding()
#    hd.checkCorrelation()
    hd.trainTest()
    hd.confusionMatrix(suffix='')

def scc():
    import sys
    hd = HD(env='scc', mode='work', outputEvery=True, trainSliding=False, windowSize=int(sys.argv[1]), downSample=int(sys.argv[2]), 
                dimension=int(sys.argv[4]), trainMethod=sys.argv[3], seed=int(sys.argv[5]), selectApp='lu', selectIntensity=[20,50,100])
    hd.genMetricVecs()
    hd.normalize()
    hd.slidingWindow()
    hd.encoding()
    hd.trainTest()
    hd.confusionMatrix()
    
def drawFromResult():
    for ws in [5]:
        for ds in [64]:
            for tm in ['closest','addFilter','HDadd']:
                hd = HD(mode='work', windowSize=ws, downSample=ds, dimension=10000, trainMethod=tm, seed=0, withData=False)
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

#================================
# main function starts.
#apps()
#normal()
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


print('finished in %d seconds.' % (datetime.datetime.now()-now).seconds)

