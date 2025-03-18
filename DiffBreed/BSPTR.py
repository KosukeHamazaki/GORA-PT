#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:34:24 2023

@author: hamazaki
"""

### Import packages
import os
import sys
import warnings
import re
import copy
import gc
import matplotlib.pyplot as plt
# from multiprocessing import Pool, get_context

from plotly.offline import plot
import plotly.express as px
import plotly.graph_objects as go

import random
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import math
import pyper

import torch
from itertools import combinations
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
# import torch.multiprocessing as mp




### Define functions
def flattenList(nestedList):
    flattened = []
    for item in nestedList:
        if isinstance(item, list):
            flattened.extend(flattenList(item))
        else:
            flattened.append(item)
            
    return flattened


def charSeq(prefix = "", seq = None, suffix = "", seqMin = 0, seqMax = 10):
    if seq is None:
        seq = range(seqMin, seqMax + 1)
    else:
        seqMin = min(seq)
        seqMax = max(seq)
    
    return [f"{prefix}{eachNo:0{len(str(seqMax))}d}{suffix}" for eachNo in seq]


def fixSeed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MyDataSet(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]

        return feature, label


def workerInitFn(workerId):
    np.random.seed(np.random.get_state()[1][0] + workerId)
    

class LinearModel(nn.Module):
    def __init__(self, inputSize, outputSize, lam, alpha):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)
        self.lam = lam
        self.alpha = alpha
        
    def forward(self, x):
        return self.linear(x)


def trainModel(model, optimizer,
               criterion, device, 
               loaderTrain, loaderTest = None):
    ### Train loop
    model.train()  ### Model learning on
    trainBatchLoss = []
    for predictor, target in loaderTrain:
        ### Tranfer to GPU
        predictor, target = predictor.to(device), target.to(device)
        
        ### 1. Rest gradient
        optimizer.zero_grad()
        
        ### 2. Inference
        output = model(predictor)
        
        ### 3. Compute loss function
        loss = criterion(output, target)
                        
        if model.alpha != 1:
            l1 = torch.tensor(0., requires_grad = True)
            for w in model.parameters():
                l1 = l1 + torch.norm(w, 1)
        else:
            l1 = 0
                    
        if model.alpha != 0:
            l2 = torch.tensor(0., requires_grad = True)
            for w in model.parameters():
                l2 = l2 + torch.norm(w, 2)
        else:
            l2 = 0
                
        loss = loss + model.alpha * l1 + (1 - model.alpha) * l2
        
        ### 4. Backward propagation
        loss.backward()
        
        #### 5. Parameter update
        optimizer.step()
        
        ### Get training loss
        trainBatchLoss.append(loss.item())


    ### Test loop
    if loaderTest is not None:
        model.eval()  # Model learning off
        testBatchLoss = []
        with torch.no_grad():  # 勾配を計算なし
            for predictor, target in loaderTest:
                predictor, target = predictor.to(device), target.to(device)
                output = model(predictor)
                loss = criterion(output, target)
                testBatchLoss.append(loss.item())
                
        return model, np.mean(trainBatchLoss), np.mean(testBatchLoss)

    else:
        return model, np.mean(trainBatchLoss), None


def ElasticNetCV(X, y, alpha = 0, nFolds = 5, 
                 nCores = 0, seed = 42,
                 nLambdas = 5, minMaxLambdaRatio = 0.0001, 
                 nEpochs = 60, learningRate = 0.01,
                 device = None,
                 includeIntercept = True):
    if device is None:
        ### Selection of computation resource (CPU/GPU)
        if sys.platform == "darwin":
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif sys.platform == "linux":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ### Set seed
    fixSeed(seed = seed)

    nSamples = X.size(0)

    ### Determine lambda candidates
    meanX = torch.mean(X, dim = 0)
    # sdX = torch.sqrt(torch.sum((X - meanX) ** 2, dim = 0) / nSamples)
    sdX = torch.std(X, dim = 0) * ((nSamples - 1) / nSamples) ** (1 / 2)
    scX = (X - meanX) / sdX
    lambdasMax = []
    for i in range(y.size(1)):
        hadamardProd = scX * y[:, torch.tensor(i).repeat_interleave(X.size(1)), ]
        lambdaMaxEach = hadamardProd.sum(0).abs().max() / nSamples
        lambdasMax.append(lambdaMaxEach)
    
    lambdaMax = max(lambdasMax)
    lambdaMin = lambdaMax * minMaxLambdaRatio

    lams = torch.exp(torch.linspace(start = torch.log(lambdaMin), 
                             end = torch.log(lambdaMax),
                             steps = nLambdas)).round(decimals = 10)
    
    ### Initial settings & fold Nos
    bestModel = None
    bestLam = None
    bestLoss = float('inf')
    
    foldIds = (torch.randperm(nSamples) % nFolds)
    
    for lam in lams:
        ### For each lambda (penalization parameter)
        yPred = torch.zeros(size = y.size(), device = device)
        
        for foldNo in range(nFolds): 
            ### For each fold
            ### Training dataset
            XTrain = X[foldIds != foldNo, :]
            yTrain = y[foldIds != foldNo, :]
            dataTrain = MyDataSet(X = XTrain, y = yTrain)
            
            ### Test dataset
            XTest = X[foldIds == foldNo, :]
            # yTest = y[foldIds == foldNo, :]
            # dataTest = MyDataSet(X = XTest, y = yTest)
            
            ### Create data loader
            miniBatchSize = 16
            loaderTrain = data.DataLoader(
                dataTrain,
                batch_size = miniBatchSize,  # バッチサイズ
                shuffle = True,  # データシャッフル
                num_workers = nCores,  # 高速化
                pin_memory = True,  # 高速化
                worker_init_fn = workerInitFn
                )
            
            # loaderTest = data.DataLoader(
            #     dataTest,
            #     batch_size = miniBatchSize,  # バッチサイズ
            #     shuffle = False,  # データシャッフル
            #     num_workers = nCores,  # 高速化
            #     pin_memory = True,  # 高速化
            #     worker_init_fn = workerInitFn
            #     )

            
            ### Create model
            model = LinearModel(inputSize = XTrain.size(1), 
                                outputSize = yTrain.size(1),
                                lam = lam, alpha = alpha).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr = learningRate)
        
            ### Model training
            # trainLossList = []
            # testLossList = []
            for epoch in range(nEpochs):
                model, trainLoss, testLoss = trainModel(
                    model = model, 
                    optimizer = optimizer, 
                    criterion = criterion, 
                    device = device, 
                    loaderTrain = loaderTrain, 
                    loaderTest = None
                    )
                
                # trainLossList.append(trainLoss)
                # testLossList.append(testLoss)
            
            yPredEach = model(XTest)
            yPred[foldIds == foldNo, :] = yPredEach
        
        lossNow = criterion(yPred, y)
        if lossNow < bestLoss:
            bestLoss = lossNow
            bestLam = lam
            # bestModel = model
    
    
    ### Fitting with best lambda
    ### All dataset
    dataAll = MyDataSet(X = X, y = y)
    loaderAll = data.DataLoader(
        dataAll,
        batch_size = miniBatchSize,  # バッチサイズ
        shuffle = True,  # データシャッフル
        num_workers = nCores,  # 高速化
        pin_memory = True,  # 高速化
        worker_init_fn = workerInitFn
        )
            
            
    ### Create model
    finalModel = LinearModel(inputSize = X.size(1), 
                             outputSize = y.size(1),
                             lam = bestLam, alpha = alpha).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(finalModel.parameters(), lr = learningRate)
        
    ### Model training
    # lossList = []
    for epoch in range(nEpochs):
        model, trainLoss, testLoss = trainModel(
            model = finalModel, 
            optimizer = optimizer, 
            criterion = criterion, 
            device = device, 
            loaderTrain = loaderAll, 
            loaderTest = None
            )
                
        # lossList.append(trainLoss)
    
    W = finalModel.linear.weight.T
    b = finalModel.linear.bias
    
    if includeIntercept:
        coef = torch.cat([b.unsqueeze(0), W], dim = 0)
    else:
        coef = W

    return coef



def computeRecombBetweenMarkers(genoMap, 
                                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")):
    chrNames = genoMap["chr"].to_list()
    genPos = genoMap["pos"].to_list()
    
    chrNamesUniq = sorted(list(set(chrNames)), key = None, reverse = False)
    nChrs = len(chrNamesUniq)
    chrNosUniq = list(range(nChrs))
    
    recombRatesList = []
    for chrNo in chrNosUniq:
        chrName = chrNamesUniq[chrNo]
        
        genPosChr = torch.tensor(
            [genPosNow for chrNameNow, genPosNow in zip(chrNames, genPos)
             if chrNameNow == chrName], device = device
            )
        nMarkersChr = genPosChr.size(0)
        
        # genPosChrDiff = torch.tensor(genPosChr).diff()
        # distVec = torch.pdist(genPosChr.unsqueeze(1))
        distVec = torch.pdist(genPosChr.unsqueeze(1).to(device = torch.device("cpu"))).to(device)
        distMat = torch.zeros(size = (nMarkersChr, nMarkersChr), 
                              device = device)
        nBefore = 0
        for i in range(nMarkersChr - 1):
            distMat[range(i + 1, nMarkersChr), i] = distVec[range(nBefore, nBefore + nMarkersChr - i - 1)]
            nBefore = nBefore + nMarkersChr - i - 1
        
        distMat = distMat + distMat.T
        
        recombRatesTorch = torch.tanh(2 * distMat / 100) / 2
        recombRatesList.append(recombRatesTorch)
        
    return recombRatesList


def computeRecombRatesMat(genoMap, 
                          device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")):
    recombRatesList = computeRecombBetweenMarkers(genoMap,
                                                  device = device)
    recombRatesMat = torch.tensor([0.5] * len(genoMap) * len(genoMap))
    recombRatesMat = recombRatesMat.resize(len(genoMap), len(genoMap))
    chrNames = genoMap["chr"].tolist()
    chrNamesUniq = sorted(list(set(chrNames)), key = None, reverse = False)
    
    chrNo = 0

    for chrNo in range(len(recombRatesList)):
        recombRatesChr = recombRatesList[chrNo]
        indicesChr = torch.nonzero(input = torch.tensor((genoMap.chr == chrNamesUniq[chrNo]).tolist())).squeeze()
        sliceChr = slice(indicesChr[0].item(), indicesChr[len(indicesChr) - 1].item() + 1)
        recombRatesMat[sliceChr, sliceChr] = recombRatesChr
        
    return recombRatesMat


def readSimulatedDataInR(dirPyTorch,
                         nChr = 10):
    fileNameFounderHaplo1 = f"{dirPyTorch}founderHaplo1.csv"
    fileNameFounderHaplo2 = f"{dirPyTorch}founderHaplo2.csv"

    founderHaplo1Pd = pd.read_csv(fileNameFounderHaplo1, index_col = 0)
    founderHaplo1 = torch.tensor(np.array(founderHaplo1Pd))
    founderHaplo2Pd = pd.read_csv(fileNameFounderHaplo2, index_col = 0)
    founderHaplo2 = torch.tensor(np.array(founderHaplo2Pd))

    founderHaplo = torch.cat((founderHaplo1.unsqueeze(2),
                              founderHaplo2.unsqueeze(2)), dim = 2)

    fileNameLociEffectsTrue = f"{dirPyTorch}lociEffectsTrue.csv"
    fileNameGenoMap = f"{dirPyTorch}genoMap.csv"
    fileNameMrkPos = f"{dirPyTorch}mrkPos.csv"

    lociEffectsTruePd = pd.read_csv(fileNameLociEffectsTrue, index_col = 0)
    lociEffectsTrue = torch.tensor(np.array(lociEffectsTruePd), dtype = torch.float32)
    genoMap = pd.read_csv(fileNameGenoMap, index_col = 0)

    def bytesToStr(byteVal):
        return byteVal.decode('utf-8') if isinstance(byteVal, bytes) else byteVal
        
    genoMap = genoMap.set_axis(list(range(genoMap.shape[0])), axis = 0, copy = False)
    genoMap = genoMap.set_axis(['lociNames', 'chr', 'pos', 'rec'], axis = 1, copy = False)
    genoMap.lociNames = genoMap.lociNames.apply(bytesToStr)
    genoMap.chr = genoMap.chr.apply(bytesToStr)
    genoMap.rec = genoMap.rec.astype("float32")
    genoMap.pos = genoMap.pos.astype("float32")

    mrkPosPd = pd.read_csv(fileNameMrkPos, index_col = 0)
    mrkPos = torch.tensor(np.array(mrkPosPd), dtype = torch.int64).squeeze()
    
    
    recombRatesList = []
    for chrNo in range(nChr):
        fileNameRecombRates = f"{dirPyTorch}recombRates{chrNo}.csv"
        recombRates = pd.read_csv(fileNameRecombRates, index_col = 0)
        recombRatesList.append(torch.tensor(np.array(recombRates), 
                                            dtype = torch.float32))
    
    simDataDict = {
        "founderHaplo": founderHaplo.to(torch.float32),
        "genoMap": genoMap,
        "lociEffectsTrue": lociEffectsTrue,
        "recombRatesList": recombRatesList,
        "mrkPos": mrkPos
        }
    
    
    return simDataDict


def simDataWithPypeR(nChr = 5, 
                     lChr = 100, 
                     chrNamePre = "C",
                     nLoci = 100,
                     effPopSize = 100,
                     nMarkers = 80,
                     nTraits = 2,
                     nQTLs = 10,
                     nOverlap = 0,
                     effCor = 0,
                     propDomi = 0,
                     interactionMean = 0,
                     actionTypeEpiSimple = True,
                     qtlOverlap = False,
                     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")):
    ### R setting
    # r = pyper.R(RCMD='/home/hamazaki/bin/R', use_pandas = False)
    r = pyper.R(RCMD='/usr/bin/R', use_pandas = False)
    
    ### Some modification
    lChr = [lChr] * nChr
    nQTLs = [nQTLs] * nTraits
    nOverlap = [nOverlap] * nChr
    propDomi = [propDomi] * nTraits
    interactionMean = [interactionMean] * nTraits
    
    
    ### Python --> R
    r.assign('nChr', nChr)
    r.assign('lChr', lChr)
    r.assign('chrNamePre', chrNamePre)
    r.assign('nLoci', nLoci)
    r.assign('effPopSize', effPopSize)
    r.assign('nMarkers', nMarkers)
    r.assign('nTraits', nTraits)
    r.assign('nQTLs', nQTLs)
    r.assign('nOverlap', nOverlap)
    r.assign('nOverlap', nOverlap)
    r.assign('effCor', effCor)
    r.assign('propDomi', propDomi)
    r.assign('interactionMean', propDomi)
    r.assign('qtlOverlap', qtlOverlap)
    r.assign('actionTypeEpiSimple', actionTypeEpiSimple)
    
    
    
    ### Data simulation in R
    r('require(myBreedSimulatR)')  
    r(f'mySimInfo <- simInfo$new(simName = "Simulation Example", '
      f'simGeno = TRUE, '
      f'simPheno = TRUE, '
      f'saveDataFileName = NULL)') 
    r(f'mySpec <- specie$new(nChr = nChr, lChr = lChr, specName = "Example 1", ploidy = 2, '
      f'mutRate = 10^-8, recombRate = 10^-6, chrNames = .charSeq(chrNamePre, 1:nChr), '
      f'nLoci = nLoci, recombRateOneVal = FALSE, effPopSize = effPopSize, '
      f'simInfo = mySimInfo, verbose = TRUE)') 
    r('myLoci <- lociInfo$new(genoMap = NULL, specie = mySpec)') 
    r(f'myTrait <- traitInfo$new(lociInfo = myLoci, nMarkers = nMarkers, nTraits = nTraits, '
      f'nQTLs = nQTLs, actionTypeEpiSimple = actionTypeEpiSimple, qtlOverlap = qtlOverlap, '
      f'nOverlap = nOverlap, effCor = effCor, propDomi = propDomi, interactionMean = interactionMean)')  
    
    r(f'myLoci$computeRecombBetweenMarkers()')
    r(f'nLoci <- myLoci$nLoci()')
    r(f"print(myLoci)")
    
    ### R --> Python
    founderHaplo = torch.tensor(r.get('myLoci$founderHaplo'))
    genoMap = pd.DataFrame(r.get('myLoci$genoMap'))
    nLociAll = r.get('nLoci')
    traitNames = r.get('myTrait$traitNames')
    if nTraits == 1:
        traitNames = [traitNames]
    qtlEff = r.get('myTrait$qtlEff')
    qtlPos = r.get('myTrait$qtlPos')
    mrkPos = torch.tensor(r.get('myTrait$mrkPos')) - 1
    actionType = r.get('myTrait$actionType')
    recombRatesListRaw = r.get('myLoci$recombBetweenMarkersList')
    
    
    ### Modification of `genoMap`
    def bytesToStr(byteVal):
        return byteVal.decode('utf-8') if isinstance(byteVal, bytes) else byteVal
    
    genoMap = genoMap.set_axis(['lociNames', 'chr', 'pos', 'rec'], axis = 1, copy = False)
    genoMap.lociNames = genoMap.lociNames.apply(bytesToStr)
    genoMap.chr = genoMap.chr.apply(bytesToStr)
    genoMap.rec = genoMap.rec.astype("float32")
    genoMap.pos = genoMap.pos.astype("float32")
    

    
    ### Create object `lociEffectsTrue`
    lociEffectsTrue = torch.zeros((nLociAll, nTraits), device = device)
    for traitNo in range(nTraits):
        traitName = traitNames[traitNo]
        qtlPosNow = torch.tensor(qtlPos[traitName], device = device) - 1
        qtlEffNow = torch.tensor(qtlEff[traitName], dtype = torch.float32, device = device)
        actionTypeNow = torch.tensor(actionType[traitName], device = device)
        lociEffectsNow = lociEffectsTrue[:, traitNo]
        qtlPosNowAdd = qtlPosNow[torch.nonzero(actionTypeNow == 0).squeeze()]
        qtlEffNowAdd = qtlEffNow[torch.nonzero(actionTypeNow == 0).squeeze()]
        lociEffectsNow[qtlPosNowAdd] = qtlEffNowAdd / 2
        lociEffectsTrue[:, traitNo] = lociEffectsNow

    lociEffectsTrue = torch.cat((torch.zeros(1, nTraits, device = device), lociEffectsTrue), dim = 0)
    
    
    ### recombRatesList
    recombRatesList = []
    for recombRates in recombRatesListRaw.values():
        recombRatesList.append(torch.tensor(recombRates, dtype = torch.float32))
        
    simDataDict = {
        "founderHaplo": founderHaplo.to(torch.float32),
        "genoMap": genoMap,
        "lociEffectsTrue": lociEffectsTrue,
        "recombRatesList": recombRatesList,
        "mrkPos": mrkPos
        }
    
    return simDataDict


    
class Haplotype:
    def __init__(self,
                 genoHapInd,
                 genoMap):
        if not isinstance(genoHapInd, torch.Tensor):
            sys.exit("`genoHapInd` must be `torch.Tensor` class.")
        
        self.genoMap = genoMap
        self.genoHapInd = genoHapInd
        
        chrNamesUniq = sorted(list(set(genoMap.chr.tolist())), key = None, reverse = False)
        
        haploValues = {}
        genoMapDict = {}
        for chrName in chrNamesUniq:
            whereChr = [chrNameEach in chrName for chrNameEach in genoMap.chr.tolist()]
            haploValues[chrName] = genoHapInd[:, whereChr]
            genoMapDict[chrName] = genoMap[pd.Series(whereChr)]
        
        self.haploValues = haploValues
        self.genoMapDict = genoMapDict
        self.chrNamesUniq = chrNamesUniq

    def allelDose(self):
        genoHapInd = self.genoHapInd
        genoInd = genoHapInd.sum(0)
        
        return(genoInd)



class Individual:
    def __init__(self, 
                 haplotype,
                 lociEffectsTrue = None,
                 parent1 = None,
                 parent2 = None,
                 includeIntercept = True,
                 genGamMethod = 2,
                 device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
                 name = 'Unnamed'):
        self.haplotype = haplotype
        self.lociEffectsTrue = lociEffectsTrue
        self.parent1 = parent1
        self.parent2 = parent2
        self.name = name
        self.includeIntercept = includeIntercept
        self.genGamMethod = genGamMethod
        self.device = device

    # def generateGametes(self, n = 1, sampledForAutoGrad = None):
    #     haplotype = self.haplotype
        
    #     gametes = []
    #     for _ in range(n):
    #         gamete = []
            
    #         for haploChr, genoMapChr in zip(haplotype.haploValues.values(), haplotype.genoMapDict.values()):
    #             if self.device == torch.device("mps"):
    #                 rec0 = np.array(genoMapChr.rec.values).astype("float32").tolist()
    #             else:
    #                 rec0 = genoMapChr.rec.values.astype("float32").tolist()
    #             rec = torch.tensor(rec0, device = self.device)
    #             samples = torch.rand(len(rec), device = self.device)
    #             crossOverBool = rec - samples >= 0
    #             crossOverInt = crossOverBool.type(torch.int)
    #             selectHaplo = torch.cumsum(crossOverInt, dim = 0) % 2
    #             whichHaplo = torch.randint(2, (1, ))
    #             gamHaplo = haploChr[whichHaplo]
    #             gamHaplo[:, selectHaplo == 1] = haploChr[1 - whichHaplo, selectHaplo == 1]
    #             gamHaplo = gamHaplo.squeeze()
    #             gamete.append(gamHaplo)
           
    #         gamete = torch.cat(gamete)
            
    #         if sampledForAutoGrad is not None:
    #             whichSampled = torch.nonzero(input = sampledForAutoGrad, as_tuple = False).squeeze()
    #             gameteExtend = torch.zeros(size = (gamete.size(0), len(sampledForAutoGrad)), device = self.device)
    #             gameteExtend[:, whichSampled] = gamete
                
    #             gamete = torch.matmul(gameteExtend, sampledForAutoGrad)
            
    #         gametes.append(gamete)
    #     return gametes 

    
    def generateGametes(self, n = 1, sampledForAutoGrad = None):
        haplotype = self.haplotype
        genGamMethod = self.genGamMethod
        
        if genGamMethod >= 3:
            recombRatesList = computeRecombBetweenMarkers(haplotype.genoMap,
                                                          device = self.device)
            
            gamMeanList = []
            gamCovCholList = []
            gamCovInvBlockInvList = []
            gamCovInvBlockInvCholList = []
            lambVecList = []
            for haploChr, chrNo in zip(haplotype.haploValues.values(), 
                                       range(len(haplotype.haploValues.values()))):
                recombRates = recombRatesList[chrNo]
                gamMean = haploChr.mean(0)
                gamMeanList.append(gamMean)
                    
                gamDiff = haploChr[0, :] - haploChr[1, :]
                gamLinKer = torch.mm(gamDiff.unsqueeze(1), gamDiff.unsqueeze(0))
                gamCov = (1 - 2 * recombRates) * gamLinKer / 4
                gamCovPlus = torch.zeros(gamCov.size())
                gamCovPlus.fill_diagonal_(1e-06)
                gamCov = gamCov + gamCovPlus
                    
                if genGamMethod == 3:
                    gamCovChol = torch.linalg.cholesky(gamCov)
                    gamCovCholList.append(gamCovChol)
                elif genGamMethod == 3.5:
                    gamCovInvBlockInv = gamCov[1:, 1:] - torch.mm(gamCov[1:, 0].unsqueeze(1),
                                                                  gamCov[1:, 0].unsqueeze(0)) / gamCov[0, 0]
                    gamCovInvBlockInvChol = torch.linalg.cholesky(gamCovInvBlockInv)
                    lambVec = - torch.mm(gamCovInvBlockInv, gamCov[1:, 0].unsqueeze(1)) / gamCov[0, 0]
                    
                    gamCovInvBlockInvList.append(gamCovInvBlockInv)
                    gamCovInvBlockInvCholList.append(gamCovInvBlockInvChol)
                    lambVecList.append(lambVec)
                    
        else:
            recombRatesList = None
      
        gametes = []
        for _ in range(n):
            gamete = []

            for haploChr, genoMapChr, chrNo in zip(haplotype.haploValues.values(), 
                                                   haplotype.genoMapDict.values(),
                                                   range(len(haplotype.haploValues.values()))):
                
                if genGamMethod <= 2:
                    isCrossOver = torch.rand(1) >= 0.5
                    if isCrossOver:
                        if self.device == torch.device("mps"):
                            rec0 = np.array(genoMapChr.rec.values).astype("float32").tolist()
                        else:
                            rec0 = genoMapChr.rec.values
                
                        rec = torch.tensor(rec0, device = self.device)
                        crossOverProb = rec.clone()
                        crossOverProb[1:len(rec)] = crossOverProb[1:len(rec)] * 2
                        if genGamMethod == 1:
                            gamHaplo = torch.zeros(len(rec))
                            whichHaploNow = 0
                            for m in range(len(rec)):
                                sample = torch.rand(1, device = self.device)
                                crossOverBool = crossOverProb[m] - sample >= 0
                                if crossOverBool:
                                    whichHaploNow = 1 - whichHaploNow
                                else:
                                    whichHaploNow = whichHaploNow
                                gamHaplo[m] = haploChr[whichHaploNow, m]
                                
                        elif genGamMethod == 2:
                            samples = torch.rand(len(crossOverProb), device = self.device)
                            crossOverBool = crossOverProb - samples >= 0
                            crossOverInt = crossOverBool.type(torch.int)
                            selectHaplo = torch.cumsum(crossOverInt, dim = 0) % 2
                            whichHaplo = torch.randint(2, (1, ))
                            gamHaplo = haploChr[whichHaplo, :]
                            gamHaplo[:, selectHaplo == 1] = haploChr[1 - whichHaplo, selectHaplo == 1]
                    else:
                        whichHaplo = torch.randint(2, (1, ))
                        gamHaplo = haploChr[whichHaplo, :]
                    
                    gamHaplo = gamHaplo.squeeze()
                    
                    
                elif genGamMethod >= 3:
                    recombRates = recombRatesList[chrNo]
                    gamMean = gamMeanList[chrNo]
            
                    
                    if genGamMethod == 3:
                        gamCovChol = gamCovCholList[chrNo]
                    
                        samples = torch.normal(mean = torch.tensor(0.), 
                                               std = torch.tensor(1.), 
                                               size = (gamCov.size(0), 1))
                        gamHaplo = gamMean + torch.mm(gamCovChol, samples).squeeze()
                    elif genGamMethod == 3.5:
                        whichHaplo = torch.randint(2, (1, ))
                        gamHaplo = haploChr[whichHaplo, :].squeeze()
                    
                        gamCovInvBlockInv = gamCovInvBlockInvList[chrNo]
                        gamCovInvBlockInvChol = gamCovInvBlockInvCholList[chrNo]
                        lambVec = lambVecList[chrNo]
                        
                        samples = torch.normal(mean = torch.tensor(0.), 
                                               std = torch.tensor(1.), 
                                               size = (gamCov.size(0) - 1, 1))
                    
                        gamMeanUpd = gamMean[1:] - torch.mm(gamCovInvBlockInv, lambVec).squeeze() * (gamHaplo[0] - gamMean[0])
                        gamHaplo[1:] = gamMeanUpd + torch.mm(gamCovInvBlockInvChol, samples).squeeze()
                
                gamete.append(gamHaplo)
                    
            gamete = torch.cat(gamete)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            if sampledForAutoGrad is not None:
                whichSampled = torch.nonzero(input = sampledForAutoGrad, as_tuple = False).squeeze()
                gameteExtend = torch.zeros(size = (gamete.size(0), len(sampledForAutoGrad)), 
                                           device = self.device)
                gameteExtend[:, whichSampled] = gamete
      
                gamete = torch.matmul(gameteExtend, sampledForAutoGrad)
      
            gametes.append(gamete)
                
            
        return gametes
    
    
    def trueGV(self):
        haplotype = self.haplotype
        lociEffectsTrue = self.lociEffectsTrue
        
        if lociEffectsTrue is None:
            trueGV = None
        else:
            geno = haplotype.allelDose().to(device = self.device)
            
            if self.includeIntercept:
                geno = torch.cat([torch.tensor([1.], device = self.device), geno], dim = 0)
            
            if lociEffectsTrue.size(1) == 1: 
                trueGV = torch.reshape(input = torch.sum(geno * torch.reshape(input = lociEffectsTrue, shape = geno.size()).to(device = self.device)),
                                       shape = (1, 1)).to(device = self.device)
            else: 
                trueGV = torch.mm(torch.reshape(input = geno, shape = (1, geno.size(0))).to(device = self.device), lociEffectsTrue.to(device = self.device))

        return(trueGV)


class Population:
    def __init__(self, 
                 indsList,
                 name = None,
                 generation = None,
                 lociEffectsTrue = None,
                 crossInfo = None,
                 herit = None,
                 nRep = None,
                 phenotypicValues = None,
                 includeIntercept = True,
                 device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
                 verbose = True):
        if name is None:
            name = 'Unspecified'
        
        if crossInfo is not None:
            if generation is None:
                generationBefore = crossInfo.parentPopulation.generation
                
                if generationBefore is None:
                    generation = 1
                else:
                    generation = generationBefore + 1

        if herit is None:
            herit = torch.tensor(0.7, device = device)

        if nRep is None:
            nRep = torch.tensor(3, device = device)
        
        if type(indsList) == '__main__.Individual':
            indsList = [indsList]
        
        if lociEffectsTrue is not None:
            nTraits = lociEffectsTrue.size(1)

        indNames = [ind.name for ind in indsList]
        genoMap = indsList[0].haplotype.genoMap

        
        self.indsList = indsList
        self.indNames = indNames
        self.name = name
        self.generation = generation
        self.genoMap = genoMap
        self.lociEffectsTrue = lociEffectsTrue
        self.crossInfo = crossInfo
        self.herit = herit
        self.nRep = nRep
        self.nTraits = nTraits
        self.phenotypicValues = phenotypicValues
        self.genoMat = None
        self.af = None
        self.maf = None
        self.includeIntercept = includeIntercept
        self.device = device
        self.verbose = verbose
        
        self.inputPhenotypicValues()
        self.obtainGenoMat()

        
    def addInd(self, newIndsList):
        if type(newIndsList) == '__main__.Individual':
            newIndsList = [newIndsList]
        
        newIndNames = [ind.name for ind in newIndsList]
        self.indsList = self.indsList + newIndsList
        self.indNames = self.indNames + newIndNames
    
    def inputPhenotypicValues(self, phenotypicValues = None):
        if self.phenotypicValues is None:
            if phenotypicValues is None:
                self.obtainTrueGVMat()
                herit = self.herit
                nRep = self.nRep
                trueGVMat = self.trueGVMat
                
                trueGeneticVar = torch.var(trueGVMat, dim = 0)
                residVars = trueGeneticVar * (1 - herit) / herit * nRep
                residMeans = torch.tensor(0.).repeat_interleave(trueGVMat.size(1)).to(self.device)
                # residVars = torch.tensor([10, 100], device = self.device)
                # residMeans = torch.tensor([0, 0], device = self.device)
                
                resid = torch.concat([
                        torch.normal(mean = residMean, std = torch.sqrt(torch.clone(residVar).detach()), 
                                     size = (trueGVMat.size(0), 1, nRep), device = self.device)
                        for residMean, residVar in zip(residMeans, residVars)
                        ], dim = 1)
                # resid = torch.concat([
                #         torch.normal(mean = residMean, std = torch.sqrt(residVar), 
                #                      size = (trueGVMat.size(0), 1, nRep))
                #         for residMean, residVar in zip(residMeans, residVars)
                #         ], dim = 1)
                trueGVArray = torch.reshape(trueGVMat.repeat_interleave(nRep),
                                            shape = (trueGVMat.size(0), trueGVMat.size(1), nRep))
            
                phenotypicValues = trueGVArray + resid

            self.phenotypicValues = phenotypicValues
    
    def obtainGenoMat(self):
        indsList = self.indsList
        genoMat = torch.stack([ind.haplotype.allelDose() for ind in indsList], dim = 0)
        self.genoMat = genoMat
        
        
    def obtainHaploArray(self):
        indsList = self.indsList
        haploArray = torch.stack([ind.haplotype.genoHapInd.T for ind in indsList], dim = 0)
        self.haploArray = haploArray
    
    
    def obtainAlleleFreq(self):
        if self.af is None:
            if self.genoMat is None:
                self.obtainGenoMat()
                
            genoMat = self.genoMat
            af = torch.mean(genoMat, dim = 0) / 2
            maf = torch.minimum(af, 1 - af)
        
            self.af = af
            self.maf = maf
        
    
    def obtainTrueGVMat(self):
        indsList = self.indsList
        trueGVMat = torch.cat([ind.trueGV() for ind in indsList], dim = 0)
        self.trueGVMat = trueGVMat
        
    def plot(self, traitNo = 0):

        plt.scatter(self.trueGVMat[:, traitNo], 
                    self.phenotypicValues.mean(dim = 2)[:, traitNo])
        plt.show()
        

def createInitPop(founderHaplo,
                  genoMap,
                  lociEffectsTrue = None,
                  founderIsInitPop = False,
                  nGenerationRM = 100,
                  nGenerationRSM = 10,
                  nGenerationRM2 = 3,
                  propSelRS = 0.1,
                  indNames = None,
                  popName = None,
                  generation = None,
                  herit = None,
                  nRep = None,
                  phenotypicValues = None,
                  includeIntercept = True,
                  genGamMethod = 2,
                  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
                  verbose = True):
    founderHaplo = founderHaplo.to(device)
    if (len(founderHaplo.size()) == 3) & founderIsInitPop:
        genoHap = founderHaplo
    else:
        nFounderHaplos = founderHaplo.size(0)
        randomSamples = torch.randint(0, nFounderHaplos, (nFounderHaplos, ))
        haploPair = randomSamples.view(-1, 2)
    
        haplo = torch.stack([founderHaplo[haploPair[x, :], :] for x in range(haploPair.size(0))], dim = 2)
        genoHap = haplo.permute(2, 1, 0)
    
    if indNames is None:
        indNames = [f"G0_{indNo:0{len(str(genoHap.size(0)))}d}" for indNo in range(1, genoHap.size(0) + 1)]

    
    indsList = []
    for indNo in range(genoHap.size(0)):
        haploNow = Haplotype(genoMap = genoMap, 
                             genoHapInd = torch.transpose(genoHap[indNo, :, :], 1, 0))
        
        indNow = Individual(haplotype = haploNow, 
                            lociEffectsTrue = lociEffectsTrue,
                            name = indNames[indNo],
                            includeIntercept = includeIntercept,
                            genGamMethod = genGamMethod,
                            device = device)
        indsList.append(indNow)
    
    
    if founderIsInitPop:
        population = Population(indsList = indsList,
                                name = popName,
                                generation = generation,
                                lociEffectsTrue = lociEffectsTrue,
                                crossInfo = None,
                                herit = herit,
                                nRep = nRep,
                                phenotypicValues = phenotypicValues,
                                includeIntercept = includeIntercept,
                                device = device,
                                verbose = verbose)
    else:
        popNow = Population(indsList = indsList,
                             name = popName,
                             generation = generation,
                             lociEffectsTrue = lociEffectsTrue,
                             crossInfo = None,
                             herit = herit,
                             nRep = nRep,
                             phenotypicValues = phenotypicValues,
                             includeIntercept = includeIntercept,
                             device = device,
                             verbose = verbose)
        nGenerationTotal = nGenerationRM + nGenerationRSM + nGenerationRM2
        
        def proceedGen(popNow, proceedType):
            nNextPop = genoHap.size(0)
            
            if proceedType == "randomMate":
                selectionMethod = "nonSelection"
                nSel = nNextPop
                nGenerationProceed = nGenerationRM
            elif proceedType == "randomSelMate":
                selectionMethod = "userSpecific"
                nSel = round(nNextPop * propSelRS)
                nGenerationProceed = nGenerationRSM
            elif proceedType == "randomMate2":
                selectionMethod = "nonSelection"
                nSel = nNextPop
                nGenerationProceed = nGenerationRM2
            
            
            for generationProceedNo in range(nGenerationProceed):
                generation = popNow.generation + 1
                
                if generation == nGenerationTotal:
                    generation = 0
                    indNamesNow = indNames.copy()
                else:
                    indNamesNow = [f"G{generation:0{len(str(nGenerationTotal))}d}_{indNo:0{len(str(nNextPop))}d}" for indNo in range(1, nNextPop + 1)]
            
                if proceedType == "randomSelMate":
                    selCands = list(np.random.choice(popNow.indNames, size = nSel, replace = False))
                else:
                    selCands = None
                    
                crossInfoNow = CrossInfo(
                    parentPopulation = popNow,
                    lociEffectsEst = None,
                    selectionMethod = selectionMethod,
                    multiTraitsEvalMethod = "sum",
                    matingMethod = "randomMate",
                    allocateMethod = "equalAllocation",
                    # weightedAllocationMethod = None,
                    hSel = 1.,
                    h = torch.tensor(0.1),
                    minimumUnitAllocate = 1,
                    clusteringForSel = False,
                    nCluster = 1,
                    nTopCluster = 1,
                    nTopEach = nNextPop,
                    nSel = nSel,
                    userSC = None,
                    nProgenies = None,
                    nNextPop = nNextPop,
                    includeGVP = False,
                    targetGenForGVP = 0,
                    indNames = indNamesNow,
                    seedSimRM = None,
                    seedSimMC = None,
                    requires_grad = False,
                    includeIntercept = includeIntercept,
                    selCands = selCands,
                    device = device,
                    name = f"CrossInfo-{generationProceedNo}"
                    )
                
                
                popNow = Population(indsList = crossInfoNow.makeCrosses(),
                                    name = popName,
                                    generation = generation,
                                    lociEffectsTrue = lociEffectsTrue,
                                    crossInfo = crossInfoNow,
                                    herit = herit,
                                    nRep = nRep,
                                    phenotypicValues = phenotypicValues,
                                    includeIntercept = includeIntercept,
                                    device = device,
                                    verbose = verbose)
            return popNow
        
        if nGenerationRM > 0:
            popNow = proceedGen(popNow, "randomMate")
    
        if nGenerationRSM > 0:
            popNow = proceedGen(popNow, "randomSelMate")
            
        if nGenerationRM2 > 0:
            popNow = proceedGen(popNow, "randomMate2")
        
        population = popNow
            
            
    return population




class CrossInfo:
    def __init__(self, 
                 parentPopulation,
                 lociEffectsEst = None,
                 selectionMethod = "BV",
                 multiTraitsEvalMethod = "sum",
                 matingMethod = "randomMate",
                 allocateMethod = "equalAllocation",
                 weightedAllocationMethod = ["WBV"],
                 hSel = None,
                 h = None,
                 minimumUnitAllocate = 5,
                 clusteringForSel = True,
                 nCluster = 10,
                 nTopCluster = 5,
                 nTopEach = None,
                 nSel = None,
                 userSC = None,
                 nProgenies = None,
                 nNextPop = None,
                 includeGVP = False,
                 targetGenForGVP = 2,
                 indNames = None,
                 seedSimRM = None,
                 seedSimMC = None,
                 selCands = None,
                 requires_grad = False,
                 includeIntercept = True,
                 device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
                 name = 'Unnamed'):
        if parentPopulation.lociEffectsTrue is not None:
            nTraits = parentPopulation.lociEffectsTrue.size(1)
        else:
            nTraits = None
        
        if hSel is None:
            hSel = torch.tensor(1.0, device = device).repeat_interleave(nTraits)
            hSel = hSel / hSel.sum()
        
        if h is None:
            h = torch.tensor(1.0, device = device).repeat_interleave(len(weightedAllocationMethod) * nTraits)
        
        if nSel is None:
            nSel = len(parentPopulation.indNames) // 10

        if nCluster < nTopCluster:
            nTopCluster = nCluster
        
        if nTopEach is None:
            nTopEach = nSel // nTopCluster
        
        if nCluster == 1:
            clusteringForSel = False
        
        if clusteringForSel:
            nCluster = 1
            nTopCluster = 1
            nTopEach = nSel
        
        if nNextPop is None:
            nNextPop = len(parentPopulation.indNames)
        
        if seedSimRM is None:
            seedSimRM = random.sample(population = range(1000000), k = 1)[0]
        
        if seedSimMC is None:
            seedSimMC = random.sample(population = range(1000000), k = 1)[0]
        
        if "userSpecific" not in weightedAllocationMethod:
            nProgenies = None
            
        if indNames is None:
            generationNow = parentPopulation.generation
            if generationNow is None:
                generationNew = 1
            else:
                generationNew = generationNow + 1
                
            indNames = f"G{generationNew}"
        
        if requires_grad:
            if not h.requires_grad:
                h.requires_grad = True
        else:
            h = torch.clone(h).detach()
        
        
        self.parentPopulation = parentPopulation
        self.lociEffectsEst = lociEffectsEst
        self.selectionMethod = selectionMethod
        self.multiTraitsEvalMethod = multiTraitsEvalMethod
        self.matingMethod = matingMethod
        self.allocateMethod = allocateMethod
        self.weightedAllocationMethod = weightedAllocationMethod
        self.name = name
        self.hSel = hSel
        self.h = h
        self.minimumUnitAllocate = minimumUnitAllocate
        self.nTraits = nTraits
        self.nCluster = nCluster
        self.nTopCluster = nTopCluster
        self.nTopEach = nTopEach
        self.clusteringForSel = clusteringForSel
        self.nSel = nSel
        self.nProgenies = nProgenies
        self.nNextPop = nNextPop
        self.indNames = indNames
        self.seedSimRM = seedSimRM
        self.seedSimMC = seedSimMC
        self.requires_grad = requires_grad
        self.includeGVP = includeGVP
        self.targetGenForGVP = targetGenForGVP
        self.includeIntercept = includeIntercept
        self.device = device
        
        self.userSC = userSC
        self.BV = None
        self.WBV = None
        self.GVP = None
        self.SCEachPair = None
        self.genDistVec = None
        self.genDistMat = None
        self.group = None
        self.selCands = selCands
        self.crossTable = None
        self.crossTableWoRA = None
        
        self.indicesChrList = None
        self.d12Mat1List = None
        self.d12Mat2List = None
        self.d13Mat1List = None
        self.d13Mat2List = None
        
        if includeGVP:
            self.computeDMat()
        
    
    def computeGenDist(self):
        # genoMat = parentPopulation.genoMat
        genoMat = self.parentPopulation.genoMat.to(torch.device("cpu"))
        distVec = torch.pdist(genoMat).to(self.device)

        # 距離行列を2次元行列に変換
        # 距離行列は下三角行列であるため、2次元行列に変換する際に注意が必要
        nLine = len(genoMat)
        
        distMat = torch.zeros(size = (nLine, nLine), device = self.device)
        nBefore = 0
        for i in range(nLine - 1):
            distMat[range(i + 1, nLine), i] = distVec[range(nBefore, nBefore + nLine - i - 1)]
            nBefore = nBefore + nLine - i - 1

        distMat = distMat + distMat.T
        
        self.genDistVec = distVec
        self.genDistMat = distMat


    
    def hClustering(self, nCluster = None):
        if self.genDistVec is None:
            self.computeGenDist()
        
        distVec = torch.clone(self.genDistVec).detach().to(torch.device("cpu"))
        
        hClustRes = linkage(distVec, method='ward')

        # クラスタリング結果を取得 (3クラスタに分割)
        if nCluster is None:
            nCluster = self.nCluster
            
        group = torch.tensor(fcluster(hClustRes, nCluster, criterion='maxclust'), device = self.device)
        self.group = group

        
    
    def selectParentCands(self):
        # selectionMethod = 'BV'
        # multiTraitsEvalMethod = "prod"
        # hSel = torch.tensor(1.0, device = self.device).repeat_interleave(nTraits)
        selectionMethod = self.selectionMethod
        parentPopulation = self.parentPopulation
        multiTraitsEvalMethod = self.multiTraitsEvalMethod
        hSel = self.hSel
        clusteringForSel = self.clusteringForSel
        nTopCluster = self.nTopCluster
        nTopEach = self.nTopEach
        nSel = self.nSel
        indNamesParentPop = parentPopulation.indNames
        
        if selectionMethod != "userSpecific":
            if selectionMethod == "nonSelection":
                parentCands = indNamesParentPop
            else:
                if selectionMethod == "BV":
                    if self.BV is None:
                        self.computeBV()
                        
                    SC = self.BV
                    
                elif selectionMethod == "WBV":
                    if self.WBV is None:
                        self.computeWBV()

                    SC = self.WBV
                elif selectionMethod == "userSC":
                    SC = self.userSC
                    
                
                if multiTraitsEvalMethod == "prod":
                        SCPositive = SC - torch.min(SC, dim = 0).values
                        SCSel = torch.prod(SCPositive, dim = 1)
                else:
                    SCMean = SC.mean(dim = 0, keepdim = True)
                    SCSd = SC.std(dim = 0, keepdim = True)
                    
                    SCScaled = (SC - SCMean) / SCSd
                    if torch.any(SCSd == 0):
                        whichSdZero = (SCSd[0] == 0).nonzero(as_tuple = False)
                        SCScaled[:, whichSdZero] = 0
                        
                    SCSel = torch.matmul(SCScaled, hSel)
                    
                if clusteringForSel:
                    if self.group is None:
                        self.hClustering()
                    group = self.group
                    groupUniq = torch.unique(input = group)
                    
                    SCTopsList = []
                    SCTopsMeanList = []
                    parentCandsList = []
                    for groupUniqNo in groupUniq:
                        SCSelGroup = SCSel[group == groupUniqNo]
                        selectedIndices = torch.nonzero(input = group == groupUniqNo).squeeze()
                        if len(selectedIndices) < nTopEach:
                            nTopEachNow = len(selectedIndices)
                        else:
                            nTopEachNow = nTopEach
                        
                        indNamesParentPopGroup = [indNamesParentPop[selNo] for selNo in selectedIndices]
                        sortedResGroup = torch.sort(input = SCSelGroup, descending = True)
                        
                        SCTops = sortedResGroup.values[range(nTopEachNow)]
                        SCTopsMean = SCTops.mean().reshape(1)
                        parentCandsGroup = [indNamesParentPopGroup[sortedResGroup.indices[selNo]] for selNo in range(nTopEachNow)]
                        
                        SCTopsList.append(SCTops)
                        SCTopsMeanList.append(SCTopsMean)
                        parentCandsList.append(parentCandsGroup)
                        
                    SCTopsMeans = torch.cat(SCTopsMeanList)
                    selectedGroups = torch.sort(input = SCTopsMeans, descending = True).indices[range(nTopCluster)]

                    SCTopsSelList = [SCTopsList[selectedGroup] for selectedGroup in selectedGroups]
                    SCTopsSels = torch.cat(SCTopsSelList, dim = 0)
                    parentCandsSelList = [parentCandsList[selectedGroup] for selectedGroup in selectedGroups]
                    parentCandsSel = [item for sublist in parentCandsSelList for item in sublist]
                    
                    if len(SCTopsSels) < nSel:
                        nSelNow = len(SCTopsSels)
                    else:
                        nSelNow = nSel
                    
                    sortedResSCTopSels = torch.sort(input = SCTopsSels, descending = True)
                    parentCands = [parentCandsSel[sortedResSCTopSels.indices[selNo]] for selNo in range(nSelNow)]
                else:
                    sortedRes = torch.sort(input = SCSel, descending = True)
                    parentCands = [indNamesParentPop[sortedRes.indices[selNo]] for selNo in range(nSel)]
            
        self.selCands = parentCands

    
    def designResourceAllocation(self):
        # indNamesParentPop = self.parentPopulation.indNames
        allocateMethod = self.allocateMethod
        nNextPop = self.nNextPop
        # weightedAllocationMethod = self.weightedAllocationMethod
        h = self.h
        minimumUnitAllocate = self.minimumUnitAllocate
        if self.crossTableWoRA is None:
            self.makeCrossTableWoRA()
        crossTableWoRA = self.crossTableWoRA
        
        nPairs = len(crossTableWoRA)
        
        if allocateMethod == "equalAllocation":
            crossTableWoRASorted = crossTableWoRA.sample(frac = 1, random_state = 1)
            nPairsNow = nNextPop // minimumUnitAllocate
            
            if nPairs < nPairsNow:
                nUnitAllocate = nPairsNow // nPairs
                nProgeniesPerPair = minimumUnitAllocate * nUnitAllocate
                nProgenies = torch.tensor(nProgeniesPerPair, device = self.device).repeat_interleave(nPairs)
            else:
                nProgeniesPerPair = minimumUnitAllocate
                nProgenies = torch.tensor(nProgeniesPerPair, device = self.device).repeat_interleave(nPairsNow)
                
            nResids = nNextPop - nProgenies.sum()
            
            while nResids > 0:
                nResidsSurplus = (nResids % minimumUnitAllocate).to(torch.int)
                nResidsQuotient = (nResids // minimumUnitAllocate).to(torch.int)
                
                if nResidsQuotient >= 1:
                    residsPlus = torch.tensor(minimumUnitAllocate, device = self.device).repeat_interleave(nResidsQuotient)
                else:
                    residsPlus = torch.tensor(0, device = self.device).repeat_interleave(nResidsSurplus)
                    
                if nResidsSurplus > 0:
                    residsPlus[0:nResidsSurplus] = residsPlus[0:nResidsSurplus] + 1
                
                if len(nProgenies) >= len(residsPlus): 
                    nProgenies[0:len(residsPlus)] = nProgenies[0:len(residsPlus)] + residsPlus
                    nResids = 0
                else:
                    nProgenies = nProgenies + residsPlus[0:len(nProgenies)]
                    nResids = nNextPop - nProgenies.sum()
            
            if nPairs > nPairsNow:
                nProgenies = torch.concat([nProgenies, 
                                           torch.tensor(0, device = self.device).repeat_interleave(nPairs - nPairsNow)])
            
        elif allocateMethod == "weightedAllocation":
            if self.SCEachPair is None:
                self.computeSCEachPair(crossTableWoRA = crossTableWoRA)
            SCEachPair = self.SCEachPair
            
            weightedSCEachPair = torch.matmul(SCEachPair, h)
            
            nProgenies0 = torch.round(nNextPop * torch.exp(weightedSCEachPair - torch.max(weightedSCEachPair)) / 
                                      torch.exp(weightedSCEachPair - torch.max(weightedSCEachPair)).sum() /
                                      minimumUnitAllocate) * minimumUnitAllocate
            
            ordList = torch.sort(weightedSCEachPair, descending = True).indices.tolist()
            crossTableWoRASorted = crossTableWoRA.iloc[ordList]
            nProgenies = nProgenies0[ordList]
            
            nResids = nNextPop - nProgenies.sum()
            
            if nResids > 0:
                while nResids > 0:
                    nResidsSurplus = (nResids % minimumUnitAllocate).to(torch.int)
                    nResidsQuotient = (nResids // minimumUnitAllocate).to(torch.int)

                    if nResidsQuotient >= 1:
                        residsPlus = torch.tensor(minimumUnitAllocate, device = self.device).repeat_interleave(nResidsQuotient)
                    else:
                        residsPlus = torch.tensor(0, device = self.device).repeat_interleave(nResidsSurplus)
                            
                    if nResidsSurplus > 0:
                        residsPlus[0:nResidsSurplus] = residsPlus[0:nResidsSurplus] + 1
                
                    if len(nProgenies) >= len(residsPlus): 
                        nProgenies[0:len(residsPlus)] = nProgenies[0:len(residsPlus)] + residsPlus
                        nResids = 0
                    else:
                        nProgenies = nProgenies + residsPlus[0:len(nProgenies)]
                        nResids = nNextPop - nProgenies.sum()
            elif nResids < 0:
                    nResidsSurplus = (torch.abs(nResids) % minimumUnitAllocate).to(torch.int)
                    nResidsQuotient = (torch.abs(nResids) // minimumUnitAllocate).to(torch.int)            
                    
                    if nResidsQuotient >= 1:
                        residsMinus = torch.cat(
                            (torch.tensor(1, device = self.device).repeat_interleave(nResidsSurplus),
                             torch.tensor(minimumUnitAllocate, device = self.device).repeat_interleave(nResidsQuotient)),
                            dim = 0)
                    else:
                        residsMinus = torch.tensor(1, device = self.device).repeat_interleave(nResidsSurplus)
                        
                    crossNoWithNonZero = (nProgenies != 0).nonzero(as_tuple = False).squeeze()
                    
                    nProgenies[torch.flip(
                        torch.flip(crossNoWithNonZero, 
                                   dims = [0])[range(len(residsMinus))], 
                        dims = [0]
                        )] = nProgenies[torch.flip(
                            torch.flip(crossNoWithNonZero, 
                                       dims = [0])[range(len(residsMinus))], 
                            dims = [0]
                            )] - residsMinus
 
        elif allocateMethod == "userSpecific":
            crossTableWoRASorted = crossTableWoRA
            nProgenies = self.nProgenies
            
        
        positivePairs = (nProgenies > 0).nonzero(as_tuple = False).squeeze()
        nProgenies = nProgenies[positivePairs]
        nPairs = len(nProgenies)
        crossTableWoRASorted = crossTableWoRASorted.iloc[positivePairs.tolist()]
        crossTableWoRASorted = crossTableWoRASorted.set_index(pd.Index(range(nPairs)))
        indNames = self.indNames
        
        if isinstance(indNames, str):
            indNames = [f'{indNames}_{i:0{len(str(nPairs))}d}'.format(i)
                        for i in range(1, nPairs + 1)]
        elif isinstance(indNames, list):
            if len(indNames) == 1:
                indNames = [f'{indNames[0]}_{i:0{len(str(nPairs))}d}'.format(i)
                            for i in range(1, nPairs + 1)]
                
                
        nProgenies = nProgenies.to(torch.int)
        
        crossTable = pd.concat(
            [crossTableWoRASorted,
            pd.DataFrame({
                "n": nProgenies.to(device = torch.device("cpu")),
                "names": indNames
                })], 
            axis = 1)

        
        self.crossTable = crossTable
        self.indNames = indNames
        self.nProgenies = nProgenies
        self.nPairs = nPairs
        
    
    def makeCrossTableWoRA(self):
        matingMethod = self.matingMethod
        
        if matingMethod != "useSpecific":
            if matingMethod == "randomMate":
                crossTableWoRA = self.randomMate()
            elif matingMethod == "roundRobin":
                crossTableWoRA = self.roundRobin()
            elif matingMethod == "diallel":
                crossTableWoRA = self.diallel()
            elif matingMethod == "diallelWithSelfing":
                crossTableWoRA = self.diallelWithSelfing()
            elif matingMethod == "selfing":
                crossTableWoRA = self.selfing()
            elif matingMethod == "doubledHaploid":
                crossTableWoRA = self.doubledHaploid()
        
        self.crossTableWoRA = crossTableWoRA


    def randomMate(self):
        if self.selCands is None:
            selCands = self.selectParentCands()
        selCands = self.selCands
            
        nNextPop = self.nNextPop
        
        random.seed(a = self.seedSimRM)
        selCandsRand1 = random.choices(selCands, k = nNextPop)
        selCandsRand2 = random.choices(selCands, k = nNextPop)
        
        crossTableWoRA = pd.DataFrame(
            {
                "ind1": selCandsRand1,
                "ind2": selCandsRand2
            }
        )
        
        return crossTableWoRA


    def roundRobin(self):
        if self.selCands is None:
            selCands = self.selectParentCands()
        selCands = self.selCands

            
        random.seed(a = self.seedSimRM)
        selCandsRand = random.sample(selCands, k = len(selCands))
        selCandsRandOther = selCandsRand[1:len(selCands)]
        selCandsRandOther.append(selCandsRand[0])
        
        crossTableWoRA = pd.DataFrame(
            {
                "ind1": selCandsRand,
                "ind2": selCandsRandOther
            }
        )
        
        return crossTableWoRA
        
    def diallel(self):
        if self.selCands is None:
            selCands = self.selectParentCands()
        selCands = self.selCands

            
        random.seed(a = self.seedSimRM)
        selCandsRand = random.sample(selCands, k = len(selCands))
        selCandsRandOther = selCandsRand[1:len(selCands)]
        selCandsRandOther.append(selCandsRand[0])
        
        crossTableWoRA = pd.DataFrame(list(combinations(selCands, 2)),
                                      columns = ["ind1", "ind2"])
        
        return crossTableWoRA
    
    def diallelWithSelfing(self):
        if self.selCands is None:
            selCands = self.selectParentCands()
        selCands = self.selCands

            
        
        crossTableDiallel = pd.DataFrame(list(combinations(selCands, 2)),
                                      columns = ["ind1", "ind2"])
        
        corssTableSelfing = pd.DataFrame(
            {
                "ind1": selCands,
                "ind2": selCands
            }
            )
        
        crossTableWoRA = pd.concat([crossTableDiallel,
                                   corssTableSelfing])
        crossTableWoRA.index = range(len(crossTableWoRA))

        
        return crossTableWoRA
    
    
    def selfing(self):
        if self.selCands is None:
            selCands = self.selectParentCands()
        selCands = self.selCands

            
        crossTableWoRA = pd.DataFrame(
            {
                "ind1": selCands,
                "ind2": selCands
            }
            )
        
        
        return crossTableWoRA
    
    
    def doubledHaploid(self):
        if self.selCands is None:
            selCands = self.selectParentCands()
        selCands = self.selCands


        crossTableWoRA = pd.DataFrame(
            {
                "ind1": selCands,
                "ind2": None
            }
            )
        
        
        return crossTableWoRA


    def makeCrosses(self):
        requires_grad = self.requires_grad
        
        if not requires_grad:
            newIndsList = self.makeCrossesNoGrad()
        else:
            newIndsList = self.makeCrossesGrad()
        
        return newIndsList
    
    
    def makeCrossesNoGrad(self):
        if self.matingMethod not in ["doubledHaploid", "nonCross"]:
            if self.crossTable is None:
                self.designResourceAllocation()
                
            crossTable = self.crossTable
            parentPopulation = self.parentPopulation
            indNamesParentPop = parentPopulation.indNames
            seed = self.seedSimMC
            
            
            random.seed(seed)
            
            newIndsList = []
            for pairNo in range(len(crossTable)):
                parent1Name = crossTable.ind1[pairNo]
                parent2Name = crossTable.ind2[pairNo]
                whichParent1 = (torch.tensor([indNamesParentPop[i] == parent1Name
                                 for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()
                whichParent2 = (torch.tensor([indNamesParentPop[i] == parent2Name
                                 for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()
          
                newIndsNow = self.makeSingleCross(ind1 = parentPopulation.indsList[whichParent1],
                                                  ind2 = parentPopulation.indsList[whichParent2],
                                                  names = crossTable.names[pairNo],
                                                  n = self.nProgenies[pairNo])
                
                newIndsList.append(newIndsNow)
            
            newIndsList = flattenList(newIndsList)
            
        elif self.matingMethod == "doubledHaploid": 
            newIndsList = self.makeDHsNoGrad()
        elif self.matingMethod == "nonCross":
            parentPopulation = self.parentPopulation
            selCands = self.selCands
            
            if len(selCands) >= 2:
                whichSelCands = (torch.tensor([indNamesParentPop[i] in selCands 
                                               for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()
            else:
                whichSelCands = (torch.tensor([indNamesParentPop[i] == selCands
                                               for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()

            newIndsList = [parentPopulation.indsList[selCandsNo]
                           for selCandsNo in whichSelCands]
        
        
        return newIndsList
    
    
    def makeCrossesGrad(self):
        if self.matingMethod not in ["doubledHaploid", "nonCross"]:
            if self.crossTableWoRA is None:
                self.makeCrossTableWoRA()
            crossTableWoRA = self.crossTableWoRA
            nPairs = len(crossTableWoRA)
        
            if self.SCEachPair is None:
                self.computeSCEachPair(crossTableWoRA = crossTableWoRA)
            SCEachPair = self.SCEachPair
            h = self.h
            nNextPop = self.nNextPop
            
            parentPopulation = self.parentPopulation
            indNamesParentPop = parentPopulation.indNames
            indNames = self.indNames
        
            # weightedSCEachPair = torch.mm(SCEachPair, h.unsqueeze(1)).squeeze()
            if self.device == torch.device("mps"):
                weightedSCEachPair = torch.mm(SCEachPair.to(torch.device("cpu")), h.unsqueeze(1).to(torch.device("cpu"))).squeeze().to(self.device)
            else:
                weightedSCEachPair = torch.matmul(SCEachPair, h)
            

            if isinstance(indNames, str):
                indNames = [f'{indNames}_{i:0{len(str(nPairs))}d}'.format(i)
                            for i in range(1, nPairs + 1)]
            elif isinstance(indNames, list):
                if len(indNames) == 1:
                    indNames = [f'{indNames[0]}_{i:0{len(str(nPairs))}d}'.format(i)
                                for i in range(1, nPairs + 1)]
        
            seed = self.seedSimMC
            random.seed(seed)
            newIndsList = []
            nProgenies0 = torch.zeros(size = (len(crossTableWoRA), ), device = self.device)
            nProgenies0 = nProgenies0.to(torch.int)
            for progenyNo in range(nNextPop): 
                sampledNow = F.gumbel_softmax(weightedSCEachPair, tau = 1, hard = True)
                whichSample = torch.nonzero(input = sampledNow, as_tuple = False).squeeze()
                
                parent1Name = crossTableWoRA.ind1.loc[whichSample.item()]
                parent2Name = crossTableWoRA.ind2.loc[whichSample.item()]
                whichParent1 = (torch.tensor([indNamesParentPop[i] == parent1Name
                                              for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()
                whichParent2 = (torch.tensor([indNamesParentPop[i] == parent2Name
                                              for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()

        
                newInd = self.makeSingleCross(ind1 = parentPopulation.indsList[whichParent1], 
                                              ind2 = parentPopulation.indsList[whichParent2],
                                              n = 1,
                                              names = [f'{indNames[whichSample.item()]}-{nProgenies0[whichSample]}'],
                                              sampledForAutoGrad = sampledNow)[0]

                nProgenies0[whichSample] = nProgenies0[whichSample] + 1
                newIndsList.append(newInd)
        
            ordList = torch.sort(nProgenies0, descending = True).indices.tolist()
            crossTableWoRASorted = crossTableWoRA.iloc[ordList]
            crossTableWoRASorted = crossTableWoRASorted.set_index(pd.Index(range(nPairs)))
            
            nProgenies = nProgenies0[ordList]
            positivePairs = (nProgenies > 0).nonzero(as_tuple = False).squeeze()
            nProgenies = nProgenies[positivePairs]
            nPairs = len(nProgenies)
            crossTableWoRASorted = crossTableWoRASorted.iloc[positivePairs.tolist()]
            
            
            ordListPositivePairs = [ordList[positivePairNo] for positivePairNo in positivePairs]
            indNames = [indNames[ordNo] for ordNo in ordListPositivePairs]
            indNames = [indNames[positivePair] for positivePair in positivePairs]
            
            
            crossTable = pd.concat(
                [crossTableWoRASorted,
                pd.DataFrame({
                    "n": nProgenies.to(device = torch.device("cpu")),
                    "names": indNames
                    })], 
                axis = 1)

            
            self.crossTable = crossTable
            self.indNames = indNames
            self.nProgenies = nProgenies
            self.nPairs = nPairs

        elif self.matingMethod == "doubledHaploid": 
            newIndsList = self.makeDHsGrad()
        elif self.matingMethod == "nonCross":
            parentPopulation = self.parentPopulation
            selCands = self.selCands
            
            if len(selCands) >= 2:
                whichSelCands = (torch.tensor([indNamesParentPop[i] in selCands 
                                               for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()
            else:
                whichSelCands = (torch.tensor([indNamesParentPop[i] == selCands
                                               for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()

            newIndsList = [parentPopulation.indsList[selCandsNo]
                           for selCandsNo in whichSelCands]
        
        
        return newIndsList
    
    
    def makeDHsNoGrad(self):
        if self.crossTable is None:
            self.designResourceAllocation()
            
        crossTable = self.crossTable
        parentPopulation = self.parentPopulation
        indNamesParentPop = parentPopulation.indNames
        seed = self.seedSimMC
        
        crossTable.ind2 = None
        
        random.seed(seed)
        
        newIndsList = []
        for pairNo in range(len(crossTable)):
            parent1Name = crossTable.ind1[pairNo]
            whichParent1 = (torch.tensor([indNamesParentPop[i] == parent1Name
                             for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()
        
            newIndsNow = self.makeSingleDH(ind1 = parentPopulation.indsList[whichParent1],
                                           names = crossTable.names[pairNo],
                                           n = self.nProgenies[pairNo])
            
            newIndsList.append(newIndsNow)
        
        newIndsList = flattenList(newIndsList)
        
        
        return newIndsList
    
    
    def makeDHsGrad(self):
        if self.crossTableWoRA is None:
            self.makeCrossTableWoRA()
        crossTableWoRA = self.crossTableWoRA
        nPairs = len(crossTableWoRA)
        
        if self.SCEachPair is None:
            self.computeSCEachPair(crossTableWoRA = crossTableWoRA)
        SCEachPair = self.SCEachPair
        h = self.h
        nNextPop = self.nNextPop
            
        parentPopulation = self.parentPopulation
        indNamesParentPop = parentPopulation.indNames
        indNames = self.indNames
        crossTableWoRA.ind2 = None
        
        weightedSCEachPair = torch.matmul(SCEachPair, h)

        if isinstance(indNames, str):
            indNames = [f'{indNames}_{i:0{len(str(nPairs))}d}'.format(i)
                        for i in range(1, nPairs + 1)]
        elif isinstance(indNames, list):
            if len(indNames) == 1:
                indNames = [f'{indNames[0]}_{i:0{len(str(nPairs))}d}'.format(i)
                            for i in range(1, nPairs + 1)]
        
        seed = self.seedSimMC
        random.seed(seed)
        newIndsList = []
        nProgenies0 = torch.zeros(size = (len(crossTableWoRA), ), device = self.device)
        nProgenies0 = nProgenies0.to(torch.int)
        for progenyNo in range(nNextPop): 
            sampledNow = F.gumbel_softmax(weightedSCEachPair, tau = 1, hard = True)
            whichSample = torch.nonzero(input = sampledNow, as_tuple = False).squeeze()
            
            parent1Name = crossTableWoRA.ind1.loc[whichSample.item()]
            whichParent1 = (torch.tensor([indNamesParentPop[i] == parent1Name
                             for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()

        
            newInd = self.makeSingleDH(ind1 = parentPopulation.indsList[whichParent1],
                                       n = 1,
                                       names = [f'{indNames[whichSample.item()]}-{nProgenies0[whichSample]}'],
                                       sampledForAutoGrad = sampledNow)[0]

            nProgenies0[whichSample] = nProgenies0[whichSample] + 1
            newIndsList.append(newInd)
        
        ordList = torch.sort(nProgenies0, descending = True).indices.tolist()
        crossTableWoRASorted = crossTableWoRA.iloc[ordList]
        nProgenies = nProgenies0[ordList]
        
        positivePairs = (nProgenies > 0).nonzero(as_tuple = False).squeeze()
        nProgenies = nProgenies[positivePairs]
        nPairs = len(nProgenies)
        crossTableWoRASorted = crossTableWoRASorted.iloc[positivePairs.tolist()]
        crossTableWoRASorted = crossTableWoRASorted.set_index(pd.Index(range(nPairs)))
        ordListPositivePairs = [ordList[positivePairNo] for positivePairNo in positivePairs]
        indNames = [indNames[ordNo] for ordNo in ordListPositivePairs]
        indNames = [indNames[positivePair] for positivePair in positivePairs]
        
        
        crossTable = pd.concat(
            [crossTableWoRASorted,
            pd.DataFrame({
                "n": nProgenies.to(torch.device("cpu")),
                "names": indNames
                })], 
            axis = 1)

        
        self.crossTable = crossTable
        self.indNames = indNames
        self.nProgenies = nProgenies
        self.nPairs = nPairs
        
        
        return newIndsList

    
    
    
    def makeSingleCross(self, ind1, ind2, names = None, n = 1, sampledForAutoGrad = None):
        lociEffectsTrue = self.parentPopulation.lociEffectsTrue
        # lociEffectsEst = self.lociEffectsEst
        
        gam1 = ind1.generateGametes(n = n, sampledForAutoGrad = sampledForAutoGrad)
        gam2 = ind2.generateGametes(n = n, sampledForAutoGrad = sampledForAutoGrad)
        
            
        # Generate tensor list
        haploList = [torch.cat((g1.view(1, -1), g2.view(1, -1)), dim = 0) for g1, g2 in zip(gam1, gam2)]
        
        # Create new list of Individual class objects
        if names is None:
            newInds = [
                Individual(haplotype = Haplotype(genoMap = ind1.haplotype.genoMap, 
                                                 genoHapInd = haploList[i]),
                           parent1 = ind1.name,
                           parent2 = ind2.name,
                           device = self.device,
                           lociEffectsTrue = lociEffectsTrue) 
                for i in range(n)
                ]
        else:
            if isinstance(names, str):
                names = [f"{names}-{no}" for no in range(n)]
            elif isinstance(names, list):
                if n != 1 and len(names) == 1:
                    names = [f"{names[0]}-{no}" for no in range(n)]
            newInds = [
                Individual(haplotype = Haplotype(genoMap = ind1.haplotype.genoMap, 
                                                 genoHapInd = haploList[i]),
                           lociEffectsTrue = lociEffectsTrue,
                           parent1 = ind1.name,
                           parent2 = ind2.name,
                           device = self.device,
                           name = names[i]) 
                for i in range(n)
                ]
        
        return newInds
            
    
    def makeSingleDH(self, ind1, names = None, n = 1, sampledForAutoGrad = None):
        lociEffectsTrue = self.parentPopulation.lociEffectsTrue
        # lociEffectsEst = self.lociEffectsEst
        
        gam1 = ind1.generateGametes(n = n, sampledForAutoGrad = sampledForAutoGrad)
        
        # Generate tensor list
        haploList = [torch.cat((g1.view(1, -1), g1.view(1, -1)), dim = 0) for g1 in gam1]
        
        # Create new list of Individual class objects
        if names is None:
            newInds = [
                Individual(haplotype = Haplotype(genoMap = ind1.haplotype.genoMap, 
                                                 genoHapInd = haploList[i]),
                           parent1 = ind1.name,
                           parent2 = None,
                           device = self.device,
                           lociEffectsTrue = lociEffectsTrue) 
                for i in range(n)
                ]
        else: 
            if type(names) == str:
                names = [f"{names}-{no}" for no in range(n)]
            elif type(names) == list:
                if n != 1 and len(names) == 1:
                    names = [f"{names[0]}-{no}" for no in range(n)]
            newInds = [
                Individual(haplotype = Haplotype(genoMap = ind1.haplotype.genoMap, 
                                                 genoHapInd = haploList[i]),
                           lociEffectsTrue = lociEffectsTrue,
                           parent1 = ind1.name,
                           parent2 = None,
                           device = self.device,
                           name = names[i]) 
                for i in range(n)
                ]
        
        return newInds
    
    def computeSCEachPair(self, crossTableWoRA):
        indNamesParentPop = self.parentPopulation.indNames
        weightedAllocationMethod = self.weightedAllocationMethod
        
        SCAll = None
            
        if "BV" in weightedAllocationMethod:
            if self.BV is None:
                self.computeBV()
            SCNow = self.BV
            
            if SCAll is None:
                SCAll = SCNow
            else:
                SCAll = torch.cat((SCAll, SCNow), dim = 1)

        if "WBV" in weightedAllocationMethod:
            if self.WBV is None:
                self.computeWBV()
            SCNow = self.WBV
            
            if SCAll is None:
                SCAll = SCNow
            else:
                SCAll = torch.cat((SCAll, SCNow), dim = 1)        
                
        if "userSC" in weightedAllocationMethod:
            if self.userSC is not None:
                SCNow = self.userSC
            
                if SCAll is None:
                    SCAll = SCNow
                else:
                    SCAll = torch.cat((SCAll, SCNow), dim = 1)     
            
            
            
        if SCAll is not None:
            SCMean = SCAll.mean(dim = 0, keepdim = True)
            SCSd = SCAll.std(dim = 0, keepdim = True)
            
            SCScaled = (SCAll - SCMean) / SCSd
            if torch.any(SCSd == 0):
                whichSdZero = (SCSd[0] == 0).nonzero(as_tuple = False).squeeze()
                SCScaled[:, whichSdZero] = 0
        else:
            SCScaled = None
            
            
        if self.matingMethod != "doubledHaploid":
            if SCScaled is not None:
                # if len(crossTableWoRA) >= 2:
                    # whichParent1 = (torch.tensor([indNamesParentPop[i] in crossTableWoRA.ind1.tolist() 
                    #                               for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()
                    # whichParent2 = (torch.tensor([indNamesParentPop[i] in crossTableWoRA.ind2.tolist() 
                    #                               for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()
                whichParent1 = [indNamesParentPop.index(x) if x in indNamesParentPop  else None for x in crossTableWoRA.ind1.tolist()]
                whichParent2 = [indNamesParentPop.index(x) if x in indNamesParentPop  else None for x in crossTableWoRA.ind2.tolist()]

                # else:
                #     whichParent1 = (torch.tensor([indNamesParentPop[i] == crossTableWoRA.ind1.tolist() 
                #                                   for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()
                #     whichParent2 = (torch.tensor([indNamesParentPop[i] == crossTableWoRA.ind2.tolist() 
                #                                   for i in range(len(indNamesParentPop))], device = self.device)).nonzero(as_tuple = False).squeeze()
                    
                SCEachPair = (SCScaled[whichParent1, :] + 
                              SCScaled[whichParent2, :]) / 2
            else:
                SCEachPair = None
            
            if self.includeGVP:
                if self.GVP is None:
                    self.computeGVP()
                SCEachPair = torch.cat([SCEachPair,
                                        self.GVP],
                                       dim = 1)
        else:
            SCEachPair = SCScaled
        
        
        self.SCEachPair = SCEachPair

    def computeBV(self):
        # parentPopulation = population
        # lociEffectsEst = torch.concat([torch.tensor(0, device = self.device).repeat_interleave(lociEffectsTrue.size(1)).unsqueeze(0), lociEffectsTrue], dim = 0)
        parentPopulation = self.parentPopulation
        lociEffectsEst = self.lociEffectsEst
        genoMat = parentPopulation.genoMat
        if self.includeIntercept:
            genoMatIncInt = torch.cat(
                (torch.ones((genoMat.size(0), 1), device = self.device),
                 genoMat), dim = 1
                )
        else:
            genoMatIncInt = genoMat
            
        BV = torch.mm(genoMatIncInt, lociEffectsEst)
        self.BV = BV

            
    def computeWBV(self):
        # parentPopulation = population
        parentPopulation = self.parentPopulation
        lociEffectsEst = self.lociEffectsEst
        parentPopulation.obtainAlleleFreq()
        genoMat = parentPopulation.genoMat
        
        
        favAllel = torch.sign(lociEffectsEst[1:, ])
        w = parentPopulation.af.repeat_interleave(favAllel.size(1)).reshape(shape = favAllel.size())

        for traitNo in range(favAllel.size(1)):
            w[favAllel[:, traitNo] == 0, traitNo] = 1 - w[favAllel[:, traitNo] == 0, traitNo]
                  

        w[w == 0] = 1 # give weight 1 for fixed alleles
        wSqrtInv = 1 / torch.sqrt(w)

        wLociEffects = torch.concat([lociEffectsEst[0, ].unsqueeze(0),
                                    lociEffectsEst[1:, ] * wSqrtInv], dim = 0)

        if self.includeIntercept:
            genoMatIncInt = torch.cat(
                (torch.ones((genoMat.size(0), 1), device = self.device),
                 genoMat), dim = 1
                )
        else:
            genoMatIncInt = genoMat
            
        WBV = torch.mm(genoMatIncInt, wLociEffects)
        self.WBV = WBV
    
    def computeGVP(self):
        parentPopulation = self.parentPopulation
        parentPopulation.obtainHaploArray()
        haploArray = parentPopulation.haploArray
        lociEffectsEst = self.lociEffectsEst
        nTraits = lociEffectsEst.size(1)
        
        genoMap = parentPopulation.genoMap
        chrNames = genoMap["chr"].tolist()
        chrNamesUniq = sorted(list(set(chrNames)), key = None, reverse = False)
        nChrs = len(chrNamesUniq)
        chrNos = list(range(nChrs))

        selCands = self.selCands
        indNamesAll = parentPopulation.indNames
        
        if self.d12Mat2List is None:
            self.computeDMat()
            
        crossTableWoRA = self.crossTableWoRA
        indicesChrList = self.indicesChrList
        d12Mat2List = self.d12Mat2List
        d13Mat2List = self.d13Mat2List
        
        nPairs = len(crossTableWoRA)
        
        haploArraySel = haploArray[
            torch.nonzero(
                torch.tensor(
                    [indName in selCands for indName in indNamesAll]
                    ), as_tuple = False
                ).squeeze(), :, :
            ]
        nIndSel = haploArraySel.size(0)
        

        
        gvpWithinInd = torch.zeros(size = torch.Size([nIndSel, nTraits]),
                                   device = self.device)
        for indNo in range(nIndSel):
            genoVecP1 = haploArraySel[indNo, :, 0] * 2
            genoVecP2 = haploArraySel[indNo, :, 1] * 2
            for chrNo in chrNos:
                indicesChr = indicesChrList[chrNo]
                genoVecP1Chr = genoVecP1[indicesChr]
                genoVecP2Chr = genoVecP2[indicesChr]
                gammaWithinIndChr = genoVecP1Chr - genoVecP2Chr
                gammaProdMat = gammaWithinIndChr.unsqueeze(1) * gammaWithinIndChr.unsqueeze(0)
        
                for traitNo in range(nTraits):
                    d12Mat2 = d12Mat2List[traitNo][chrNo]
                    gvpWithinInd[indNo, traitNo] = gvpWithinInd[indNo, traitNo] + torch.sum(d12Mat2 * gammaProdMat)
        
        genVarProgenies = torch.zeros(torch.Size([nPairs, nTraits]),
                                      device = self.device)
        for pairNo in range(nPairs):
            parentNamesNow = crossTableWoRA.iloc[pairNo].tolist()
            parentIndicesNow = torch.tensor(
                [selCands.index(parentName) 
                if parentName in selCands else None 
                for parentName in parentNamesNow],
                device = self.device
                )
            gvpWithinIndPair = gvpWithinInd[parentIndicesNow, :]
            gvpWithinIndSum = gvpWithinIndPair.sum(dim = 0)
            
            if self.targetGenForGVP >= 2:
                genoVecP1 = haploArraySel[parentIndicesNow[0], :, 0] * 2
                genoVecP2 = haploArraySel[parentIndicesNow[0], :, 1] * 2
                genoVecP3 = haploArraySel[parentIndicesNow[1], :, 0] * 2
                genoVecP4 = haploArraySel[parentIndicesNow[1], :, 1] * 2
                
                gvpBeyondIndSum = torch.zeros_like(gvpWithinIndSum, device = self.device)
                for chrNo in chrNos:
                    indicesChr = indicesChrList[chrNo]
                    genoVecP1Chr = genoVecP1[indicesChr]
                    genoVecP2Chr = genoVecP2[indicesChr]
                    genoVecP3Chr = genoVecP3[indicesChr]
                    genoVecP4Chr = genoVecP4[indicesChr]
                    gamma13 = genoVecP1Chr - genoVecP3Chr
                    gamma14 = genoVecP1Chr - genoVecP4Chr
                    gamma23 = genoVecP2Chr - genoVecP3Chr
                    gamma24 = genoVecP2Chr - genoVecP4Chr
                        
                    gammaProdMat = gamma13.unsqueeze(1) * gamma13.unsqueeze(0) + gamma14.unsqueeze(1) * gamma14.unsqueeze(0) + gamma23.unsqueeze(1) * gamma23.unsqueeze(0) + gamma24.unsqueeze(1) * gamma24.unsqueeze(0)
                    
                    for traitNo in range(nTraits):
                        d13Mat2 = d13Mat2List[traitNo][chrNo]
                        gvpBeyondIndSum[traitNo] = gvpBeyondIndSum[traitNo] + torch.sum(d13Mat2 * gammaProdMat)
            else:  
                gvpBeyondIndSum = 0
            
            gvpEachPair = gvpWithinIndSum + gvpBeyondIndSum
            genVarProgenies[pairNo, :] = gvpEachPair
        
        # genVarProgeniesMean = genVarProgenies.mean(dim = 0)
        # genVarProgeniesStd = genVarProgenies.std(dim = 0)
        
        # genVarProgeniesScaled = genVarProgenies.clone()
        # genVarProgeniesScaled = genVarProgenies
        
        genVarProgeniesMean = genVarProgenies.mean(dim = 0, keepdim = True)
        genVarProgeniesSd = genVarProgenies.std(dim = 0, keepdim = True)
        
        genVarProgeniesScaled = (genVarProgenies - genVarProgeniesMean) / genVarProgeniesSd
        if torch.any(genVarProgeniesSd == 0):
            whichSdZero = (genVarProgeniesSd[0] == 0).nonzero(as_tuple = False).squeeze()
            genVarProgeniesScaled[:, whichSdZero] = 0
        # for traitNo in range(nTraits):
        #     genVarProgeniesScaled[:, traitNo] = genVarProgeniesScaled[:, traitNo] - genVarProgeniesMean[traitNo]
            
        #     if genVarProgeniesStd[traitNo] != 0:
        #         genVarProgeniesScaled[:, traitNo] = genVarProgeniesScaled[:, traitNo] / genVarProgeniesStd[traitNo]
                
        
        self.GVP = genVarProgeniesScaled

    
    
    def computeDMat(self):
        parentPopulation = self.parentPopulation
        parentPopulation.obtainHaploArray()
        lociEffectsEst = self.lociEffectsEst
        nTraits = lociEffectsEst.size(1)
        
        genoMap = parentPopulation.genoMap
        chrNames = genoMap["chr"].tolist()
        chrNamesUniq = sorted(list(set(chrNames)), key = None, reverse = False)
        nChrs = len(chrNamesUniq)
        chrNos = list(range(nChrs))
        targetGenForGVP = self.targetGenForGVP

        recombRatesList = computeRecombBetweenMarkers(
            genoMap = genoMap, device = self.device
            )
        
        indicesChrList = []
        d12Mat1List = []
        d13Mat1List = []
        d12Mat2List = [[] for _ in range(nTraits)]
        d13Mat2List = [[] for _ in range(nTraits)]
        for chrNo in chrNos:
            recombRatesMat = recombRatesList[chrNo]
            if targetGenForGVP >= 2:
                l = targetGenForGVP - 1
                ckMat = 2. * recombRatesMat * (1. - (0.5 ** l) * (1. - 2. * recombRatesMat) ** l) / (1. + 2. * recombRatesMat)
                d12Mat1 = 0.25 * (1. - ckMat) * (1. - 2. * recombRatesMat)
                d13Mat1 = 0.25 * (1. - 2. * ckMat - (0.5 * (1. - 2. * recombRatesMat)) ** l)
            else:
                d12Mat1 = 0.25 * (1. - 2. * recombRatesMat)
                d13Mat1 = None
            d12Mat1List.append(d12Mat1)
            d13Mat1List.append(d13Mat1)
            
            indicesChr = torch.nonzero(
                torch.tensor(
                    [chrNameEach in chrNamesUniq[chrNo] 
                     for chrNameEach in chrNames]
                    ), as_tuple = False
                ).squeeze()
            indicesChrList.append(indicesChr)
        
            for traitNo in range(nTraits):
                lociEffectsNow = lociEffectsEst[:, traitNo] 
                lociEffectsNowChr = lociEffectsNow[indicesChr]
                lociEffectsMatPow = lociEffectsNowChr.unsqueeze(1) * lociEffectsNowChr.unsqueeze(0)
                
                d12Mat2 = d12Mat1 * lociEffectsMatPow
                if d13Mat1 is not None:
                    d13Mat2 = d13Mat1 * lociEffectsMatPow
                else:
                    d13Mat2 = None
                d12Mat2List[traitNo].append(d12Mat2)
                d13Mat2List[traitNo].append(d13Mat2)
        
        self.indicesChrList = indicesChrList
        self.d12Mat1List = d12Mat1List
        self.d12Mat2List = d12Mat2List
        self.d13Mat1List = d13Mat1List
        self.d13Mat2List = d13Mat2List


        



class BsInfo:
    def __init__(self,
                 # traitInfo,
                 founderHaplo,
                 genoMap,
                 # initPopulation,
                 founderIsInitPop = False,
                 nGenerationRM = 100,
                 nGenerationRSM = 10,
                 nGenerationRM2 = 3,
                 propSelRS = 0.1,
                 lociEffectsTrue = None,
                 bsName = "Undefined",
                 popNameBase = "Population",
                 initIndNames = None,
                 herit = None,
                 nRep = None,
                 # envSpecificEffects = None,
                 # residCor = None,
                 initGeneration = 0,
                 includeIntercept = True,
                 genGamMethod = 2,
                 createInitPopulation = True,
                 device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
                 verbose = True):
        
        founderHaplo = founderHaplo.to(device)
        lociEffectsTrue = lociEffectsTrue.to(device)
        populations = []
        if createInitPopulation:
            initPopulation = createInitPop(founderHaplo = founderHaplo,
                                           genoMap = genoMap,
                                           lociEffectsTrue = lociEffectsTrue,
                                           founderIsInitPop = founderIsInitPop,
                                           nGenerationRM = nGenerationRM,
                                           nGenerationRSM = nGenerationRSM,
                                           nGenerationRM2 = nGenerationRM2,
                                           propSelRS = propSelRS,
                                           indNames = None,
                                           popName = f"{popNameBase}{initGeneration}",
                                           generation = initGeneration,
                                           herit = herit,
                                           nRep = nRep,
                                           phenotypicValues = None,
                                           includeIntercept = includeIntercept,
                                           genGamMethod = genGamMethod,
                                           device = device,
                                           verbose = verbose)
            populations.append(initPopulation)
        
        
        
        self.bsName = bsName
        self.founderHaplo = founderHaplo
        self.initIndNames = initIndNames
        self.initGeneration = initGeneration
        self.genoMap = genoMap
        self.lociEffectsTrue = lociEffectsTrue
        self.populations = populations
        self.crossInfoList = []
        self.popNameBase = popNameBase
        self.generation = initGeneration
        self.herit = herit
        self.nRep = nRep
        self.includeIntercept = includeIntercept
        self.genGamMethod = genGamMethod
        self.device = device
        
        # self.envSpecificEffects = envSpecificEffects
        # self.residCor = residCor
    def nextGeneration(self, crossInfo):
        populations = self.populations
        generation = self.generation
        crossInfoList = self.crossInfoList
        latestPop = populations[len(populations) - 1]
        
        # crossInfoName = f"{generation}_to_{generation + 1}"
        generation = generation + 1
        newPopName = f"{self.popNameBase}{generation}"
        
        newPop = Population(indsList = crossInfo.makeCrosses(),
                            name = newPopName,
                            generation = generation,
                            lociEffectsTrue = self.lociEffectsTrue,
                            crossInfo = crossInfo,
                            herit = latestPop.herit,
                            nRep = latestPop.nRep,
                            phenotypicValues = None,
                            includeIntercept = self.includeIntercept,
                            device = self.device,
                            verbose = False)
        populations.append(newPop)
        crossInfoList.append(crossInfo)
        
        
        self.generation = generation
        self.populations = populations
        self.crossInfoList = crossInfoList
        
    def removeLatestPop(self):
        populations = self.populations
        generation = self.generation
        crossInfoList = self.crossInfoList
        
        if len(populations) == 1:
            warnings.warn("We cannot remove all the populations. Please remain at least 1 population or redefine the initial population with `self.__init__`!")
        else:
            populations.remove(populations[len(populations)])
            crossInfoList.remove(crossInfoList[len(crossInfoList) - 1])
            generation = generation - 1

        self.generation = generation
        self.populations = populations
        self.crossInfoList = crossInfoList
        
    def removeInitialPop(self):
        populations = self.populations
        generation = self.generation
        crossInfoList = self.crossInfoList
        
        if len(populations) == 1:
            warnings.warn("We cannot remove all the populations. Please remain at least 1 population or redefine the initial population with `self.__init__`!")
        else:
            populations.remove(populations[0])
            crossInfoList.remove(crossInfoList[0])
            generation = generation - 1
            
        for popNo in range(len(populations)):
            populations[popNo].generation = populations[popNo].generation - 1

        self.generation = generation
        self.populations = populations
        self.crossInfoList = crossInfoList
    
    def overGeneration(self, targetPop = None):
        populations = self.populations
        
        if targetPop is None:
            targetPop = list(range(len(populations)))
        

        targetPop = [targetPopNow for targetPopNow in targetPop if targetPopNow in list(range(len(populations)))]
        targetPopulations = [populations[targetPopNow] for targetPopNow in targetPop]
        nTargetPop = len(targetPop)
        
        indsOverGeneration = []
        
        for i in range(nTargetPop):
            indsNow = targetPopulations[i].indsList
            indsOverGeneration.append(indsNow)
        
        indsOverGeneration = flattenList(indsOverGeneration)

        popOGName = self.popNameBase+"_".join(list(map(str, targetPop)))
        
        popOverGeneration = Population(
            indsList = indsOverGeneration,
            name = popOGName,
            generation = None,
            lociEffectsTrue = self.lociEffectsTrue,
            crossInfo = None,
            herit = self.herit,
            nRep = self.nRep,
            includeIntercept = self.includeIntercept,
            device = self.device,
            verbose = False
            )
        
        return popOverGeneration
    
    def parentInd(self, indName):
        populations = self.populations
        whichPop = torch.nonzero(
            input = torch.tensor([
                indName in [
                    ind.name for ind in populations[i].indsList
                    ] for i in range(len(populations))
                ], device = self.device), as_tuple = False).squeeze()
        indNamesNow = [ind.name for ind in populations[whichPop].indsList]
        whichInd = torch.nonzero(
            input = torch.tensor([indNameNow == indName for indNameNow in indNamesNow], device = self.device), as_tuple = False).squeeze()
        indInfo = populations[whichPop].indsList[whichInd]
        
        parent1 = indInfo.parent1
        parent2 = indInfo.parent2
        
        return [parent1, parent2]
    
    
    def deepcopy(self):
        dictAll = self.__dict__
        
        bsInfoCopied = BsInfo(
            founderHaplo = self.founderHaplo,
            genoMap = self.genoMap,
            lociEffectsTrue = self.lociEffectsTrue,
            bsName = self.bsName,
            popNameBase = self.popNameBase,
            initIndNames = self.initIndNames,
            herit = self.herit,
            nRep = self.nRep,
            initGeneration = self.initGeneration,
            includeIntercept = self.includeIntercept,
            createInitPopulation = False,
            device = self.device,
            verbose = False)
        
        for eachKey in dictAll:
            valueNow = dictAll[eachKey]
            
            if isinstance(valueNow, torch.Tensor):
                valueCopied = valueNow.clone()
            else:
                valueCopied = copy.deepcopy(valueNow)
                
            setattr(bsInfoCopied, 
                    eachKey, 
                    valueCopied)
        
        return bsInfoCopied
    
    def quickcopy(self):
        bsInfoCopied = BsInfo(
            founderHaplo = self.founderHaplo,
            genoMap = self.genoMap,
            lociEffectsTrue = self.lociEffectsTrue,
            bsName = self.bsName,
            popNameBase = self.popNameBase,
            initIndNames = self.initIndNames,
            herit = self.herit,
            nRep = self.nRep,
            initGeneration = self.initGeneration,
            includeIntercept = self.includeIntercept,
            createInitPopulation = False,
            device = self.device,
            verbose = False)
        
        for i in range(len(self.populations)):
            bsInfoCopied.populations.append(copy.copy(self.populations[i]))
        
        return bsInfoCopied



class PopulationFB:
    def __init__(self, 
                 population,
                 mrkPos,
                 device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
                 genotyping = True,
                 genotypedIndNames = None,
                 mrkNames = None,
                 mafThres = 0.05,
                 heteroThres = 1
                 # calculateGRM = True,
                 # methodsGRM = "addNOIA",
                 # calcEpistasis = False
                 ):
        
        self.population = population
        self.mrkPos = mrkPos
        self.nInd = len(population.indsList)
        self.indNames = population.indNames
        self.generation = population.generation
        self.crossInfo = population.crossInfo
        self.name = population.name
        self.verbose = population.verbose
        self.nTraits = population.nTraits
        self.device = device
        
        self.genotyping = genotyping
        
        if self.genotyping:
            self.genotyper(genotypedIndNames = genotypedIndNames,
                           mrkNames = mrkNames,
                           mafThres = mafThres,
                           heteroThres = heteroThres)

        

        
    def genotyper(self, 
                  genotypedIndNames = None,
                  mrkNames = None,
                  mafThres = 0.05,
                  heteroThres = 1):
        population = self.population
        indNames = self.indNames
        mrkPos = self.mrkPos
        nMarkers = len(mrkPos)
        
        if not isinstance(mafThres, torch.Tensor):
            mafThres = torch.tensor(mafThres, device = self.device)
        if not isinstance(heteroThres, torch.Tensor):
            heteroThres = torch.tensor(float(heteroThres), device = self.device)
        
        if genotypedIndNames is None:
            genotypedIndNames = indNames
        
        if mrkNames is None:
            mrkNames = [f"Marker_{i}" for i in range(nMarkers)]
        
        # genotypedIndNos = torch.nonzero(input = torch.tensor(
        #    [genotypedIndName in indNames for genotypedIndName in genotypedIndNames], device = self.device
        #    ), as_tuple = False).squeeze()
        genotypedIndNos = [genotypedIndName in indNames
                           for genotypedIndName in genotypedIndNames]
        genoMatFB = population.genoMat[:, mrkPos]
        genoMatFB = genoMatFB[genotypedIndNos, :]

        af = genoMatFB.sum(0) / (2 * genoMatFB.size(0))
        maf = torch.minimum(af, 1 - af)
        
        heteroRate = torch.tensor([
            len(torch.nonzero(
                        input = genoMatFB[:, mrkNo] == 1,
                        as_tuple = False
                        ).squeeze(0)
                    ) / genoMatFB.size(0) for mrkNo in range(genoMatFB.size(1))
            ], device = self.device)
         
        
        removedCond = (maf < mafThres) | (heteroRate >= heteroThres)
        remainedCond = (maf >= mafThres) & (heteroRate < heteroThres)
        removedMarkers = mrkPos[removedCond.to(torch.device("cpu"))]
        remainedMarkers = mrkPos[remainedCond.to(torch.device("cpu"))] 
        
        genoMapAll = population.genoMap
        genoMapFB = genoMapAll.loc[mrkPos, :]
        genoMapFB.iloc[:, 0] = mrkNames
        genoMapFB.rename(columns = {'lociNames': 'mrkNames'}, inplace = True)
        
        genoMatMC = genoMatFB[:, remainedCond]
        genoMapMC = genoMapFB.loc[remainedMarkers, :]
        
        self.genotypedIndNames = genotypedIndNames
        self.mrkNames = mrkNames
        self.af = af
        self.maf = maf
        self.mafThres = mafThres
        self.heteroRate = heteroRate
        self.heteroThres = heteroThres
        self.removedCond = removedCond
        self.remainedCond = remainedCond
        self.removedMarkers = removedMarkers
        self.remainedMarkers = remainedMarkers
        self.removedMarkers = removedMarkers
        self.genoMatFB = genoMatFB
        self.genoMatMC = genoMatMC
        self.genoMapFB = genoMapFB
        self.genoMapMC = genoMapMC
        self.genotyping = True
        
    
    def phenotyper(self,
                   phenotypedIndNames = None,
                   estimateGV = True,
                   estimatedGVMethod = "mean",
                   includeIntercept = True):
        population = self.population
        indNames = self.indNames
        
        if phenotypedIndNames is None:
            phenotypedIndNames = indNames
            
        population.inputPhenotypicValues()
        phenotypedIndNos = torch.nonzero(input = torch.tensor(
            [phenotypedIndName in indNames for phenotypedIndName in phenotypedIndNames], device = self.device
            ), as_tuple = False).squeeze()
        phenotypicValuesFB = population.phenotypicValues[phenotypedIndNos, :, :]
        
        trialInfoFB = {
            "nRep": population.nRep,
            "phenotypedIndNames": phenotypedIndNames
            }
        
        self.phenotypicValuesFB = phenotypicValuesFB
        self.trialInfoFB = trialInfoFB
        self.estimateGV = estimateGV
        
        if estimateGV:
            self.estimateGVByRep(
                estimatedGVMethod = estimatedGVMethod,
                includeIntercept = includeIntercept
                )
    
    def estimateGVByRep(self,
                        estimatedGVMethod = "mean",
                        includeIntercept = True):
        phenotypicValuesFB = self.phenotypicValuesFB
        trialInfoFB = self.trialInfoFB
        
        # phenotypedIndNames = trialInfoFB["phenotypedIndNames"]
        nRep = trialInfoFB["nRep"]
        
        if nRep >= 2:
            if estimatedGVMethod == "mean":
                estimatedGVByRep = phenotypicValuesFB.mean(2)
                intercept = estimatedGVByRep.mean(0)
                residualsByRep = phenotypicValuesFB - torch.reshape(
                    estimatedGVByRep.repeat_interleave(nRep), 
                    shape = (estimatedGVByRep.size(0), estimatedGVByRep.size(1), nRep))
                
                if not includeIntercept:
                    estimatedGVByRep = estimatedGVByRep - torch.reshape(
                        intercept.repeat_interleave(estimatedGVByRep.size(0)), 
                        shape = (estimatedGVByRep.size(1), 
                                 estimatedGVByRep.size(0))).T
                    
                VuByRep = estimatedGVByRep.var(0)
                VeByRep = torch.tensor([residualsByRep[:, i, :].var() for i in range(residualsByRep.size(1))], device = self.device)
                heritIndByRep = VuByRep / (VuByRep + VeByRep)
                heritLineByRep = VuByRep / (VuByRep + VeByRep / nRep)

                VarHeritByRep = torch.stack([VuByRep, VeByRep,
                                             heritIndByRep,
                                             heritLineByRep])

                estimatedEnvEffByRep = None
                lmerResList = None
        else:
            estimatedGVByRep = phenotypicValuesFB.mean(2)
            estimatedGVMethod = None
            lmerResList = None
            residualsByRep = None
            VarHeritByRep = None
            estimatedEnvEffByRep = None
        
        self.estimatedGVByRep = estimatedGVByRep
        self.estimatedGVMethod = estimatedGVMethod
        self.lmerResList = lmerResList
        self.residualsByRep = residualsByRep
        self.VarHeritByRep = VarHeritByRep
        self.estimatedEnvEffByRep = estimatedEnvEffByRep
        self.estimateGV = True
        self.includeIntercept = includeIntercept
    




class BreederInfo:
    def __init__(self,
                 bsInfo,
                 mrkPos,
                 breederName = "Undefined",
                 mrkNames = None,
                 initGenotyping = True,
                 initGenotypedIndNames = None,
                 mafThres = 0.05,
                 heteroThres = 1.,
                 # calculateGRMBase = True,
                 # methodsGRMBase = "addNOIA",
                 # calcEpistasisBase = False,
                 # multiTraitsAsEnvs = False,
                 includeIntercept = True,
                 traitNames = None,
                 device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
                 verbose = True):
        
        
        genoMapAll = bsInfo.genoMap
        genoMapFB = genoMapAll.iloc[mrkPos]
        nMarkers = len(mrkPos)
        
        if mrkNames is None:
            mrkNames = [f"Marker_{i}" for i in range(nMarkers)]
        
        # genoMapFB.iloc[:, 0] = mrkNames
        # genoMapFB.rename(columns = {'lociNames': 'mrkNames'}, inplace = True)
        if not isinstance(mafThres, torch.Tensor):
            mafThres = torch.tensor(mafThres, device = device)
        if not isinstance(heteroThres, torch.Tensor):
            heteroThres = torch.tensor(float(heteroThres), device = device)
        
        lociInfoFB = {
            "genoMapFB": genoMapFB,
            "mrkNames": mrkNames,
            "nMarkers": nMarkers,
            "mafThres": mafThres,
            "heteroThres": heteroThres
            }
        
        initPopulationFB = PopulationFB(
            population = bsInfo.populations[0], 
            mrkPos = mrkPos,
            device = device,
            genotyping = initGenotyping,
            genotypedIndNames = initGenotypedIndNames,
            mrkNames = mrkNames,
            mafThres = mafThres,
            heteroThres = heteroThres
            )
        populationsFB = [initPopulationFB]
        popNameBase = bsInfo.popNameBase
        
        nTraits = initPopulationFB.nTraits
        if traitNames is None:
            traitNames = [f"Trait_{traitNo:0{len(str(nTraits))}d}" for traitNo in range(1, nTraits + 1)]
        
        traitInfoFB = {
            "nTraits": nTraits,
            "traitNames": traitNames
            }

        
        self.bsInfo = bsInfo
        self.breederName = breederName
        self.initGenotyping = initGenotyping
        self.initGenotypedIndNames = initGenotypedIndNames
        self.mrkPos = mrkPos
        self.lociInfoFB = lociInfoFB
        self.traitInfoFB = traitInfoFB
        self.populationsFB = populationsFB
        self.crossInfoList = []
        self.popNameBase = popNameBase
        self.generation = initPopulationFB.generation
        self.includeIntercept = includeIntercept
        self.device = device
        self.verbose = verbose
        
        self.mrkEffMat = None
        self.estimatedGVByMLR = None
        

    def getNewPopulation(self, 
                         bsInfo,
                         generationNew = None,
                         genotyping = True,
                         genotypedIndNames = None):
        populationsFB = self.populationsFB
        crossInfoList = self.crossInfoList
        populations = bsInfo.populations
        generation = self.generation
        generations = [pop.generation for pop in populations]
        
        if generationNew is None:
            generationNew = generation + 1

        whichGen = [gen == generationNew for gen in generations]
        
        newPopulation = populations[torch.nonzero(torch.tensor(whichGen, device = self.device),
                                                  as_tuple = False).squeeze()]
        
        newPopulationFB = PopulationFB(
            population = newPopulation,
            mrkPos = self.mrkPos,
            device = self.device,
            genotyping = genotyping,
            genotypedIndNames = genotypedIndNames,
            mrkNames = self.lociInfoFB["mrkNames"],
            mafThres = self.lociInfoFB["mafThres"],
            heteroThres = self.lociInfoFB["heteroThres"]
            )
        
        crossInfoList.append(newPopulation.crossInfo)
        populationsFB.append(newPopulationFB)
        
        self.generation = generationNew
        self.populationsFB = populationsFB
        self.crossInfoList = crossInfoList


        
    def phenotyper(self,
                   bsInfo,
                   generationOfInterest = None,
                   nRep = 1,
                   phenotypedIndNames = None,
                   estimateGV = True,
                   estimatedGVMethod = "mean"):
        populationsFB = self.populationsFB
        
        if generationOfInterest is None:
            generationOfInterest = self.generation
        
        if nRep is None:
            nRep = 1
            
        generationsFB = [populationFB.generation for populationFB in populationsFB]
        whichGenFB = [gen == generationOfInterest for gen in generationsFB]
        currentPopulationFB = populationsFB[torch.nonzero(torch.tensor(whichGenFB, device = self.device),
                                                  as_tuple = False).squeeze()]
        
        currentPopulationFB.phenotyper(
            phenotypedIndNames = phenotypedIndNames,
            estimateGV = estimateGV,
            estimatedGVMethod = estimatedGVMethod,
            includeIntercept = self.includeIntercept
            )
        
        populationsFB[torch.nonzero(torch.tensor(whichGenFB, device = self.device),
                                                  as_tuple = False).squeeze()] = currentPopulationFB
        
        self.populationsFB = populationsFB

        
    def estimateMrkEff(self,
                       trainingPop = None,
                       trainingIndNames = None,
                       methodMLR = "Ridge",
                       multiTrait = False,
                       alpha = 0.5):
        populationsFB = self.populationsFB
        # generation = self.generation
        
        nTraits = self.traitInfoFB["nTraits"]
        
        if nTraits == 1:
            multiTrait = False
        
        if trainingPop is None:
            trainingPop = len(populationsFB) - 1
        
        if trainingIndNames is None:
            if isinstance(trainingPop, int):
                trainingIndNames = populationsFB[trainingPop].indNames
            else:
                trainingIndNames = flattenList([populationsFB[trainingPopNo].indNames 
                                                for trainingPopNo in trainingPop])
        if methodMLR == "Ridge":
            alpha = 0
        elif methodMLR == "LASSO":
            alpha = 1
        elif alpha is None:
            alpha = 0.5
        
        if isinstance(trainingPop, int):
            trainingPopFB = populationsFB[trainingPop]
        else:
            trainingPopFB = self.overGeneration(targetPop = trainingPop)
        
        trainingIndOrNot = [indName in trainingIndNames
                            for indName in trainingPopFB.indNames]
        genoMat = trainingPopFB.genoMatFB[torch.tensor(trainingIndOrNot, device = self.device), :]
        phenoMat = trainingPopFB.estimatedGVByRep[torch.tensor(trainingIndOrNot, device = self.device), :]
        
        if multiTrait:
            mrkEffMat = ElasticNetCV(
                X = genoMat, y = phenoMat, 
                alpha = alpha, nFolds = 5, 
                nCores = 0, seed = 42,
                nLambdas = 5, minMaxLambdaRatio = 0.0001, 
                nEpochs = 60, learningRate = 0.01,
                device = self.device,
                includeIntercept = self.includeIntercept)
        else:
            mrkEffMat = torch.zeros(size = (genoMat.size(1) + self.includeIntercept, 
                                            phenoMat.size(1)), device = self.device)
            for i in range(phenoMat.size(1)):
                mrkEffMat[:, i:(i + 1)] = ElasticNetCV(
                    X = genoMat, y = phenoMat[:, i:(i + 1)], 
                    alpha = alpha, nFolds = 5, 
                    nCores = 0, seed = 42,
                    nLambdas = 5, minMaxLambdaRatio = 0.0001, 
                    nEpochs = 60, learningRate = 0.01,
                    device = self.device,
                    includeIntercept = self.includeIntercept)
        
        mrkEffMatNoGrad = mrkEffMat.detach()
        
        self.mrkEffMat = mrkEffMatNoGrad

    
    def estimateGVByMLR(self,
                       trainingPop = None,
                       trainingIndNames = None,
                       testingPop = None,
                       methodMLR = "Ridge",
                       multiTrait = False,
                       update = False,
                       alpha = 0.5):
        populationsFB = self.populationsFB
        # generation = self.generation
        
        nTraits = self.traitInfoFB["nTraits"]
        
        if nTraits == 1:
            multiTrait = False
            
        if trainingPop is None:
            trainingPop = len(populationsFB) - 1
            
        if trainingIndNames is None:
            if isinstance(trainingPop, int):
                trainingIndNames = populationsFB[trainingPop].indNames
            else:
                trainingIndNames = flattenList([populationsFB[trainingPopNo].indNames 
                                                for trainingPopNo in trainingPop])

        
        if self.mrkEffMat is None or update:
            self.estimateMrkEff(self,
                               trainingPop = trainingPop,
                               trainingIndNames = trainingIndNames,
                               methodMLR = methodMLR,
                               multiTrait = multiTrait,
                               alpha = alpha)
        
        mrkEffMat = self.mrkEffMat
        
        if testingPop is None:
            testingPop = len(populationsFB) - 1
        
        if isinstance(testingPop, int):
            testingPopFB = populationsFB[testingPop]
        else:
            testingPopFB = self.overGeneration(targetPop = testingPop)
        
        genoMat = testingPopFB.genoMatFB
        if self.includeIntercept:
            genoMatIncInt = torch.cat(
                [torch.ones((genoMat.size(0), 1), device = self.device),
                genoMat],
                dim = 1
                )
        else:
            genoMatIncInt = genoMat
        estimatedGVByMLR = torch.mm(genoMatIncInt, mrkEffMat)
        
        
        self.estimatedGVByMLR = estimatedGVByMLR
        

    def removeLatestPop(self):
        populationsFB = self.populationsFB
        generation = self.generation
        crossInfoList = self.crossInfoList
        
        if len(populationsFB) == 1:
            warnings.warn("We cannot remove all the populations. Please remain at least 1 population or redefine the initial population with `self.__init__`!")
        else:
            populationsFB.remove(populationsFB[len(populationsFB)])
            crossInfoList.remove(crossInfoList[len(crossInfoList) - 1])
            generation = generation - 1

        self.generation = generation
        self.populationsFB = populationsFB
        self.crossInfoList = crossInfoList
        
    def removeInitialPop(self):
        populationsFB = self.populationsFB
        generation = self.generation
        crossInfoList = self.crossInfoList
        
        if len(populationsFB) == 1:
            warnings.warn("We cannot remove all the populations. Please remain at least 1 population or redefine the initial population with `self.__init__`!")
        else:
            populationsFB.remove(populationsFB[0])
            crossInfoList.remove(crossInfoList[0])
            generation = generation - 1
            
        for popNo in range(len(populationsFB)):
            populationsFB[popNo].generation = populationsFB[popNo].generation - 1

        self.generation = generation
        self.populationsFB = populationsFB
        self.crossInfoList = crossInfoList
    
    def overGeneration(self, targetPop = None):
        populationsFB = self.populationsFB
        
        if targetPop is None:
            targetPop = list(range(len(populationsFB)))
        

        targetPop = [targetPopNow for targetPopNow in targetPop if targetPopNow in list(range(len(populationsFB)))]
        targetPopulationsFB = [populationsFB[targetPopNow] for targetPopNow in targetPop]
        nTargetPop = len(targetPop)
        
        indsOverGeneration = []
        
        for i in range(nTargetPop):
            indsNow = targetPopulationsFB[i].population.indsList
            indsOverGeneration.append(indsNow)
        
        indsOverGeneration = flattenList(indsOverGeneration)

        popOGName = self.popNameBase+"_".join(list(map(str, targetPop)))
        lociEffectsTrue = populationsFB[0].population.lociEffectsTrue
        herit = populationsFB[0].population.herit
        nRep = populationsFB[0].population.nRep
        popOverGeneration = Population(
            indsList = indsOverGeneration,
            name = popOGName,
            generation = None,
            lociEffectsTrue = lociEffectsTrue,
            crossInfo = None,
            herit = herit,
            nRep = nRep,
            device = self.device,
            verbose = False
            )
    
    
        whichGenGenotyped = [popFB.genotyping for popFB in targetPopulationsFB]
        
        whichGenGenotypedNo = torch.nonzero(input = torch.tensor(whichGenGenotyped, device = self.device),
                                            as_tuple = False).squeeze()
        
        if sum(whichGenGenotyped) >= 2:
            genotypedIndNamesOG = flattenList([targetPopulationsFB[genNo].genotypedIndNames for genNo in whichGenGenotypedNo])
            genotypingOG = True
        elif sum(whichGenGenotyped) == 1:
            genotypedIndNamesOG = targetPopulationsFB[whichGenGenotypedNo].genotypedIndNames
            genotypingOG = True
        else:
            genotypedIndNamesOG = None
            genotypingOG = False
        
        popOverGenerationFB = PopulationFB(population = popOverGeneration,
                                           mrkPos = self.mrkPos,
                                           device = self.device,
                                           genotyping = genotypingOG,
                                           genotypedIndNames = genotypedIndNamesOG,
                                           mrkNames = self.lociInfoFB["mrkNames"],
                                           mafThres = self.lociInfoFB["mafThres"],
                                           heteroThres = self.lociInfoFB["heteroThres"]
                                           )
        
        return popOverGenerationFB
    
    def parentInd(self, indName):
        populationsFB = self.populationsFB
        whichPop = torch.nonzero(
            input = torch.tensor([
                indName in [
                    ind.name for ind in populationsFB[i].population.indsList
                    ] for i in range(len(populationsFB))
                ], device = self.device), as_tuple = False).squeeze()
        indNamesNow = [ind.name for ind in populationsFB[whichPop].population.indsList]
        whichInd = torch.nonzero(
            input = torch.tensor([indNameNow == indName for indNameNow in indNamesNow], device = self.device), as_tuple = False).squeeze()
        indInfo = populationsFB[whichPop].population.indsList[whichInd]
        
        parent1 = indInfo.parent1
        parent2 = indInfo.parent2
        
        return [parent1, parent2]
    
    
    def lociEffectsFunc(self,
                        bsInfo,
                        trainingPop = None,
                        trainingIndNames = None,
                        methodMLR = "Ridge",
                        multiTrait = False,
                        alpha = 0.5,
                        update = False):
        lociEffectsEst = torch.zeros(size = bsInfo.lociEffectsTrue.size(), device = self.device)
        
        if self.mrkEffMat is None or update:
            self.estimateMrkEff(trainingPop = trainingPop,
                                trainingIndNames = trainingIndNames,
                                methodMLR = methodMLR,
                                multiTrait = multiTrait,
                                alpha = alpha)
            
        lociEffectsEst[torch.cat([torch.tensor([0], device = self.device), self.mrkPos + 1], dim = 0), :] = self.mrkEffMat
        
        return lociEffectsEst
    
    
    def deepcopy(self):
        dictAll = self.__dict__
        
        breederInfoCopied = BreederInfo(
            bsInfo = self.bsInfo,
            mrkPos = self.mrkPos,
            breederName = self.breederName,
            mrkNames = self.lociInfoFB["mrkNames"],
            initGenotyping = self.initGenotyping,
            initGenotypedIndNames = self.initGenotypedIndNames,
            mafThres = self.lociInfoFB["mafThres"],
            heteroThres = self.lociInfoFB["heteroThres"],
            includeIntercept = self.includeIntercept,
            traitNames = self.traitInfoFB["traitNames"],
            device = self.device,
            verbose = False)
        
        for eachKey in dictAll:
            valueNow = dictAll[eachKey]
            
            if isinstance(valueNow, torch.Tensor):
                valueCopied = valueNow.clone()
            else:
                valueCopied = copy.deepcopy(valueNow)
                
            setattr(breederInfoCopied, 
                    eachKey, 
                    valueCopied)
        
        return breederInfoCopied
    
    def quickcopy(self):
        breederInfoCopied = BreederInfo(
            bsInfo = self.bsInfo,
            mrkPos = self.mrkPos,
            breederName = self.breederName,
            mrkNames = self.lociInfoFB["mrkNames"],
            initGenotyping = self.initGenotyping,
            initGenotypedIndNames = self.initGenotypedIndNames,
            mafThres = self.lociInfoFB["mafThres"],
            heteroThres = self.lociInfoFB["heteroThres"],
            includeIntercept = self.includeIntercept,
            traitNames = self.traitInfoFB["traitNames"],
            device = self.device,
            verbose = False)
        breederInfoCopied.mrkEffMat = self.mrkEffMat
       
        return breederInfoCopied


class SimBs:
    def __init__(self,
                 bsInfoInit,
                 mrkPos,
                 breederInfoInit = None,
                 simBsName = "Undefined",
                 lociEffMethod = "estimated",
                 includeIntercept = None,
                 trainingPopType = "latest",
                 trainingPopInit = None,
                 trainingIndNamesInit = None,
                 methodMLRInit = "Ridge",
                 multiTraitInit = False,
                 nIterSimulation = 100,
                 nGenerationProceed = 4,
                 nRefreshMemoryEvery = 2,
                 updateBreederInfo = True,
                 phenotypingInds = False,
                 nRepForPhenoInit = 3,
                 nRepForPheno = 3,
                 updateModels = False,
                 methodMLR = "Ridge",
                 multiTrait = False,
                 mrkNames = None,
                 traitNames = None,
                 selectionMethodVec = "BV",
                 multiTraitsEvalMethodVec = "sum",
                 matingMethodVec = "randomMate",
                 allocateMethodVec = "equalAllocation",
                 weightedAllocationMethodList = ["WBV"],
                 logHSelList = torch.tensor(0.),
                 logitHTensor = torch.tensor(0.),
                 lowerBound = torch.tensor(0.),
                 upperBound = torch.tensor(3.),
                 minimumUnitAllocateVec = 5,
                 clusteringForSelVec = True,
                 nClusterVec = 10,
                 nTopClusterVec = 5,
                 nTopEachVec = None,
                 nSelVec = None,
                 nNextPopVec = None,
                 includeGVPVec = False,
                 targetGenForGVPVec = 2,
                 nameMethod = "pairBase",
                 nCores = 1,
                 overWriteRes = False,
                 showProgress = True,
                 returnMethod = "all",
                 saveAllResAt = None,
                 evaluateGVMethod = "true",
                 requires_grad = False,
                 nTopEval = 5,
                 logHEval = torch.log(torch.tensor(0.5)),
                 summaryAllResAt = None,
                 device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
                 verbose = True):
        ### define some variables
        nIndNow = len(bsInfoInit.populations[len(bsInfoInit.populations) - 1].indsList)
        nTraits = bsInfoInit.lociEffectsTrue.size(1)
        nMarkers = len(mrkPos)
        
        
        ### updateBreederInfo
        if isinstance(updateBreederInfo, bool):
            updateBreederInfo = [updateBreederInfo] * nGenerationProceed
        
        ### phenotypingInds
        if isinstance(phenotypingInds, bool):
            phenotypingInds = [phenotypingInds] * nGenerationProceed
        
        ### nRepForPheno
        if isinstance(nRepForPheno, int):
            nRepForPheno = [nRepForPheno] * nGenerationProceed
        
        ### updateModels
        if isinstance(updateModels, bool):
            updateModels = [updateModels] * nGenerationProceed

        ### mrkNames
        if mrkNames is None:
            mrkNames = [f"Marker_{i}" for i in range(nMarkers)]

        ### traitNames
        if traitNames is None:
            traitNames = [f"Trait_{traitNo:0{len(str(nTraits))}d}" for traitNo in range(1, nTraits + 1)]

        ### breederInfoInit
        if includeIntercept is None:
            includeIntercept = bsInfoInit.includeIntercept
        
        if breederInfoInit is None:
            breederInfoInit = BreederInfo(
                breederName = "Undefined", 
                bsInfo = bsInfoInit, 
                mrkPos = mrkPos,
                mrkNames = mrkNames,
                traitNames = traitNames,
                initGenotyping = True,
                initGenotypedIndNames = None,
                mafThres = 0.05,
                heteroThres = 1,
                includeIntercept = includeIntercept,
                device = device,
                verbose = verbose
                )
            
        breederInfoInit.phenotyper(
            bsInfo = bsInfoInit,
            generationOfInterest = None,
            nRep = nRepForPhenoInit,
            phenotypedIndNames = None,
            estimateGV = True,
            estimatedGVMethod = "mean"
            )
        
        ### selectionMethodVec
        if isinstance(selectionMethodVec, str):
            selectionMethodVec = [selectionMethodVec] * nGenerationProceed

        ### clusteringForSelVec
        if isinstance(clusteringForSelVec, bool):
            clusteringForSelVec = [clusteringForSelVec] * nGenerationProceed

        ### nSelVec
        if nSelVec is None:
            nSelVec = nIndNow // 10
            
        if isinstance(nSelVec, torch.Tensor):
            if len(nSelVec.size()) == 0:
                nSelVec = nSelVec.unsqueeze(0)
            
            if len(nSelVec) == 1:
                nSelVec = nSelVec.repeat_interleave(nGenerationProceed)

        else:
            if isinstance(nSelVec, int):
                nSelVec = [nSelVec] * nGenerationProceed
            nSelVec = torch.tensor(nSelVec, device = device)


        ### nClusterVec
        if isinstance(nClusterVec, torch.Tensor):
            if len(nClusterVec.size()) == 0:
                nClusterVec = nClusterVec.unsqueeze(0)
            
            if len(nClusterVec) == 1:
                nClusterVec = nClusterVec.repeat_interleave(nGenerationProceed)

        else:
            if isinstance(nClusterVec, int):
                nClusterVec = [nClusterVec] * nGenerationProceed
            nClusterVec = torch.tensor(nClusterVec, device = device)

        ### nTopClusterVec
        if isinstance(nTopClusterVec, torch.Tensor):
            if len(nTopClusterVec.size()) == 0:
                nTopClusterVec = nTopClusterVec.unsqueeze(0)
            
            if len(nTopClusterVec) == 1:
                nTopClusterVec = nTopClusterVec.repeat_interleave(nGenerationProceed)
                
        else:
            if isinstance(nTopClusterVec, int):
                nTopClusterVec = [nTopClusterVec] * nGenerationProceed
            nTopClusterVec = torch.tensor(nTopClusterVec, device = device)

        ### nTopEachVec
        if nTopEachVec is None:
            nTopEachVec = [nSelVec[genNo] // nTopClusterVec[genNo] 
                            for genNo in range(nGenerationProceed)]
            
        if isinstance(nTopEachVec, torch.Tensor):
            if len(nTopEachVec.size()) == 0:
                nTopEachVec = nTopEachVec.unsqueeze(0)
            
            if len(nTopEachVec) == 1:
                nTopEachVec = nTopEachVec.repeat_interleave(nGenerationProceed)

        else:
            if isinstance(nTopEachVec, int):
                nTopEachVec = [nTopEachVec] * nGenerationProceed
            nTopEachVec = torch.tensor(nTopEachVec, device = device)
        
        ### multiTraitsEvalMethodVec
        if isinstance(multiTraitsEvalMethodVec, str):
            multiTraitsEvalMethodVec = [multiTraitsEvalMethodVec] * nGenerationProceed

        ### hSelList
        if not isinstance(logHSelList, list):
            if not isinstance(logHSelList, torch.Tensor):
                logHSelList = torch.tensor(float(logHSelList))
            
            if len(logHSelList.size()) == 0:
                hSelList = torch.exp(logHSelList)
                hSelList = hSelList.repeat_interleave(nTraits)
            
            hSelList = hSelList.to(device)
            
            hSelList = [hSelList] * nGenerationProceed
        else:
            hSelList = [torch.exp(logHSel) for logHSel in logHSelList]

        ### matingMethodVec
        if isinstance(matingMethodVec, str):
            matingMethodVec = [matingMethodVec] * nGenerationProceed

        ### allocateMethodVec
        if isinstance(allocateMethodVec, str):
            allocateMethodVec = [allocateMethodVec] * nGenerationProceed
            
        ### weightedAllocationMethodList
        if not isinstance(weightedAllocationMethodList, list):
            weightedAllocationMethodList = [[weightedAllocationMethodList]] * nGenerationProceed
        else:
            if not ((len(weightedAllocationMethodList) == nGenerationProceed) & isinstance(weightedAllocationMethodList[0], list)):
                weightedAllocationMethodList = [weightedAllocationMethodList] * nGenerationProceed
        
        ### includeGVPVec
        if isinstance(includeGVPVec, bool):
            includeGVPVec = [includeGVPVec] * nGenerationProceed

        ### targetGenForGVPVec
        if isinstance(targetGenForGVPVec, int):
            targetGenForGVPVec = [targetGenForGVPVec] * nGenerationProceed
    
    
        ### hLens
        hLens = []
        
        for generationProceedNo in range(nGenerationProceed):
            hLenNow = nTraits * (len(weightedAllocationMethodList[generationProceedNo]) + 
                                 int(includeGVPVec[generationProceedNo]))
            hLens.append(hLenNow)
        
        
        
        ### lowerBound
        if not isinstance(lowerBound, torch.Tensor):
            if isinstance(lowerBound, int) or isinstance(lowerBound, float):
                lowerBound = torch.tensor(float(lowerBound))
        
        
        lowerBound = lowerBound.to(device)
        lowerBoundSize = lowerBound.size()
        if len(lowerBoundSize) == 0:
            lowerBound = torch.cat([lowerBound.repeat_interleave(hLenNow).unsqueeze(0).to(device) 
                                    for hLenNow in hLens],
                                   dim = 0)
        elif len(lowerBoundSize) == 1:
            lowerBound = torch.cat([lowerBound.unsqueeze(0).to(device) for genNo in range(nGenerationProceed)],
                                   dim = 0)
            
        ### upperBound
        if not isinstance(upperBound, torch.Tensor):
            if isinstance(upperBound, int) or isinstance(upperBound, float):
                upperBound = torch.tensor(float(upperBound))
        
        
        upperBound = upperBound.to(device)
        upperBoundSize = upperBound.size()
        if len(upperBoundSize) == 0:
            upperBound = torch.cat([upperBound.repeat_interleave(hLenNow).unsqueeze(0).to(device) 
                                    for hLenNow in hLens],
                                   dim = 0)
        elif len(upperBoundSize) == 1:
            upperBound = torch.cat([upperBound.unsqueeze(0).to(device) for genNo in range(nGenerationProceed)],
                                   dim = 0)
                     
            
        
        
        ### hTensor
        if not isinstance(logitHTensor, torch.Tensor):
            if isinstance(logitHTensor, int) or isinstance(logitHTensor, float):
                logitMinusExpHTensor = torch.exp(-torch.tensor(float(logitHTensor)))
        else:
            logitMinusExpHTensor = torch.exp(-logitHTensor)
        
        hTensor = lowerBound + (upperBound - lowerBound) / (1. + logitMinusExpHTensor)
        # hTensor[torch.isinf(logitExpHTensor)] = upperBound[torch.isinf(logitExpHTensor)]
        
        hTensor = hTensor.to(device)
        hSize = hTensor.size()
        if len(hSize) == 0:
            hTensor = torch.cat([hTensor.repeat_interleave(hLenNow).unsqueeze(0).to(device) 
                                 for hLenNow in hLens],
                                dim = 0)
        elif len(hSize) == 1:
            hTensor = torch.cat([hTensor.unsqueeze(0).to(device) for genNo in range(nGenerationProceed)],
                                dim = 0)



        ### minimumUnitAllocateVec
        if isinstance(minimumUnitAllocateVec, int):
            minimumUnitAllocateVec = [minimumUnitAllocateVec] * nGenerationProceed

        ### nNextPopVec
        if nNextPopVec is None:
            nNextPopVec = nIndNow
            
        if isinstance(nNextPopVec, int):
            nNextPopVec = [nNextPopVec] * nGenerationProceed


        ### hEval
        if not isinstance(logHEval, torch.Tensor):
            hEval = torch.exp(torch.tensor(float(logHEval)))
        else:
            hEval = torch.exp(logHEval)
            
        if len(hEval.size()) == 0:
            hEval = hEval.repeat_interleave(nTraits)
        
        hEval = hEval.to(device)


        
        ### trainingPopInit
        estimatedGVInitExist = breederInfoInit.estimatedGVByMLR is not None
        
        if not estimatedGVInitExist:
            if trainingPopInit is None:
                if trainingPopType == "all":
                    trainingPopInit = list(range(len(breederInfoInit.populationsFB)))
                else:
                    trainingPopInit = len(breederInfoInit.populationsFB) - 1

        ### popNames
        popNames = [f"{bsInfoInit.popNameBase}{genNo:0{len(str(nGenerationProceed))}d}" 
                    for genNo in range(nGenerationProceed + 1)]
        
        ###iterNames
        iterNames = [f"Iteration_{iterNo:0{len(str(nIterSimulation))}d}" 
                     for iterNo in range(nIterSimulation)]
        
    
        ### Save arguments in `self`
        self.bsInfoInit = bsInfoInit
        self.mrkPos = mrkPos
        self.breederInfoInit = breederInfoInit
        self.simBsName = simBsName
        self.lociEffMethod = lociEffMethod
        self.includeIntercept = includeIntercept
        self.trainingPopType = trainingPopType
        self.trainingPopInit = trainingPopInit
        self.trainingIndNamesInit = trainingIndNamesInit
        self.methodMLRInit = methodMLRInit
        self.multiTraitInit = multiTraitInit
        self.nIterSimulation = nIterSimulation
        self.nGenerationProceed = nGenerationProceed
        self.nRefreshMemoryEvery = nRefreshMemoryEvery
        self.updateBreederInfo = updateBreederInfo
        self.phenotypingInds = phenotypingInds
        self.nRepForPhenoInit = nRepForPhenoInit
        self.nRepForPheno = nRepForPheno
        self.updateModels = updateModels
        self.methodMLR = methodMLR
        self.multiTrait = multiTrait
        self.mrkNames = mrkNames
        self.traitNames = traitNames
        self.selectionMethodVec = selectionMethodVec
        self.multiTraitsEvalMethodVec = multiTraitsEvalMethodVec
        self.matingMethodVec = matingMethodVec
        self.allocateMethodVec = allocateMethodVec
        self.weightedAllocationMethodList = weightedAllocationMethodList
        self.hSelList = hSelList
        self.hTensor = hTensor
        self.hLens = hLens
        self.minimumUnitAllocateVec = minimumUnitAllocateVec
        self.clusteringForSelVec = clusteringForSelVec
        self.nClusterVec = nClusterVec
        self.nTopClusterVec = nTopClusterVec
        self.nTopEachVec = nTopEachVec
        self.nSelVec = nSelVec
        self.nNextPopVec = nNextPopVec
        self.includeGVPVec = includeGVPVec
        self.targetGenForGVPVec = targetGenForGVPVec
        self.nameMethod = nameMethod
        self.nCores = nCores
        self.overWriteRes = overWriteRes
        self.showProgress = showProgress
        self.returnMethod = returnMethod
        self.saveAllResAt = saveAllResAt
        self.evaluateGVMethod = evaluateGVMethod
        self.requires_grad = requires_grad
        self.nTopEval = nTopEval
        self.hEval = hEval
        self.summaryAllResAt = summaryAllResAt
        self.device = device
        self.verbose = verbose
        
        self.popNames = popNames
        self.summaryNames = ["max", "mean", "median",
                             "min", "var"]
        self.iterNames = iterNames
        
        ### trueGVMatList
        populationInit = bsInfoInit.populations[len(bsInfoInit.populations) - 1]
        populationInit.obtainGenoMat()
        trueGVMatInit = populationInit.trueGVMat
        trueGVMatInitList = [[trueGVMatInit]] * nIterSimulation
        
        self.trueGVMatInit = trueGVMatInit
        self.trueGVMatList = trueGVMatInitList
        self.trueGVSummaryArray = None
        self.estimatedGVSummaryArray = None
        
        ### estimatedGVMatInit
        # if breederInfoInit.estimatedGVByMLR is None:
        #     breederInfoInit.estimateGVByMLR(
        #         trainingPop = trainingPopInit,
        #         trainingIndNames = trainingIndNamesInit,
        #         testingPop = trainingPopInit,
        #         methodMLR = methodMLRInit,
        #         multiTrait = multiTraitInit,
        #         update = False,
        #         alpha = 0.5
        #         )
        # estimatedGVMatInit = breederInfoInit.estimatedGVByMLR
        
        self.computeLociEffInit()
        lociEffects = self.lociEffectsInit
        genoMatNow = populationInit.genoMat
        genoMatIncIntNow = torch.cat(
            [torch.ones((genoMatNow.size(0), 1), device = self.device),
             genoMatNow],
            dim = 1
        )
        estimatedGVMatInit = torch.mm(genoMatIncIntNow, lociEffects)
        estimatedGVMatInitList = [[estimatedGVMatInit]] * nIterSimulation
        
        self.estimatedGVMatInit = estimatedGVMatInit
        self.estimatedGVMatList = estimatedGVMatInitList

        self.simResAll = [] 
    
    def computeLociEffInit(self):
        bsInfoInit = self.bsInfoInit
        mrkPos = self.mrkPos
        breederInfoInit = self.breederInfoInit
        lociEffMethod = self.lociEffMethod
        trainingPopType = self.trainingPopType
        trainingPopInit = self.trainingPopInit
        trainingIndNamesInit = self.trainingIndNamesInit
        methodMLRInit = self.methodMLRInit
        multiTraitInit = self.multiTraitInit
        verbose = self.verbose
        
        
        if lociEffMethod == "true":
            lociEffectsInit = bsInfoInit.lociEffectsTrue
        else:
            if breederInfoInit.mrkEffMat is None:
                breederInfoInit.estimateMrkEff(trainingPop = trainingPopInit,
                                               trainingIndNames = trainingIndNamesInit,
                                               methodMLR = methodMLRInit,
                                               multiTrait = multiTraitInit,
                                               alpha = 0.5)
            
            lociEffectsInit = breederInfoInit.lociEffectsFunc(
                bsInfo = bsInfoInit,
                trainingPop = trainingPopInit,
                trainingIndNames = trainingIndNamesInit,
                methodMLR = methodMLRInit,
                multiTrait = multiTraitInit,
                alpha = 0.5,
                update = False
                )
            
        self.lociEffectsInit = lociEffectsInit
    
    
    def startSimulation(self):
        ### Read arguments from `self`
        nIterSimulation = self.nIterSimulation
        # nCores = self.nCores
        
        # if nCores == 1:
        for iterNo in range(nIterSimulation):
            simRes = self.performOneSimulationTryError(iterNo)
            self.simResAll.append(simRes)
            self.trueGVMatList[iterNo] = simRes["trueGVMatList"]
            self.estimatedGVMatList[iterNo] = simRes["estimatedGVMatList"]
        
        # else:
        #     p = get_context("fork").Pool(nCores) 
        #     simResAll = p.map(self.performOneSimulationTryError, 
        #                        range(nIterSimulation))
        #     self.simResAll = simResAll
            
        #     for iterNo in range(nIterSimulation):
        #         simRes = simResAll[iterNo]
        #         self.trueGVMatList[iterNo] = simRes["trueGVMatList"]
        #         self.estimatedGVMatList[iterNo] = simRes["estimatedGVMatList"]
            
    
    def summaryResults(self):
        # Read arguments from `self`
        trueGVMatList = self.trueGVMatList
        estimatedGVMatList = self.estimatedGVMatList
        
        trueGVSummaryArray = torch.stack(
            [self.extractSummaryRes(trueGVMatListEach) 
             for trueGVMatListEach in trueGVMatList],
            dim = 3
        )
        estimatedGVSummaryArray = torch.stack(
            [self.extractSummaryRes(estimatedGVMatListEach) 
             for estimatedGVMatListEach in estimatedGVMatList],
            dim = 3
        )
        
        self.trueGVSummaryArray = trueGVSummaryArray
        self.estimatedGVSummaryArray = estimatedGVSummaryArray

    def scatter(self,
                iterNo = 0,
                targetTrait = None,
                targetPop = None,
                plotGVMethod = "true"   # plotGVMethodOffered = ["true", "estimated"]
                ):
        if plotGVMethod == "true":
            gvMatList = self.trueGVMatList[iterNo]
        else:
            gvMatList = self.trueGVMatList[iterNo]
        
        if targetTrait is None:
            targetTrait = torch.arange(2)
        
        if len(targetTrait) >= 2:
            targetTrait = targetTrait[:2]
        targetTraitNames = [self.traitNames[targetTraitNo] for targetTraitNo in targetTrait]
        targetTraitName = " v.s. ".join(targetTraitNames)
        
        if targetPop is None:
            targetPop = torch.arange(len(gvMatList))

        fig = px.scatter()
        
        for popNo in targetPop:
            fig.add_trace(
                go.Scatter(x = gvMatList[popNo][:, 0].detach(),
                           y = gvMatList[popNo][:, 1].detach(),
                           mode = "markers",
                           name = self.popNames[popNo]
                           ))
        
        fig.update_layout(
            title = dict(text = targetTraitName),
            yaxis = dict(title = dict(text = f"{plotGVMethod} GV"))
        )
            
        plot(fig, "temp-plot.html")
        
        self.fig = fig
            

    def plot(self,
             targetTrait = 0,
             targetPop = None,
             plotType = "box",    # plotTypeOffered = ["box", "violin", "lines"]
             # plotTargetDensity = "max",
             plotGVMethod = "true"   # plotGVMethodOffered = ["true", "estimated"]
             ):
        

        if isinstance(targetTrait, int):
            targetTraitName = self.traitNames[targetTrait]
        elif isinstance(targetTrait, str):
            targetTraitName = targetTrait
        else:
            print("`targetTraitName` must be `numeric` or `character`!")
        


        if plotGVMethod is None:
            plotGVMethod = "true"

        if plotGVMethod == "true":
            if self.trueGVSummaryArray is None:
                self.summaryResults()
            trueGVSummaryArray = self.trueGVSummaryArray
        else:
            if self.estimatedGVSummaryArray is None:
                self.summaryResults()
            trueGVSummaryArray = self.estimatedGVSummaryArray

        if targetPop is None:
            targetPop = torch.arange(trueGVSummaryArray.size(2))

        if isinstance(targetPop, int):
            if targetPop != 0:
                trueGVSummaryArray = trueGVSummaryArray[:, :, (targetPop - 1):targetPop, :]
            else:
                trueGVSummaryArray = trueGVSummaryArray[:, :, :targetPop, :]
        else:
            trueGVSummaryArray = trueGVSummaryArray[:, :, targetPop, :]
        
        dimSummary = trueGVSummaryArray.shape

        trueGVSummaryDf = pd.DataFrame({
            'SummaryStatistics': np.repeat(self.summaryNames, np.prod(dimSummary[1:4])),
            'Trait': np.repeat(np.tile(self.traitNames, dimSummary[0]), np.prod(dimSummary[2:4])),
            'Population': np.repeat(np.tile(self.popNames, np.prod(dimSummary[0:2])), dimSummary[3]),
            'Iteration': np.tile(self.iterNames, np.prod(dimSummary[0:3])),
            'Value': trueGVSummaryArray.flatten()
        })


        if plotType in ["box", "violin"]:
            trueGVSummaryDfTarget = trueGVSummaryDf[trueGVSummaryDf['Trait'] == targetTraitName]
            
            if plotType == "box":
                fig = px.box(trueGVSummaryDfTarget, 
                             x = "Population", 
                             y = "Value", 
                             color = "SummaryStatistics")
                fig.update_traces(quartilemethod = "exclusive") # or "inclusive", or "linear" by default
            elif plotType == "violin":
                fig = px.violin(trueGVSummaryDfTarget, 
                                x = "Population", 
                                y = "Value", 
                                color = "SummaryStatistics")
                
            fig.update_layout(
                title = dict(text = targetTraitName),
                yaxis = dict(title = dict(text = f"{plotGVMethod} GV"))
            )
            plot(fig, "temp-plot.html")


        elif plotType[0] == "lines":
            trueGVSummaryMeanArray = torch.mean(trueGVSummaryArray, axis = 3)

            dimSummaryMean = trueGVSummaryMeanArray.shape

            trueGVSummaryMeanDf = pd.DataFrame({
                'SummaryStatistics': np.repeat(self.summaryNames, np.prod(dimSummaryMean[1:3])),
                'Trait': np.repeat(np.tile(self.traitNames, dimSummaryMean[0]), dimSummaryMean[2]),
                'Population': np.tile(self.popNames, np.prod(dimSummaryMean[0:2])),
                'Value': trueGVSummaryMeanArray.flatten()
            })

            trueGVSummaryMeanDfTarget = trueGVSummaryMeanDf[trueGVSummaryMeanDf['Trait'] == targetTraitName]
            
            fig = px.line(trueGVSummaryMeanDfTarget, 
                          x = "Population", 
                          y = "Value", 
                          color = "SummaryStatistics")
            fig.update_layout(
                title = dict(text = targetTraitName),
                yaxis = dict(title = dict(text = f"{plotGVMethod} GV"))
            )
            
            plot(fig, "temp-plot.html")
        
        self.fig = fig
    
    def performOneSimulation(self, iterNo):
        ### Read arguments from `self`
        bsInfoInit = self.bsInfoInit
        breederInfoInit = self.breederInfoInit
        nGenerationProceed = self.nGenerationProceed
        updateBreederInfo = self.updateBreederInfo
        phenotypingInds = self.phenotypingInds
        nRepForPheno = self.nRepForPheno
        updateModels = self.updateModels
        selectionMethodVec = self.selectionMethodVec
        multiTraitsEvalMethodVec = self.multiTraitsEvalMethodVec
        matingMethodVec = self.matingMethodVec
        allocateMethodVec = self.allocateMethodVec
        weightedAllocationMethodList = self.weightedAllocationMethodList
        hSelList = self.hSelList
        hTensor = self.hTensor
        minimumUnitAllocateVec = self.minimumUnitAllocateVec
        clusteringForSelVec = self.clusteringForSelVec
        nClusterVec = self.nClusterVec
        nTopClusterVec = self.nTopClusterVec
        nTopEachVec = self.nTopEachVec
        nSelVec = self.nSelVec
        nNextPopVec = self.nNextPopVec
        includeGVPVec = self.includeGVPVec
        targetGenForGVPVec = self.targetGenForGVPVec
        returnMethod = self.returnMethod
        evaluateGVMethod = self.evaluateGVMethod
        requires_grad = self.requires_grad
        nTopEval = self.nTopEval
        hEval = self.hEval
        includeIntercept = self.includeIntercept
        
        trueGVMatInit = self.trueGVMatInit
        estimatedGVMatInit = self.estimatedGVMatInit
        
        lociEffectsInit = self.lociEffectsInit
        
        simRes = {"trueGVMatList": [],
                  "estimatedGVMatList": [],
                  "all": None,
                  "max": None,
                  "mean": None,
                  "median": None,
                  "min": None,
                  "var": None}
        # bsInfo = bsInfoInit.deepcopy()
        # bsInfo = copy.deepcopy(bsInfoInit)
        bsInfo = bsInfoInit.quickcopy()
        # breederInfo = breederInfoInit.deepcopy()
        # breederInfo = copy.deepcopy(breederInfoInit)
        breederInfo = breederInfoInit.quickcopy()
        breederInfo.bsInfo = bsInfo
        lociEffects = lociEffectsInit.clone()
        
        # estimatedGVMatList[iterNo]
        simRes["trueGVMatList"].append(trueGVMatInit)
        simRes["estimatedGVMatList"].append(estimatedGVMatInit)
        
        
        for genProceedNo in range(nGenerationProceed):
            # print(genProceedNo)
            crossInfoNow = CrossInfo(
                parentPopulation = bsInfo.populations[len(bsInfo.populations) - 1],
                lociEffectsEst = lociEffects,
                selectionMethod = selectionMethodVec[genProceedNo],
                multiTraitsEvalMethod = multiTraitsEvalMethodVec[genProceedNo],
                matingMethod = matingMethodVec[genProceedNo],
                allocateMethod = allocateMethodVec[genProceedNo],
                weightedAllocationMethod = weightedAllocationMethodList[genProceedNo],
                hSel = hSelList[genProceedNo].to(self.device),
                h = hTensor[genProceedNo, :],
                minimumUnitAllocate = minimumUnitAllocateVec[genProceedNo],
                clusteringForSel = clusteringForSelVec[genProceedNo],
                nCluster = nClusterVec[genProceedNo],
                nTopCluster = nTopClusterVec[genProceedNo],
                nTopEach = nTopEachVec[genProceedNo],
                nSel = nSelVec[genProceedNo],
                userSC = None,
                nProgenies = None,
                nNextPop = nNextPopVec[genProceedNo],
                includeGVP = includeGVPVec[genProceedNo],
                targetGenForGVP = targetGenForGVPVec[genProceedNo],
                indNames = None,
                seedSimRM = None,
                seedSimMC = None,
                requires_grad = requires_grad,
                includeIntercept = includeIntercept,
                device = self.device,
                name = f"CrossInfo-{genProceedNo}"
                )
            
            bsInfo.nextGeneration(crossInfo = crossInfoNow)
            
            if updateBreederInfo[genProceedNo]:
                breederInfo.getNewPopulation(
                    bsInfo = bsInfo,
                    generationNew = None,
                    genotyping = True,
                    genotypedIndNames = None
                    )
                
                if phenotypingInds[genProceedNo]:
                    breederInfo.phenotyper(
                        bsInfo = bsInfo,
                        generationOfInterest = None,
                        nRep = nRepForPheno[genProceedNo],
                        phenotypedIndNames = None,
                        estimateGV = True,
                        estimatedGVMethod = "mean"
                        )
                    
                    if updateModels[genProceedNo]:
                        lociEffects = breederInfo.lociEffectsFunc(
                            bsInfo = bsInfo,
                            breederInfo = breederInfo,
                            update = updateModels[genProceedNo]
                            )
        
            returnMethodSummary = ["all", "summary"]
            if any([returnMethodSummaryEach in returnMethod 
                    for returnMethodSummaryEach in returnMethodSummary]):
                popNow = bsInfo.populations[len(bsInfo.populations) - 1]
                trueGVMat = popNow.trueGVMat
                simRes["trueGVMatList"].append(trueGVMat)
                
                genoMatNow = popNow.genoMat
                if includeIntercept:
                    genoMatIncIntNow = torch.cat(
                        (torch.ones((genoMatNow.size(0), 1), device = self.device),
                         genoMatNow), dim = 1
                        )
                else:
                    genoMatIncIntNow = genoMatNow
                
                estimatedGVMat = torch.mm(genoMatIncIntNow, lociEffects)
                simRes["estimatedGVMatList"].append(estimatedGVMat)
        
        
        if "all" in returnMethod:
            simRes["all"] = [bsInfo, breederInfo]
        
        if any([returnMethodSummaryEach in returnMethod 
                for returnMethodSummaryEach in returnMethodSummary]):
            trueGVMat = simRes["trueGVMatList"][len(simRes["trueGVMatList"]) - 1]
            estimatedGVMat = simRes["estimatedGVMatList"][len(simRes["estimatedGVMatList"]) - 1]
            
        if evaluateGVMethod == "true":
            evalGVMat = trueGVMat
        else:
            evalGVMat = estimatedGVMat
        
        
        evalGVMatScaled = torch.zeros(size = evalGVMat.size(), device = self.device)
        for traitNo in range(evalGVMat.size(1)):
            trueGVMean = trueGVMatInit[:, traitNo].mean()
            trueGVSd = trueGVMatInit[:, traitNo].std()
            
            evalGVCentered = evalGVMat[:, traitNo] - trueGVMean
            
            if trueGVSd != 0:
                evalGVScaled = evalGVCentered / trueGVSd
            else:
                evalGVScaled = evalGVCentered
            
            evalGVMatScaled[:, traitNo] = evalGVScaled
            
        evalVals = torch.mm(evalGVMatScaled, hEval.unsqueeze(1)).squeeze(1)
        
        simRes["max"] = evalVals.sort(descending = True).values[0:nTopEval].mean(0)
        simRes["mean"] = evalVals.mean(0)
        simRes["median"] = evalVals.median(0).values
        simRes["min"] = evalVals.sort(descending = False).values[0:nTopEval].mean(0)
        simRes["var"] = evalVals.var(0)
        
        # del bsInfo
        # del breederInfo
        # gc.collect()
        
        return simRes
                        
           
    def performOneSimulationTryError(self, iterNo):
        continueLoop = True
        
        while continueLoop:
            try:
                simRes = self.performOneSimulation(iterNo)
            except:
                continueLoop = True
            else:
                continueLoop = False
        
        return simRes
    
    
    # def extractGVMatList(self, iterNo):
    #     # Read arguments from `self`
    #     bsInfoInit = self.bsInfoInit
    #     showProgress = self.showProgress
    #     evaluateGVMethod = self.evaluateGVMethod
    #     nTopEval = self.nTopEval
    #     simResAll = self.simResAll
    #     summaryAllResAt = self.summaryAllResAt
    #     nIterSimulation = self.nIterSimulation
    #     lociEffectsInit = self.lociEffectsInit

    
    def extractSummaryRes(self, gvMatListEach):
        nTopEval = self.nTopEval
        gvSummaryArrayEach = torch.zeros(size = (5, gvMatListEach[0].size(1),
                                                     len(gvMatListEach)), device = self.device)
        
        for genNo in range(len(gvMatListEach)):
            gvMatEachPop = gvMatListEach[genNo].detach()
            
            for traitNo in range(gvMatEachPop.size(1)):
                gvMatEachPopEachTrait = gvMatEachPop[:, traitNo]
                
                gvSummaryArrayEach[0, traitNo, genNo] = gvMatEachPopEachTrait.sort(descending = True).values[0:nTopEval].mean(0)
                gvSummaryArrayEach[1, traitNo, genNo] = gvMatEachPopEachTrait.mean(0)
                gvSummaryArrayEach[2, traitNo, genNo] = gvMatEachPopEachTrait.median(0).values
                gvSummaryArrayEach[3, traitNo, genNo] = gvMatEachPopEachTrait.sort(descending = False).values[0:nTopEval].mean(0)
                gvSummaryArrayEach[4, traitNo, genNo] = gvMatEachPopEachTrait.var(0)
                
        return gvSummaryArrayEach
                        
    
    def lociEffectsFunc(self, 
                        bsInfo, 
                        breederInfo,
                        update = True):
        
        lociEffMethod = self.lociEffMethod
        trainingPopType = self.trainingPopType
        methodMLR = self.methodMLR
        multiTrait = self.multiTrait
        
        
        if lociEffMethod == "true":
            lociEffects = bsInfo.lociEffectsTrue
            
        elif lociEffMethod == "estimated":
            if trainingPopType == "all":
                trainingPop = list(range(len(breederInfo.populationsFB)))
            else:
                trainingPop = len(breederInfo.populationsFB) - 1
            
            if breederInfo.mrkEffMat is None or update:
                breederInfo.estimateMrkEff(
                    trainingPop = trainingPop,
                    trainingIndNames = None,
                    methodMLR = methodMLR,
                    multiTrait = multiTrait,
                    alpha = 0.5
                    )
                
            lociEffects = breederInfo.lociEffectsFunc(
                bsInfo = bsInfo,
                trainingPop = trainingPop,
                trainingIndNames = None,
                methodMLR = methodMLR,
                multiTrait = multiTrait,
                alpha = 0.5,
                update = False
                )

        
        return lociEffects
                
