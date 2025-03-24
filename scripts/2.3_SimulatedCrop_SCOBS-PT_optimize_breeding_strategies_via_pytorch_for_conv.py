#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:59:08 2024

@author: hamazaki
"""

###### 0. Settings ######
##### 0.1. Load modules #####
### Import open modules
# import pyper
print("Load modules")
import time
import torch.optim as optim
import torch
# import random
# import numpy as np
# import pandas as pd
import joblib
import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import linkage, fcluster
# from itertools import combinations
import os
import sys
import gc
import psutil

process = psutil.Process()
print(process.memory_info().rss / 1024 ** 2, "[MB]")



### Set working directory and import local module, BSPT
dirSCOBSPT = "/Users/hamazaki/GitHub/SCOBS-PT/"
os.chdir(dirSCOBSPT)
fromDiffBreedimport BSPT



### Selection of computation resource (CPU/GPU)
print("Set arguments")
useGpu = False
if useGpu:
    if sys.platform == "darwin":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif sys.platform == "linux":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


### Scenario No, Trial No, ...
progName = "SCOBS-PT"
scenarioNo = 1
trialNo = 1
simGenoNo = 1
simPhenoNo = 1
trainingPopName = "true"

scriptIDData = "0.1"
scriptID = "2.3"

### Directory Names for input
dirMid = f"{dirSCOBSPT}midstream/"
dirScenario = f"{dirMid}{scriptIDData}_Scenario_{scenarioNo}/"
dirTrial = f"{dirScenario}{scriptIDData}_{progName}_Trial_{trialNo}/"
dirSimGeno = f"{dirTrial}{scriptIDData}_Geno_{simGenoNo}/"
dirSimPheno = f"{dirSimGeno}{scriptIDData}_Pheno_{simPhenoNo}/"
dirTrainingPop = f"{dirSimPheno}{scriptIDData}_Model_with_{trainingPopName}/"

if not os.path.isdir(dirTrainingPop):
    os.mkdir(path = dirTrainingPop)

dirTrainingPop250 = f"{dirSimPheno}{scriptIDData}_Model_with_250/"

dirPyTorch = f"{dirTrainingPop250}PyTorch/"



### Parameter Settings for breeding programs and simulations
simTypeName = "RecurrentGS"

nGenerationProceed = 4      # int(args[1])
nIterSimulation = 100       # int(args[2])
nIterSimulationUnit = 25

selectTypeName = "SI2"      # str(args[3]): "SI1"; "SI2"; "SI3"; "SI4"
selCriTypeName = "WBV+GVP"  # str(args[4]): "WBV"; "BV+WBV"; "WBV+GVP"; "BV+WBV+GVP"


### Parameter Settings for optimization  
optimMethod = "SGD"         # str(args[5]): "SGD", "Adam", "Adadelta"
useScheduler = True         # bool(args[6])

learningRate = 0.15         # float(args[7])
nIterOptimization = 200     # int(args[8])
gvpInit = "Zero"            # str(args[9]): "Mean", "Zero"
betas = [0.1, 0.1]




### Parameter Settings for others
nTraits = 1
traitNo = 1
traitName = f"Trait_{traitNo}"
gpTypeName = "TrueMrkEff"    # "BRRSet"; "BayesBSet"; "TrueMrkEff"


### Simulated breeding scheme name
simBsNameBaseWoAllNIter = f"{simTypeName}-{selectTypeName}-{selCriTypeName}-{optimMethod}_{learningRate}-Sche_{useScheduler}-{nGenerationProceed}_Gen-InitGVP_{gvpInit}"
simBsNameBaseWoNIterOpt = f"{simBsNameBaseWoAllNIter}-{nIterSimulation}_BS"
simBsNameBase = f"{simBsNameBaseWoNIterOpt}-{nIterOptimization}_Eval-for-{traitName}-by-{gpTypeName}"


### Directory Names for output
dirSimBs = f"{dirTrainingPop}{scriptID}_{simBsNameBase}/"

if not os.path.isdir(dirSimBs):
    os.mkdir(path = dirSimBs)

dirSimBsMid = f"{dirSimBs}{scriptID}_current_results/"

if not os.path.isdir(dirSimBsMid):
    os.mkdir(path = dirSimBsMid)


### Parameter Settings for data simulation
nChr = 10 
herit = 0.7



### Dome settings related to breeding strategies
if selectTypeName == "SI1":
    nCluster = 1
    nTopCluster = 15
    nSel = 15
elif selectTypeName == "SI2":
    nCluster = 25 
    nTopCluster = 1
    nSel = 25
elif selectTypeName == "SI3":
    nCluster = 10 
    nTopCluster = 5
    nSel = 50
elif selectTypeName == "SI4":
    nCluster = 1
    nTopCluster = 50
    nSel = 50


if selCriTypeName == "WBV":
    weightedAllocationMethodList = ["WBV"]
    includeGVP = False
elif selCriTypeName == "BV+WBV":
    weightedAllocationMethodList = ["BV", "WBV"]
    includeGVP = False
elif selCriTypeName == "WBV+GVP":
    weightedAllocationMethodList = ["WBV"]
    includeGVP = True
elif selCriTypeName == "BV+WBV+GVP":
    weightedAllocationMethodList = ["BV", "WBV"]
    includeGVP = True


lowerBoundEach = - 0.5
upperBoundEach = 3.5
if includeGVP:
    if gvpInit == "Zero":
        logitHInit = [0., - torch.log(torch.tensor(- (upperBoundEach - lowerBoundEach) / lowerBoundEach - 1.)).item()]
    else:
        logitHInit = 0.
else:
    logitHInit = 0.
    
    


###### 1. Generate marker genotyope ######
##### 1.1. Chromosome numbers and loci information #####
### Read dataset
print("Read dataset")
simDataDict = BSPT.readSimulatedDataInR(dirPyTorch = dirPyTorch,
                                        nChr = nChr)

founderHaplo = simDataDict["founderHaplo"]
genoMap = simDataDict["genoMap"]
lociEffectsTrue = simDataDict["lociEffectsTrue"]
recombRatesList = simDataDict["recombRatesList"]
mrkPos = simDataDict["mrkPos"]


print("Create bsInfo & breederInfo objects")
bsInfoInit = BSPT.BsInfo(founderHaplo = founderHaplo.to(device),
                         genoMap = genoMap,
                         lociEffectsTrue = lociEffectsTrue.to(device),
                         founderIsInitPop = True,
                         nGenerationRM = 0,
                         nGenerationRSM = 0,
                         nGenerationRM2 = 0,
                         propSelRS = 1.,
                         bsName = "Undefined",
                         popNameBase = "Population",
                         initIndNames = None,
                         herit = herit,
                         nRep = 3,
                         initGeneration = 0,
                         includeIntercept = True,
                         genGamMethod = 2,
                         device = device,
                         verbose = True)



breederInfoInit = BSPT.BreederInfo(breederName = "Hamazaki",
                                   bsInfo = bsInfoInit,
                                   mrkPos = mrkPos,
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
                                   device = device,
                                   verbose = True)

if isinstance(logitHInit, float):
    logitHTensor = torch.tensor([[logitHInit] * (nTraits * (1 + int(includeGVP)))] * nGenerationProceed, device = device,
                                requires_grad = True)
else:
    logitHTensor = torch.tensor([logitHInit * nTraits] * nGenerationProceed, 
                                device = device,
                                requires_grad = True)

lowerBound = torch.tensor([[lowerBoundEach] * (nTraits * (1 + int(includeGVP)))] * nGenerationProceed, device = device)
upperBound = torch.tensor([[upperBoundEach] * (nTraits * (1 + int(includeGVP)))] * nGenerationProceed, device = device)

def breedingSimulations(logitHTensor):
    simResMax = torch.zeros(size = (1, nIterSimulation)).squeeze(0)
    
    nIterRemainder = nIterSimulation % nIterSimulationUnit
    nIterQuotient = nIterSimulation // nIterSimulationUnit
    
    if nIterRemainder == 0:
        nRepeat = nIterQuotient + 0
    else:
        nRepeat = nIterQuotient + 1
    
    for repeatNo in range(nRepeat):
        if repeatNo == nRepeat - 1:
            nIterSimulationNow = nIterRemainder + 0
            if nIterSimulationNow == 0:
                nIterSimulationNow = nIterSimulationUnit
        else:
            nIterSimulationNow = nIterSimulationUnit
            
        self = BSPT.SimBs(bsInfoInit = bsInfoInit,
                           mrkPos = mrkPos,
                           breederInfoInit = breederInfoInit,
                           simBsName = "Undefined",
                           lociEffMethod = "true",
                           includeIntercept = True,
                           trainingPopType = "latest",
                           trainingPopInit = None,
                           trainingIndNamesInit = None,
                           methodMLRInit = "Ridge",
                           multiTraitInit = False,
                           nIterSimulation = nIterSimulationNow,
                           nGenerationProceed = nGenerationProceed,
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
                           selectionMethodVec = "WBV",
                           multiTraitsEvalMethodVec = "sum",
                           matingMethodVec = "diallelWithSelfing",
                           allocateMethodVec = "weightedAllocation",
                           weightedAllocationMethodList = weightedAllocationMethodList,
                           logHSelList = torch.tensor(0.),
                           logitHTensor = logitHTensor.unsqueeze(0).reshape(torch.Size([nGenerationProceed, nTraits + int(includeGVP)])),
                           lowerBound = lowerBound,
                           upperBound = upperBound,
                           minimumUnitAllocateVec = 5,
                           clusteringForSelVec = True,
                           nClusterVec = nCluster,
                           nTopClusterVec = nTopCluster,
                           nTopEachVec = None,
                           nSelVec = nSel,
                           nNextPopVec = None,
                           includeGVPVec = includeGVP,
                           nameMethod = "pairBase",
                           nCores = 1,
                           overWriteRes = False,
                           showProgress = True,
                           returnMethod = ["all", "summary"],
                           saveAllResAt = None,
                           evaluateGVMethod = "true",
                           requires_grad = True,
                           nTopEval = 5,
                           logHEval = torch.log(torch.tensor(0.5)),
                           summaryAllResAt = None,
                           device = device,
                           verbose = True)
        self.startSimulation()
        

        for iterNo in range(nIterSimulationNow):
            simResMax[nIterSimulationUnit * repeatNo + iterNo] = self.simResAll[iterNo]["max"]
        
        print(process.memory_info().rss / 1024 ** 2, "[MB]")
        del self
        time.sleep(20.)
        print(gc.collect())
        print(process.memory_info().rss / 1024 ** 2, "[MB]")
    
    return - simResMax.mean()

print(process.memory_info().rss / 1024 ** 2, "[MB]")


del founderHaplo
del genoMap
del lociEffectsTrue
del simDataDict

time.sleep(60.)
print(gc.collect())
print(process.memory_info().rss / 1024 ** 2, "[MB]")

fileNameCurrentInfo = f"{dirSimBsMid}{scriptID}_current_optimization_information.jb"
fileNamePlt = f"{dirSimBsMid}{scriptID}_{simBsNameBase}_check_convergence.png"
# fileNameLogitHTensorList = f"{dirSimBsMid}{scriptID}_{simBsNameBase}_logit_weighted_parameters.pt"
# fileNameLogitHTensorGradList = f"{dirSimBsMid}{scriptID}_{simBsNameBase}_logit_weighted_parameters_gradient.pt"
# fileNameOutputList = f"{dirSimBsMid}{scriptID}_{simBsNameBase}_outputs.pt"


if os.path.isfile(fileNameCurrentInfo):
    currentInfoDict = joblib.load(fileNameCurrentInfo)
    
    optimizer = currentInfoDict["optimizer"]
    scheduler = currentInfoDict["scheduler"]
    logitHTensor = currentInfoDict["logitHTensor"]
    epochNow = currentInfoDict["epoch"] + 1
    
    logitHTensorList = currentInfoDict["logitHTensorList"]
    logitHTensorGradList = currentInfoDict["logitHTensorGradList"]
    outputList = currentInfoDict["outputList"]
else:
    epochNow = 0

    if optimMethod == "SGD":
        optimizer = optim.SGD([logitHTensor], lr = learningRate)
    elif optimMethod == "Adam":
        optimizer = optim.Adam([logitHTensor], lr = learningRate, betas = betas)
    elif optimMethod == "Adadelta":
        optimizer = optim.Adadelta([logitHTensor], lr = learningRate)


    ## StepLR
    if useScheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.8)
    else:
        scheduler = None

    ## MultiSteLR
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25,50,], gamma=0.1)

    ## LAMBDA LR
    ## lambda1 = lambda epoch: 0.955** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    ## ExpotentialLR
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.955)

    ## CosineAnnealing LR
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)


    logitHTensorList = []
    logitHTensorGradList = []
    outputList = []


print("Start optimization")
for epoch in range(epochNow, nIterOptimization):
    print(f"Epoch-{epoch}")
    print(logitHTensor)
    
    if epoch >= 1:
        logitHDiff = logitHTensor.detach().clone() - logitHTensorList[epoch - 1]
    else:
        logitHDiff = logitHTensor.detach().clone() - 1.
    
    logitHDiffSize = torch.mean(logitHDiff * logitHDiff).sqrt().item()
    
    print(f"Parameter differentce score: {logitHDiffSize}")
    if not isinstance(logitHTensor, torch.Tensor):
        if isinstance(logitHTensor, int) or isinstance(logitHTensor, float):
            logitMinusExpHTensor = torch.exp(-torch.tensor(float(logitHTensor)))
    else:
        logitMinusExpHTensor = torch.exp(-logitHTensor)
        hTensor = lowerBound + (upperBound - lowerBound) / (1. + logitMinusExpHTensor)
    # hTensor[torch.isinf(logitExpHTensor)] = upperBound[torch.isinf(logitExpHTensor)]
    
    print(hTensor)
    logitHTensorList.append(logitHTensor.detach().clone())
    
    ### 1. Reset gradient
    includeNan = True
    while includeNan:
        optimizer.zero_grad()
                
        ### 2. Inference
        start = time.time()
        output = breedingSimulations(logitHTensor)
        end = time.time()
        elapsed = end - start
        print(f"Elapsed Time: {elapsed} seconds")
        if useScheduler:
            print(scheduler.get_last_lr())
    
    
        ### 3. Backward propagation
        output.backward()
    
        includeNan = torch.isnan(logitHTensor.grad).any().item()
        
    ### 4. Parameter update
    optimizer.step()
    if useScheduler:
        scheduler.step()
    
    ### 5.Save Results
    logitHTensorGradList.append(logitHTensor.grad)
    
    outputNow = -output.detach()
    outputList.append(outputNow)
    print(logitHTensor.grad)
    print(outputNow)
    

    
    ### 5. plot
    plt.plot(outputList, marker='o', linestyle='-')
    
    plt.title(f'Gen: {nGenerationProceed}, Iter: {nIterSimulation}, Alloc: {selCriTypeName}, OptType: {optimMethod}')
    plt.xlabel('Epoch')
    plt.ylabel('Function Values')
    
    plt.grid(True)
    plt.savefig(fileNamePlt)
    plt.show()
    
    if epoch % 5 == 4:
        currentInfoDict = {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "logitHTensor": logitHTensor,
            "logitHTensorList": logitHTensorList,
            "logitHTensorGradList": logitHTensorGradList,
            "outputList": outputList,
            "epoch": epoch
            }
        
        joblib.dump(currentInfoDict, fileNameCurrentInfo)
        
        del currentInfoDict
    
        # torch.save(logitHTensorList, fileNameLogitHTensorList)
        # torch.save(logitHTensorGradList, fileNameLogitHTensorGradList)
        # torch.save(outputList, fileNameOutputList)
    
    del output
    gc.collect() 
    
    # if logitHDiffSize >= 0.0005:
    #     if not isinstance(logitHTensor, torch.Tensor):
    #         if isinstance(logitHTensor, int) or isinstance(logitHTensor, float):
    #             logitMinusExpHTensor = torch.exp(-torch.tensor(float(logitHTensor)))
    #     else:
    #         logitMinusExpHTensor = torch.exp(-logitHTensor)
    #         hTensor = lowerBound + (upperBound - lowerBound) / (1. + logitMinusExpHTensor)
    #     # hTensor[torch.isinf(logitExpHTensor)] = upperBound[torch.isinf(logitExpHTensor)]
        
    #     print(hTensor)
    #     logitHTensorList.append(logitHTensor.detach().clone())
        
    #     ### 1. Reset gradient
    #     includeNan = True
    #     while includeNan:
    #         optimizer.zero_grad()
                    
    #         ### 2. Inference
    #         start = time.time()
    #         output = breedingSimulations(logitHTensor)
    #         end = time.time()
    #         elapsed = end - start
    #         print(f"Elapsed Time: {elapsed} seconds")
    #         if useScheduler:
    #             print(scheduler.get_last_lr())
        
        
    #         ### 3. Backward propagation
    #         output.backward()
        
    #         includeNan = torch.isnan(logitHTensor.grad).any().item()
            
    #     ### 4. Parameter update
    #     optimizer.step()
    #     if useScheduler:
    #         scheduler.step()
        
    #     ### 5.Save Results
    #     logitHTensorGradList.append(logitHTensor.grad)
        
    #     outputNow = -output.detach()
    #     outputList.append(outputNow)
    #     print(logitHTensor.grad)
    #     print(outputNow)
        

        
    #     ### 5. plot
    #     plt.plot(outputList, marker='o', linestyle='-')
        
    #     plt.title(f'Gen: {nGenerationProceed}, Iter: {nIterSimulation}, Alloc: {selCriTypeName}, OptType: {optimMethod}')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Function Values')
        
    #     plt.grid(True)
    #     plt.savefig(fileNamePlt)
    #     plt.show()
        
    #     if epoch % 5 == 4:
    #         currentInfoDict = {
    #             "optimizer": optimizer,
    #             "scheduler": scheduler,
    #             "logitHTensor": logitHTensor.detach().clone(),
    #             "logitHTensorList": logitHTensorList,
    #             "logitHTensorGradList": logitHTensorGradList,
    #             "outputList": outputList,
    #             "epoch": epoch
    #             }
            
    #         joblib.dump(currentInfoDict, fileNameCurrentInfo)
            
    #         del currentInfoDict
        
    #         # torch.save(logitHTensorList, fileNameLogitHTensorList)
    #         # torch.save(logitHTensorGradList, fileNameLogitHTensorGradList)
    #         # torch.save(outputList, fileNameOutputList)
        
    #     del output
    #     gc.collect() 
    # else:
    #     break
    
   
