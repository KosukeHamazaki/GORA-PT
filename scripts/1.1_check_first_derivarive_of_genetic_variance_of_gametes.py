#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:30:31 2024

@author: hamazaki
"""

###### 0. Settings ######
##### 0.1. Load modules #####
### Import open modules
# import pyper
import time
# import torch.optim as optim
import torch
# import random
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import linkage, fcluster
# from itertools import combinations
import os
# import sys


### Set working directory and import local module, BSPT
dirSCOBSPT = "/Users/hamazaki/GitHub/SCOBS-PT/"
os.chdir(dirSCOBSPT)
fromDiffBreedimport BSPT



### Selection of computation resource (CPU/GPU)
# if sys.platform == "darwin":
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# elif sys.platform == "linux":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")



### Scenario No, Trial No, ...
progName = "SCOBS-PT"
scenarioNo = 2
trialNo = 1
simGenoNo = 1
simPhenoNo = 1
trainingPopName = "true"

scriptIDData = "0.1"
scriptID = "1.1"

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

nGenerationProceed = 1      # int(args[1])
# nIterSimulation = 100       # int(args[2])
# nIterSimulationUnit = 100
nGametes = 50000
# selectTypeName = "SI1"      # str(args[3]): "SI1"; "SI2"; "SI3"; "SI4"
# selCriTypeName = "WBV+GVP"  # str(args[4]): "WBV"; "BV+WBV"; "WBV+GVP"; "BV+WBV+GVP"


### Parameter Settings for optimization  
# optimMethod = "SGD"         # str(args[5]): "SGD", "Adam", "Adadelta"
# useScheduler = True         # bool(args[6])

# learningRate = 0.15         # float(args[7])
# nIterOptimization = 200     # int(args[8])
# betas = [0.1, 0.1]

# logitHInit = 0.
# lowerBoundEach = -0.5
# upperBoundEach = 3.5



### Parameter Settings for others
nTraits = 1
traitNo = 1
traitName = f"Trait_{traitNo}"
gpTypeName = "TrueMrkEff"    # "BRRSet"; "BayesBSet"; "TrueMrkEff"


### Simulated breeding scheme name
# simBsNameBaseWoAllNIter = f"{simTypeName}-{selectTypeName}-{selCriTypeName}-{optimMethod}_{learningRate}-Sche_{useScheduler}-{nGenerationProceed}_Gen"
# simBsNameBaseWoNIterOpt = f"{simBsNameBaseWoAllNIter}-{nIterSimulation}_BS"
# simBsNameBase = f"{simBsNameBaseWoNIterOpt}-{nIterOptimization}_Eval-for-{traitName}-by-{gpTypeName}"

checkDeriName = f"check_first_derivative_of_genetic_variance_of_gametes_with_respect_to_genotype_score_of_parents"


### Directory Names for output
dirCheckDeri = f"{dirTrainingPop}{scriptID}_{checkDeriName}/"

if not os.path.isdir(dirCheckDeri):
    os.mkdir(path = dirCheckDeri)


### Parameter Settings for data simulation
nChr = 10 
herit = 0.7



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



def generateGametes(haplotype, bsInfoInit, n = 1, sampledForAutoGrad = None, method = 2):
    # haplotype = bsInfoInit.populations[0].indsList[0].haplotype
    
    if method >= 3:
        recombRatesList = BSPT.computeRecombBetweenMarkers(haplotype.genoMap,
                                                           device = bsInfoInit.device)
        
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
                
            if method == 3:
                gamCovChol = torch.linalg.cholesky(gamCov)
                gamCovCholList.append(gamCovChol)
            elif method == 3.5:
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
            
            if method <= 2.5:
                if isinstance(method, int):
                    isCrossOver = torch.rand(1) >= 0.5
                else:
                    isCrossOver = True
                    
                if isCrossOver:
                    if bsInfoInit.device == torch.device("mps"):
                        rec0 = np.array(genoMapChr.rec.values).astype("float32").tolist()
                    else:
                        rec0 = genoMapChr.rec.values
            
                    rec = torch.tensor(rec0, device = bsInfoInit.device)
                    crossOverProb = rec.clone()
                    if isinstance(method, int):
                        crossOverProb[1:len(rec)] = crossOverProb[1:len(rec)] * 2
                    ## if not isinstance(method, int), crossOverProb --> recombProb
                    if int(method) == 1:
                        gamHaplo = torch.zeros(len(rec))
                        whichHaploNow = 0
                        for m in range(len(rec)):
                            sample = torch.rand(1, device = bsInfoInit.device)
                            crossOverBool = crossOverProb[m] - sample >= 0
                            if crossOverBool:
                                whichHaploNow = 1 - whichHaploNow
                            else:
                                whichHaploNow = whichHaploNow
                            gamHaplo[m] = haploChr[whichHaploNow, m]
                            
                    elif int(method) == 2:
                        samples = torch.rand(len(crossOverProb), device = bsInfoInit.device)
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
                
                
            elif method >= 3:
                recombRates = recombRatesList[chrNo]
                gamMean = gamMeanList[chrNo]
        
                
                if method == 3:
                    gamCovChol = gamCovCholList[chrNo]
                
                    samples = torch.normal(mean = torch.tensor(0.), 
                                           std = torch.tensor(1.), 
                                           size = (gamCov.size(0), 1))
                    gamHaplo = gamMean + torch.mm(gamCovChol, samples).squeeze()
                elif method == 3.5:
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
            gameteExtend = torch.zeros(size = (gamete.size(0), len(sampledForAutoGrad)), device = bsInfoInit.device)
            gameteExtend[:, whichSampled] = gamete
  
            gamete = torch.matmul(gameteExtend, sampledForAutoGrad)
  
        gametes.append(gamete)
            
    return gametes 


def computeGenVarForGametesSim(bsInfoInit, autoGrad = False, 
                               method = 2, popNo = 0, indNo = 0, 
                               traitNo = 0, n = 100):
    haplotype = bsInfoInit.populations[popNo].indsList[indNo].haplotype
    genoMap = haplotype.genoMap
    chrNames = genoMap["chr"].tolist()
    chrNamesUniq = sorted(list(set(chrNames)), key = None, reverse = False)
    if autoGrad:
        for haploChr, chrNo in zip(haplotype.haploValues.values(),
                                   range(len(haplotype.haploValues.values()))):
            chrName = chrNamesUniq[chrNo]
            haploChr.requires_grad = True
            haplotype.haploValues[chrName] = haploChr
    
    gametes = generateGametes(haplotype, bsInfoInit, n = n, 
                              sampledForAutoGrad = None, method = method)
    genoMatGametes = torch.stack(gametes, dim = 0)
    
    genoMatGametesIncInt = torch.cat(
        (torch.ones((genoMatGametes.size(0), 1), device = device),
         genoMatGametes), dim = 1
        )
    genValsGametes = torch.mm(genoMatGametesIncInt, bsInfoInit.lociEffectsTrue[:, traitNo].unsqueeze(1))
    genVarGametes = torch.var(genValsGametes)
    
    
    return genVarGametes



def computeDerivGenVarForGametes(bsInfoInit, recombRatesMat, 
                                 popNo = 0, indNo = 0,
                                 traitNo = 0):
    bsInfoInit.populations[popNo].obtainHaploArray()
    genoHap = bsInfoInit.populations[popNo].haploArray
    
    genVarGrads = torch.tensor([0.] * genoHap.size(1))
    for m in range(genoHap.size(1)):
        # print(m)
        genVarGradNow = torch.tensor(0.)
        
        for i in range(genoHap.size(1)):
            genVarGradNowEach = (1 - 2 * recombRatesMat[i, m]) * (genoHap[indNo, i, 0] - genoHap[indNo, i, 1]) * bsInfoInit.lociEffectsTrue[i + 1, traitNo] * bsInfoInit.lociEffectsTrue[m + 1, traitNo] / 2
            genVarGradNow = genVarGradNow + genVarGradNowEach
            
        genVarGrads[m] = genVarGradNow
    
    
    return genVarGrads




indNo = 2
popNo = 0
traitNo = 0

chrNames = genoMap["chr"].tolist()
chrNamesUniq = sorted(list(set(chrNames)), key = None, reverse = False)



recombRatesMat = BSPT.computeRecombRatesMat(genoMap, 
                                            device = bsInfoInit.device)
x = computeDerivGenVarForGametes(bsInfoInit, recombRatesMat, 
                                 popNo = popNo, indNo = indNo,
                                 traitNo = traitNo)




bsInfoInitNow1 = bsInfoInit.deepcopy()
st = time.time()
genVar1 = computeGenVarForGametesSim(bsInfoInitNow1, method = 1.5, autoGrad = True,
                                     popNo = popNo, indNo = indNo, 
                                     traitNo = traitNo, n = nGametes)
ed = time.time()
print(ed - st)

genVar1.backward()

y1 = torch.zeros(size = (x.size(0), ))
for haploChr, chrName in zip(bsInfoInitNow1.populations[popNo].indsList[indNo].haplotype.haploValues.values(), chrNamesUniq):
    whichChr = torch.tensor([chrNameNow == chrName for chrNameNow in chrNames])
    y1[whichChr] = haploChr.grad[0, :]




bsInfoInitNow2 = bsInfoInit.deepcopy()
st2 = time.time()
genVar2 = computeGenVarForGametesSim(bsInfoInitNow2, method = 2.5, autoGrad = True,
                                     popNo = popNo, indNo = indNo, 
                                     traitNo = traitNo, n = nGametes)
ed2 = time.time()
print(ed2 - st2)

genVar2.backward()

y2 = torch.zeros(size = (x.size(0), ))
for haploChr, chrName in zip(bsInfoInitNow2.populations[popNo].indsList[indNo].haplotype.haploValues.values(), chrNamesUniq):
    whichChr = torch.tensor([chrNameNow == chrName for chrNameNow in chrNames])
    y2[whichChr] = haploChr.grad[0, :]
    
    

# bsInfoInitNow3 = bsInfoInit.deepcopy()
# st = time.time()
# genVar3 = computeGenVarForGametesSim(bsInfoInitNow3, method = 3, autoGrad = True,
#                                      popNo = popNo, indNo = indNo, 
#                                      traitNo = traitNo, n = nGametes)
# ed = time.time()
# print(ed - st)

# genVar3.backward()


# y3 = torch.zeros(size = (x.size(0), ))
# for haploChr, chrName in zip(bsInfoInitNow3.populations[popNo].indsList[indNo].haplotype.haploValues.values(), chrNamesUniq):
#     whichChr = torch.tensor([chrNameNow == chrName for chrNameNow in chrNames])
#     y3[whichChr] = haploChr.grad[0, :]
    
    

# bsInfoInitNow4 = bsInfoInit.deepcopy()
# st = time.time()
# genVar4 = computeGenVarForGametesSim(bsInfoInitNow4, method = 3.5, autoGrad = True,
#                                      popNo = popNo, indNo = indNo, 
#                                      traitNo = traitNo, n = 10000)
# ed = time.time()
# print(ed - st)

# genVar4.backward()

# y4 = torch.zeros(size = (x.size(0), ))
# for haploChr, chrName in zip(bsInfoInitNow4.populations[popNo].indsList[indNo].haplotype.haploValues.values(), chrNamesUniq):
#     whichChr = torch.tensor([chrNameNow == chrName for chrNameNow in chrNames])
#     y4[whichChr] = haploChr.grad[0, :]



x_np = x.detach().numpy()
y1_np = y1.detach().numpy()
y2_np = y2.detach().numpy()
# y3_np = y3.detach().numpy()
# y4_np = y4.detach().numpy()


corCoef1 = np.round(np.corrcoef(x_np, y1_np)[0, 1], 3)
plt.figure(figsize = (8, 6))
plt.plot(x_np, x_np, color = 'red', label = 'y = x') 
plt.scatter(x_np, y1_np, alpha = 0.5)  # alpha: Transparency
plt.text(x = np.min(x_np), y = np.max(y1_np), s = f"r = {corCoef1}")
plt.title(f'First derivative of genetic variance of {nGametes} gametes')
plt.xlabel('Derived from formula')
plt.ylabel('Estimated from simulation (Method 1)')
plt.grid(True)
fileNameCheckDri1 = f"{dirCheckDeri}{scriptID}_check_first_derivative_of_genetic_variance_of_gametes_with_respect_to_parent_genotype_score_Analytical_vs_Method_1_{nGametes}_gametes.png"
plt.savefig(fileNameCheckDri1)
plt.show()

corCoef2 = np.round(np.corrcoef(x_np, y2_np)[0, 1], 3)
plt.figure(figsize = (8, 6))
plt.plot(x_np, x_np, color = 'red', label = 'y = x') 
plt.scatter(x_np, y2_np, alpha = 0.5)  # alpha: Transparency
plt.text(x = np.min(x_np), y = np.max(y2_np), s = f"r = {corCoef2}")
plt.title(f'First derivative of genetic variance of {nGametes} gametes')
plt.xlabel('Derived from formula')
plt.ylabel('Estimated from simulation (Method 2)')
plt.grid(True)
fileNameCheckDri2 = f"{dirCheckDeri}{scriptID}_check_first_derivative_of_genetic_variance_of_gametes_with_respect_to_parent_genotype_score_Analytical_vs_Method_2_{nGametes}_gametes.png"
plt.savefig(fileNameCheckDri2)
plt.show()



# plt.figure(figsize = (8, 6))
# plt.plot(x_np, x_np, color = 'red', label = 'y = x') 
# plt.scatter(x_np, y3_np, alpha = 0.5)  # alpha: Transparency
# plt.title('First derivative of genetic variance of gametes')
# plt.xlabel('Derived from formula')
# plt.ylabel('Estimated from simulation (Method 3)')
# plt.grid(True)
# plt.show()

# plt.figure(figsize = (8, 6))
# plt.plot(x_np, x_np, color = 'red', label = 'y = x') 
# plt.scatter(x_np, y4_np, alpha = 0.5)  # alpha: Transparency
# plt.title('First derivative of genetic variance of gametes')
# plt.xlabel('Derived from formula')
# plt.ylabel('Estimated from simulation (Method 3.5)')
# plt.grid(True)
# plt.show()

corCoef12 = np.round(np.corrcoef(y1_np, y2_np)[0, 1], 3)
plt.figure(figsize = (8, 6))
plt.plot(y1_np, y1_np, color = 'red', label = 'y = x') 
plt.scatter(y1_np, y2_np, alpha = 0.5)  # alpha: Transparency
plt.text(x = np.min(y1_np), y = np.max(y2_np), s = f"r = {corCoef12}")
plt.title(f'First derivative of genetic variance of {nGametes} gametes')
plt.xlabel('Estimated from simulation (Method 1)')
plt.ylabel('Estimated from simulation (Method 2)')
plt.grid(True)
fileNameCheckDri12 = f"{dirCheckDeri}{scriptID}_check_first_derivative_of_genetic_variance_of_gametes_with_respect_to_parent_genotype_Method_1_vs_Method_2_{nGametes}_gametes.png"
plt.savefig(fileNameCheckDri12)
plt.show()

# plt.figure(figsize = (8, 6))
# plt.plot(y3_np, y3_np, color = 'red', label = 'y = x') 
# plt.scatter(y3_np, y4_np, alpha = 0.5)  # alpha: Transparency
# plt.title('First derivative of genetic variance of gametes')
# plt.xlabel('Estimated from simulation (Method 3)')
# plt.ylabel('Estimated from simulation (Method 3.5)')
# plt.grid(True)
# plt.show()
