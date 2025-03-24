###########################################################################################################################
######  Title: 3.4_SimulatedCrop_SCOBS-PT_evalaute_optimized_weighted_parameter_by_PyTorch_plot_convergence_change   ######
######  Author: Kosuke Hamazaki (hamazaki@ut-biomet.org)                                                             ######
######  Affiliation: Lab. of Biometry and Bioinformatics, The University of Tokyo                                    ######
######  Date: 2024/11/11 (Created), 2024/11/11 (Last Updated)                                                        ######
###########################################################################################################################




###### 1. Settings ######
##### 1.0. Reset workspace ######
# rm(list=ls())



##### 1.1. Setting working directory to the "SCOBS" directory #####
cropName <- "SimulatedCrop"
project <- "SCOBS-PT"
os <- osVersion

if (stringr::str_detect(string = os, pattern = "Ubuntu")) {
  dirSCOBSPT <- "/home/hamazaki/GitHub/SCOBS-PT/"
} else {
  dirSCOBSPT <- "/Users/hamazaki/GitHub/SCOBS-PT/"
}
setwd(dirSCOBSPT)

isRproject <- function(path = getwd()) {
  files <- list.files(path)

  if (length(grep(".Rproj", files)) >= 1) {
    out <- TRUE
  } else {
    out <-  FALSE
  }
  return(out)
}






##### 1.2. Setting some parameters #####
#### 1.2.1. Script IDs, etc...  ####
scriptIDData <- "0.1"
scriptIDOpt <- "2.3"
scriptIDSto <- "2.2"
scriptIDConv <- "3.3"
scriptID <- "3.4"

dirMid <- paste0(dirSCOBSPT, "midstream/")



#### 1.2.2. Setting some parameters related to scenario, trial, and geno names ####
scenarioNo <- 1
simName <- paste0("Scenario_", scenarioNo)
breederName <- "K. Hamazaki"

dirScenario <- paste0(dirMid, scriptIDData, "_", simName, "/")
if (!dir.exists(dirScenario)) {
  dir.create(dirScenario)
}

trialNo <- 1
trialName <- paste0("Trial_", trialNo)

dirTrial <- paste0(dirScenario, scriptIDData,
                   "_", project, "_", trialName, "/")
if (!dir.exists(dirTrial)) {
  dir.create(dirTrial)
}

simGenoNo <- 1
dirSimGeno <- paste0(dirTrial, scriptIDData,
                     "_Geno_", simGenoNo, "/")
if (!dir.exists(dirSimGeno)) {
  dir.create(dirSimGeno)
}



#### 1.2.3. Number of cores ####
nCores <- 25
nCoresWhenOPV <- 25
nCoresWhenEMBV <- 12
nCoresSummary <- 12



#### 1.2.4. Number of generations and simulations ####
simTypeName <- "RecurrentGS"

nGenerationProceed <- 4      # int(args[1])
nGenerationProceedSimulation <- nGenerationProceed
nIterSimulationOpt <- 100       # int(args[2])
nIterSimulation <- 10000


#### 1.2.5. Traits ####
nTraits <- 1
traitNo <- 1
traitName <- paste0("Trait_", traitNo)
gpTypeName <- "TrueMrkEff"    # "BRRSet"; "BayesBSet"; "TrueMrkEff"


#### 1.2.6. Information on breeding schemes and models ####
nNextPopVec <- rep(250, nGenerationProceedSimulation)

nRefreshMemoryEvery <- 2
updateBreederInfo <- rep(TRUE, nGenerationProceed)
phenotypingInds <- rep(FALSE, nGenerationProceed)
nRepForPhenoInit <- 3
nRepForPheno <- rep(1, nGenerationProceed)
updateModels <- rep(FALSE, nGenerationProceed)

updateBreederInfoSimulation <- rep(TRUE, nGenerationProceedSimulation)
phenotypingIndsSimulation <- rep(FALSE, nGenerationProceedSimulation)
nRepForPhenoSimulation <- rep(1, nGenerationProceedSimulation)
updateModelsSimulation <- rep(FALSE, nGenerationProceedSimulation)


# strategyNames <- "selectWBV"
strategyName <- "selectWBV"
selectionMethodList <- rep(list(strategyName), nGenerationProceedSimulation)

# lociEffMethods <- "true"
lociEffMethod <- "true"
trainingPopType <- "latest"
trainingPopInit <- 1
trainingIndNamesInit <- NULL


setGoalAsFinalGeneration <- TRUE
performOptimization <- c(TRUE, rep(FALSE, nGenerationProceed - 1))
useFirstOptimizedValue <- TRUE
performRobustOptimization <- FALSE
lowerQuantile <- 0.25
sameAcrossGeneration <- FALSE


if (gpTypeName == "BRRSet") {
  methodMLRInit <- "BRR"
  multiTraitInit <- FALSE
  methodMLR <- "Ridge"
  multiTrait <- FALSE
} else if (gpTypeName == "BayesBSet") {
  methodMLRInit <- "BayesB"
  multiTraitInit <- FALSE
  methodMLR <- "LASSO"
  multiTrait <- FALSE
} else if (gpTypeName == "TrueMrkEff") {
  methodMLRInit <- "BRR"
  multiTraitInit <- FALSE
  methodMLR <- "Ridge"
  multiTrait <- FALSE
}

alpha <- 0.5
nIter <- 20000
burnIn <- 5000
thin <- 5
bayesian <- TRUE


#### 1.2.7. Parameters related to selection ####
nSelectionWaysVec <- rep(1, nGenerationProceedSimulation)
traitNoSelList <- rep(list(list(1)), nGenerationProceedSimulation)

blockSplitMethod <- "minimumSegmentLength"
nMrkInBlock <- 1
minimumSegmentLength <- 100
nSelInitOPVList <- rep(list(50), nGenerationProceedSimulation)
nIterOPV <- 2e04
nProgeniesEMBVVec <- rep(40, nGenerationProceedSimulation)
nIterEMBV <- 5
nCoresEMBV <- 1


multiTraitsEvalMethodList <- rep(list("sum"), nGenerationProceedSimulation)
hSelList <- rep(list(list(1)), nGenerationProceedSimulation)


#### 1.2.8. Parameters related to mating methods ####
matingMethodVec <- rep("diallelWithSelfing", nGenerationProceedSimulation)
allocateMethodVec <- rep("weightedAllocation", nGenerationProceedSimulation)
traitNoRAList <- rep(list(1), nGenerationProceedSimulation)
minimumUnitAllocateVec <- rep(5, nGenerationProceedSimulation)



#### 1.2.9. Parameters related to OCS ####
targetGenGVPVec <- rep(0, nGenerationProceedSimulation)
performOCSVec <- rep(FALSE, nGenerationProceedSimulation)
targetGenOCSVec <- rep(nGenerationProceedSimulation, nGenerationProceedSimulation)
HeStarRatioVec <- rep(0.01, nGenerationProceedSimulation)
degreeOCSVec <- rep(1, nGenerationProceedSimulation)
includeGVPOCSVec <- rep(FALSE, nGenerationProceedSimulation)
hOCSList <- rep(list(rep(1, nTraits)), nGenerationProceedSimulation)
weightedAllocationMethodOCSList <- rep(list(rep("selectBV", nTraits)), nGenerationProceedSimulation)
traitNoOCSList <- rep(1:nTraits, nGenerationProceedSimulation)
nCrossesOCSVec <- rep(25, nGenerationProceedSimulation)


#### 1.2.10. Other fixed parameters ####
nameMethod <- "pairBase"
overWriteRes <- FALSE
showProgress <- TRUE
returnMethod <- "summary"
evaluateGVMethod <- "true"
nTopEval <- 5
traitNoEval <- 1
hEval <- 1
verbose <- TRUE

saveAllResAtBase <- "all_results"
summaryAllResAtBase <- "all_results"

nChr <- 10
herit <- 0.7




#### 1.2.11. Parameter related to gradient-based optimization ####
optimMethod <- "SGD"         # str(args[5]): "SGD", "Adam", "Adadelta"
useScheduler <- "True"       # bool(args[6])

learningRate <- 0.15         # float(args[7])
betas <- c(0.1, 0.1)

lowerBound <- -0.5
upperBound <- 3.5



#### 1.2.12. Directory names ####
simPhenoNo <- 1
dirSimPheno <- paste0(dirSimGeno, scriptIDData,
                      "_Pheno_", simPhenoNo, "/")
if (!dir.exists(dirSimPheno)) {
  dir.create(dirSimPheno)
}


nTrainingPopInit <- "true"
dirTrainingPop <- paste0(dirSimPheno, scriptIDData,
                         "_Model_with_", nTrainingPopInit, "/")
if (!dir.exists(dirTrainingPop)) {
  dir.create(dirTrainingPop)
}

dirTrainingPop250 <- paste0(dirSimPheno, scriptIDData,
                            "_Model_with_250/")
# dirPyTorch <- paste0(dirTrainingPop250, "_PyTorch/")





#### 1.2.12. Changing parameters depending on strategies ####
# selectTypeNames <- paste0("SI", c(1, 2, 4))
selectTypeNames <- paste0("SI4")

# selectTypeName <- "SI4"      # str(args[3]): "SI1"; "SI2"; "SI3"; "SI4"
for (selectTypeName in selectTypeNames) {
  if (selectTypeName == "SI1") {
    clusteringForSelList <- rep(list(FALSE), nGenerationProceedSimulation)
    nClusterList <- rep(list(1), nGenerationProceedSimulation)
    nTopClusterList <- rep(list(1), nGenerationProceedSimulation)
    nTopEachList <- rep(list(15), nGenerationProceedSimulation)
    nSelList <- rep(list(15), nGenerationProceedSimulation)
  } else if (selectTypeName == "SI2") {
    clusteringForSelList <- rep(list(TRUE), nGenerationProceedSimulation)
    nClusterList <- rep(list(25), nGenerationProceedSimulation)
    nTopClusterList <- rep(list(25), nGenerationProceedSimulation)
    nTopEachList <- rep(list(1), nGenerationProceedSimulation)
    nSelList <- rep(list(25), nGenerationProceedSimulation)
  } else if (selectTypeName == "SI3") {
    clusteringForSelList <- rep(list(TRUE), nGenerationProceedSimulation)
    nClusterList <- rep(list(10), nGenerationProceedSimulation)
    nTopClusterList <- rep(list(10), nGenerationProceedSimulation)
    nTopEachList <- rep(list(5), nGenerationProceedSimulation)
    nSelList <- rep(list(50), nGenerationProceedSimulation)
  } else if (selectTypeName == "SI4") {
    clusteringForSelList <- rep(list(FALSE), nGenerationProceedSimulation)
    nClusterList <- rep(list(1), nGenerationProceedSimulation)
    nTopClusterList <- rep(list(1), nGenerationProceedSimulation)
    nTopEachList <- rep(list(50), nGenerationProceedSimulation)
    nSelList <- rep(list(50), nGenerationProceedSimulation)
  }


  # selCriTypeNames <- c("WBV", "WBV+GVP")  # str(args[4]): "WBV"; "BV+WBV"; "WBV+GVP"; "BV+WBV+GVP"
  selCriTypeNames <- "WBV+GVP"  # str(args[4]): "WBV"; "BV+WBV"; "WBV+GVP"; "BV+WBV+GVP"
  # selCriTypeName <- "WBV+GVP"  # str(args[4]): "WBV"; "BV+WBV"; "WBV+GVP"; "BV+WBV+GVP"
  for (selCriTypeName in selCriTypeNames) {
    if (selCriTypeName == "WBV") {
      weightedAllocationMethodList <- rep(list("selectWBV"), nGenerationProceedSimulation)
      includeGVPVec <- rep(FALSE, nGenerationProceedSimulation)
    } else if (selCriTypeName == "BV+WBV") {
      weightedAllocationMethodList <- rep(list(c("selectBV", "selectWBV")), nGenerationProceedSimulation)
      includeGVPVec <- rep(FALSE, nGenerationProceedSimulation)
    } else if (selCriTypeName == "WBV+GVP") {
      weightedAllocationMethodList <- rep(list("selectWBV"), nGenerationProceedSimulation)
      includeGVPVec <- rep(TRUE, nGenerationProceedSimulation)
    } else if (selCriTypeName == "BV+WBV+GVP") {
      weightedAllocationMethodList <- rep(list(c("selectBV", "selectWBV")), nGenerationProceedSimulation)
      includeGVPVec <- rep(TRUE, nGenerationProceedSimulation)
    }


    if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
      # gvpInits <- c("Mean", "Zero")
      # gvpInits <- "Zero"
      gvpInits <- "Mean"
    } else {
      gvpInits <- "Mean"
    }

    # gvpInit <- "Zero"            # str(args[9]): "Mean", "Zero"
    for (gvpInit in gvpInits) {
      nIterOptimization <- 200     # int(args[8])


      #### 1.2.13. Simulated breeding scheme name ####
      simBsNameBaseWoAllNIter <- paste0(simTypeName, "-", selectTypeName, "-",
                                        selCriTypeName, "-", optimMethod, "_",
                                        learningRate, "-Sche_", useScheduler, "-",
                                        nGenerationProceed, "_Gen-InitGVP_", gvpInit)
      simBsNameBaseWoNIterOpt <- paste0(simBsNameBaseWoAllNIter, "-",
                                        nIterSimulationOpt, "_BS")
      simBsNameBase <- paste0(simBsNameBaseWoNIterOpt, "-",
                              nIterOptimization, "_Eval-for-",
                              traitName, "-by-", gpTypeName)


      #### 1.2.14. Directory Names for output ####
      dirSimBs <- paste0(dirTrainingPop, scriptIDOpt, "_", simBsNameBase, "/")
      if (!dir.exists(dirSimBs)) {
        dir.create(dirSimBs)
      }

      dirSimBsMid <- paste0(dirSimBs, scriptIDOpt, "_current_results/")
      if (!dir.exists(dirSimBsMid)) {
        dir.create(dirSimBsMid)
      }




      ##### 1.3. Import packages #####
      require(myBreedSimulatR)
      require(data.table)
      require(MASS)
      require(rrBLUP)
      require(BGLR)
      require(RAINBOWR)
      require(ggplot2)
      require(plotly)
      require(reticulate)

      np <- reticulate::import("numpy")
      joblib <- reticulate::import("joblib")
      torch <- reticulate::import("torch")



      ##### 1.4. Project options #####
      options(stringAsFactors = FALSE)



      ###### 2. Evaluation of optimized parameter by PyTorch in R ######
      ##### 2.1. Preparation for breeding simulation: read bsInfo $ breederInfo into R #####
      fileNameBsInfoInitRmInit <- paste0(dirTrainingPop250, scriptIDData,
                                         "_bsInfoInit_remove_initial_population.rds")
      fileNameBreederInfoInitWithMrkEffRmInit <- paste0(dirTrainingPop250, scriptIDData,
                                                        "_breederInfoInit_with_estimated_marker_effects_",
                                                        methodMLRInit, "_", 250,
                                                        "_individuals_remove_initial_population.rds")
      myBsInfoInit <- readRDS(file = fileNameBsInfoInitRmInit)
      myBreederInfoInit <- readRDS(file = fileNameBreederInfoInitWithMrkEffRmInit)


      if (any(includeGVPVec)) {
        if (is.null(myBsInfoInit$lociInfo$recombBetweenMarkersList)) {
          myBsInfoInit$lociInfo$computeRecombBetweenMarkers()
          myBsInfoInit$traitInfo$lociInfo$computeRecombBetweenMarkers()
          myBsInfoInit$populations[[myBsInfoInit$generation]]$traitInfo$lociInfo$computeRecombBetweenMarkers()
        }
      }









      ##### 2.2. Perform breeding simulations for optimized parameters #####
      #### 2.2.1. Information on the breesding simulations ####
      # print(paste0(paste0("Geno_", simGenoNo), "  ",
      #              paste0("Pheno_", simPhenoNo), "  ",
      #              strategyName, "  ", lociEffMethod))
      print(paste0(paste0("Pheno_", simPhenoNo), "  ",
                   paste0("nGenerations = ", nGenerationProceed), "  ",
                   selectTypeName, "  ", selCriTypeName,
                   "  ", gvpInit))
      simBsName <- simBsNameBase

      saveAllResAt <- NULL
      summaryAllResAt <- NULL


      #### 2.2.2. Read optimized parameter by PyTorch into R ####
      fileNameOptimInfo <- paste0(dirSimBsMid, scriptIDOpt,
                                  "_current_optimization_information.jb")

      if (file.exists(fileNameOptimInfo)) {
        nGrid <- 5
        print("Gradient-based Optimized Resource Allocation")
        optimInfo <- joblib$load(filename = fileNameOptimInfo)
        logitHEvalList <- lapply(
          X = optimInfo$logitHTensorList[c(1, seq(nGrid, length(optimInfo$logitHTensorList), by = nGrid))],
          FUN = np$array
        )
        names(logitHEvalList) <- paste0("NBS", seq(0, length(optimInfo$logitHTensorList), by = nGrid) * nIterSimulationOpt)
        logitHEvalList <- logitHEvalList[!duplicated(logitHEvalList)]
        evalGORANames <- names(logitHEvalList)

        nEvalsGORA <- length(logitHEvalList)
        geneticGainsGORA <- rep(NA, nEvalsGORA)
        names(geneticGainsGORA) <- evalGORANames
        for (evalGORANo in 1:nEvalsGORA) {
          st <- Sys.time()
          evalGORAName <- evalGORANames[evalGORANo]
          print(paste0("GORA: ", evalGORAName))
          logitH <- logitHEvalList[[evalGORANo]]

          hListNow <- apply(X = lowerBound + (upperBound - lowerBound) / (1 + exp(-logitH)),
                            MARGIN = 1, FUN = function(x) x, simplify = FALSE)

          simBsNameConv <- paste0(simBsName, "-GORA")
          fileNameSimBsBVConv <- paste0(dirSimBs, scriptIDConv, "_", simBsNameConv,
                                        "-", evalGORAName, "_simBs.rds")

          if (file.exists(fileNameSimBsBVConv)) {
            mySimBsBVNow <- readRDS(file = fileNameSimBsBVConv)

            summaryRes <- mySimBsBVNow$trueGVSummaryArray
            # whichTrueRes <- apply(diff(summaryRes[1, 1, 1:5, ]), 2, function(x) all(x > 0))
            trueGVAdjusted <- (summaryRes[1, 1, 1:(nGenerationProceed + 1), ] -
                                 summaryRes[2, 1, rep(1, nGenerationProceed + 1), ]) / sqrt(summaryRes[5, 1, rep(1, nGenerationProceed + 1), ])
            trueGVAdjustedMean <- apply(trueGVAdjusted, 1, mean)
            geneticGainGORANow <- trueGVAdjustedMean[nGenerationProceed + 1] / trueGVAdjustedMean[1]
            geneticGainsGORA[evalGORANo] <- geneticGainGORANow
          }
        }
      }
    }




    ##### 2.3. Perform breeding simulations for equal allocation #####
    print("Equal allocation")

    simBsNameEqual <- paste0(simBsName, "-EQ")
    fileNameSimBsEqual <- paste0(dirSimBs, scriptIDConv, "_", simBsNameEqual, "_simBs.rds")



    if (file.exists(fileNameSimBsEqual)) {
      mySimBsBVEqual <- readRDS(file = fileNameSimBsEqual)
      summaryRes <- mySimBsBVEqual$trueGVSummaryArray
      trueGVAdjusted <- (summaryRes[1, 1, 1:(nGenerationProceed + 1), ] -
                           summaryRes[2, 1, rep(1, nGenerationProceed + 1), ]) / sqrt(summaryRes[5, 1, rep(1, nGenerationProceed + 1), ])
      trueGVAdjustedMean <- apply(trueGVAdjusted, 1, mean)
      geneticGainEQ <- trueGVAdjustedMean[nGenerationProceed + 1] / trueGVAdjustedMean[1]
    } else {
      geneticGainEQ <- NA
    }


    ##### 2.4. Perform breeding simulations for optimized allocation with SToSOO #####
    print("Optimized allocation with StoSOO")


    if (selCriTypeName == "WBV") {
      selCriTypeNameSto <- "SimplestOpt"
    } else if (selCriTypeName == "WBV+GVP") {
      selCriTypeNameSto <- "IncludeGVPOpt"
    }

    if (nIterOptimization == 200) {
      nIterOptimizationSto <- 20000
    } else if (nIterOptimization == 100) {
      nIterOptimizationSto <- 10000
    }
    nIterSimulationSto <- 50

    simBsNameSto <- paste0(simTypeName, "-", selectTypeName,
                           "-", selCriTypeNameSto, "-",
                           nGenerationProceed, "_Gen-",
                           nIterSimulationSto, "_BS-1_ME-",
                           nIterOptimizationSto, "_Eval-for-", traitName,
                           "-by-BRRSet-", nTrainingPopInit, "-", strategyName,
                           "-", lociEffMethod)
    dirSimBsSto <- paste0(dirTrainingPop, scriptIDSto, "_",
                          simBsNameSto, "/")
    fileNameOptimInfoSto <- paste0(dirSimBsSto, scriptIDSto, "_",
                                   simBsNameSto, "_simBs.rds")

    optimInfoSto <- readRDS(file = fileNameOptimInfoSto)
    hVecMat <- optimInfoSto$optimalHyperParamMatsList$Initial
    rownames(hVecMat) <- paste0("NBS",
                                readr::parse_number(rownames(hVecMat)) * nIterSimulationSto)
    rownames(hVecMat)[1] <- "NBS0"
    evalORANames <- rownames(hVecMat)
    nEvalsORA <- nrow(hVecMat)

    geneticGainsORA <- rep(NA, nEvalsORA)
    names(geneticGainsORA) <- evalORANames

    for (evalORANo in 1:nEvalsORA) {
      evalORAName <- evalORANames[evalORANo]
      print(paste0("ORA: ", evalORAName))

      simBsNameStoSOO <- paste0(simBsName, "-ORA")
      fileNameSimBsStoSOO <- paste0(dirSimBs, scriptIDConv, "_", simBsNameStoSOO,
                                    "-", evalORAName, "_simBs.rds")

      mySimBsBVStoSOO <- readRDS(file = fileNameSimBsStoSOO)
      summaryRes <- mySimBsBVStoSOO$trueGVSummaryArray
      trueGVAdjusted <- (summaryRes[1, 1, 1:(nGenerationProceed + 1), ] -
                           summaryRes[2, 1, rep(1, nGenerationProceed + 1), ]) / sqrt(summaryRes[5, 1, rep(1, nGenerationProceed + 1), ])
      trueGVAdjustedMean <- apply(trueGVAdjusted, 1, mean)
      geneticGainORANow <- trueGVAdjustedMean[nGenerationProceed + 1] / trueGVAdjustedMean[1]
      geneticGainsORA[evalORANo] <- geneticGainORANow
    }


  }
}

geneticGainsGORAImp <- geneticGainsGORA
geneticGainsGORAImp[is.na(geneticGainsGORAImp)] <- 0
geneticGainsGORAImp <- c(geneticGainsGORAImp, 0)
nbsGORAs <- c(readr::parse_number(evalGORANames), 50 * nIterOptimizationSto)
geneticGainsGORAMax <- rep(NA, length(nbsGORAs))
geneticGainsGORAMaxNow <- geneticGainsGORAImp[1]
for (i in 1:length(nbsGORAs)) {
  if (geneticGainsGORAImp[i] > geneticGainsGORAMaxNow) {
    geneticGainsGORAMax[i] <- geneticGainsGORAImp[i]
    geneticGainsGORAMaxNow <- geneticGainsGORAImp[i]
  } else {
    geneticGainsGORAMax[i] <- geneticGainsGORAMaxNow
  }
}
plot(nbsGORAs, geneticGainsGORAMax, col = 2, type = "l")


geneticGainsORAImp <- c(geneticGainsORA, geneticGainsORA[length(geneticGainsORA)])
nbsORAs <- c(readr::parse_number(evalORANames), 50 * nIterOptimizationSto)
geneticGainsORAMax <- rep(NA, length(nbsORAs))
geneticGainsORAMaxNow <- geneticGainsORAImp[1]
for (i in 1:length(nbsORAs)) {
  if (geneticGainsORAImp[i] > geneticGainsORAMaxNow) {
    geneticGainsORAMax[i] <- geneticGainsORAImp[i]
    geneticGainsORAMaxNow <- geneticGainsORAImp[i]
  } else {
    geneticGainsORAMax[i] <- geneticGainsORAMaxNow
  }
}


plot(nbsORAs, geneticGainsORAMax, col = 2, type = "l")
points(nbsGORAs, geneticGainsGORAMax, col = 2, type = "l")

# plot(nbsORAs, geneticGainsORAMax, type = "l",
#      ylim = range(c(geneticGainsORAMax,
#                     geneticGainsGORAMax,
#                     geneticGainEQ)))
# points(nbsGORAs, geneticGainsGORAMax, col = 2, type = "l")
# abline(h = geneticGainEQ, col = 3, lty = 3)


nIterConvDiffORA <- diff(c(nbsORAs, 50 * nIterOptimizationSto + 1))
trueGVFinalGenConvORA <- rep(geneticGainsORAMax,
                             nIterConvDiffORA)
nIterConvDiffGORA <- diff(c(nbsGORAs, 50 * nIterOptimizationSto + 1))
trueGVFinalGenConvGORA <- rep(geneticGainsGORAMax,
                              nIterConvDiffGORA)

dfForGGPlot <- data.frame(Iteration = rep(seq(from = 0, to = 50 * nIterOptimizationSto, by = 1), 2),
                          Value = c(trueGVFinalGenConvGORA, trueGVFinalGenConvORA),
                          Strategy = rep(c("GORA1", "ORA"), each = 50 * nIterOptimizationSto + 1))

selectTypeCols20 <- c("#cc99ff", "#660099", "#0066cc", "#333333")
selectTypeLwds <- 1.75
selectTypeCols2 <- selectTypeCols20[c(1, 3)]

pltConv <- ggplot2::ggplot(data = dfForGGPlot) +
  ggplot2::geom_line(mapping = aes(x = Iteration,
                                   y = Value,
                                   group = Strategy,
                                   color = Strategy),
                     # lty = rep(selectTypeLtys2, each = 50 * nIterOptimizationSto + 1),
                     lwd = selectTypeLwds) +
  # ggplot2::geom_hline(yintercept = trueGVSummaryConv[nrow(trueGVSummaryConv), ncol(trueGVSummaryConv)] -
  #                       trueGVSummaryConv[nrow(trueGVSummaryConv), 1],
  #                     color = "red", linetype = "dashed") +
  ggplot2::xlim(0, 50 * nIterOptimizationSto) +
  ggplot2::scale_color_manual(values = selectTypeCols2) +
  ggplot2::theme(text = element_text(size = 48)) +
  ggplot2::ylab(label = "Genetic gain")

fileNamePltConv <- paste0(dirSimBs, scriptID, "_", simBsName,
                          "_true_convergence_evaluation.png")

png(filename = fileNamePltConv, width = 1200, height = 1000)
print(pltConv)
dev.off()
