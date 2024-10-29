###################################################################################################
######  Title: 3.1_SimulatedCrop_SCOBS-PT_evalaute_optimized_weighted_parameter_by_PyTorch   ######
######  Author: Kosuke Hamazaki (hamazaki@ut-biomet.org)                                     ######
######  Affiliation: Lab. of Biometry and Bioinformatics, The University of Tokyo            ######
######  Date: 2024/07/09 (Created), 2024/07/09 (Last Updated)                                ######
###################################################################################################




###### 1. Settings ######
##### 1.0. Reset workspace ######
# rm(list=ls())



##### 1.1. Setting working directory to the "SCOBS" directory #####
cropName <- "SimulatedCrop"
project <- "SCOBS-PT"
os <- osVersion

if (stringr::str_detect(string = os, pattern = "Ubuntu")) {
  dirSCOBSPT <- "/home/hamazaki/SCOBS-PT/"
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
scriptIDOpt <- "2.1"
scriptIDSto <- "2.2"
scriptID <- "3.1"

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
nCores <- 5
nCoresWhenOPV <- 5
nCoresWhenEMBV <- 3
nCoresSummary <- 3



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
# nSimPheno <- 1
nSimPheno <- 10
simPhenoNos <- 1:nSimPheno
for (simPhenoNo in simPhenoNos) {
  # simPhenoNo <- 1
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
  selectTypeNames <- paste0("SI", c(1, 2, 4))

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

    if (nSimPheno == 1) {
      selCriTypeNames <- c("WBV", "WBV+GVP")  # str(args[4]): "WBV"; "BV+WBV"; "WBV+GVP"; "BV+WBV+GVP"
    } else {
      selCriTypeNames <- "WBV+GVP"  # str(args[4]): "WBV"; "BV+WBV"; "WBV+GVP"; "BV+WBV+GVP"
    }
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
        if (nSimPheno == 1) {
          gvpInits <- c("Mean", "Zero")
        } else {
          gvpInits <- "Zero"
        }
      } else {
        gvpInits <- "Mean"
      }

      # gvpInit <- "Zero"            # str(args[9]): "Mean", "Zero"
      for (gvpInit in gvpInits) {
        if (nSimPheno == 1) {
          nIterOptimization <- 200     # int(args[8])
        } else {
          nIterOptimization <- 100     # int(args[8])
        }
        #### 1.2.13. Simulated breeding scheme name ####
        simBsNameBaseWoAllNIter <- paste0(simTypeName, "-", selectTypeName, "-",
                                          selCriTypeName, "-", optimMethod, "_",
                                          learningRate, "-Sche_", useScheduler, "-",
                                          nGenerationProceed, "_Gen-InitGVP_", gvpInit)
        simBsNameBaseWoNIterOpt <- paste0(simBsNameBaseWoAllNIter, "-",
                                          nIterSimulationOpt, "_BS")
        if (simPhenoNo == 1) {
          simBsNameBase <- paste0(simBsNameBaseWoNIterOpt, "-",
                                  200, "_Eval-for-",
                                  traitName, "-by-", gpTypeName)
        } else {
          simBsNameBase <- paste0(simBsNameBaseWoNIterOpt, "-",
                                  nIterOptimization, "_Eval-for-",
                                  traitName, "-by-", gpTypeName)
        }

        #### 1.2.14. Directory Names for output ####
        dirSimBs <- paste0(dirTrainingPop, scriptIDOpt, "_", simBsNameBase, "/")
        if (!dir.exists(dirSimBs)) {
          dir.create(dirSimBs)
        }

        if (simPhenoNo == 1) {
          simBsNameBase <- paste0(simBsNameBaseWoNIterOpt, "-",
                                  nIterOptimization, "_Eval-for-",
                                  traitName, "-by-", gpTypeName)
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
          st <- Sys.time()

          print("Gradient-based Optimized Resource Allocation")
          optimInfo <- joblib$load(filename = fileNameOptimInfo)
          logitH <- np$array(optimInfo$logitHTensorList[[nIterOptimization]]$detach())
          hListNow <- apply(X = lowerBound + (upperBound - lowerBound) / (1 + exp(-logitH)),
                            MARGIN = 1, FUN = function(x) x, simplify = FALSE)

          simBsNameConv <- paste0(simBsName, "-GORA")
          fileNameSimBsBVConv <- paste0(dirSimBs, scriptID, "_", simBsNameConv, "_simBs.rds")


          if (!file.exists(fileNameSimBsBVConv)) {
            #### 2.2.3. Set simBs class object ####
            mySimBsBVNow <- myBreedSimulatR::simBs$new(simBsName = simBsNameConv,
                                                       bsInfoInit = myBsInfoInit,
                                                       breederInfoInit = myBreederInfoInit,
                                                       lociEffMethod = lociEffMethod,
                                                       trainingPopType = trainingPopType,
                                                       trainingPopInit = trainingPopInit,
                                                       trainingIndNamesInit = trainingIndNamesInit,
                                                       methodMLRInit = methodMLRInit,
                                                       multiTraitInit = multiTraitInit,
                                                       nIterSimulation = nIterSimulation,
                                                       nGenerationProceed = nGenerationProceed,
                                                       nRefreshMemoryEvery = nRefreshMemoryEvery,
                                                       updateBreederInfo = updateBreederInfo,
                                                       phenotypingInds = phenotypingInds,
                                                       nRepForPhenoInit = nRepForPhenoInit,
                                                       nRepForPheno = nRepForPheno,
                                                       updateModels = updateModels,
                                                       methodMLR = methodMLR,
                                                       multiTrait = multiTrait,
                                                       nSelectionWaysVec = nSelectionWaysVec,
                                                       selectionMethodList = selectionMethodList,
                                                       traitNoSelList = traitNoSelList,
                                                       blockSplitMethod = blockSplitMethod,
                                                       nMrkInBlock = nMrkInBlock,
                                                       minimumSegmentLength = minimumSegmentLength,
                                                       nSelInitOPVList = nSelInitOPVList,
                                                       nIterOPV = nIterOPV,
                                                       nProgeniesEMBVVec = nProgeniesEMBVVec,
                                                       nIterEMBV = nIterEMBV,
                                                       nCoresEMBV = nCoresEMBV,
                                                       clusteringForSelList = clusteringForSelList,
                                                       nClusterList = nClusterList,
                                                       nTopClusterList = nTopClusterList,
                                                       nTopEachList = nTopEachList,
                                                       nSelList = nSelList,
                                                       multiTraitsEvalMethodList = multiTraitsEvalMethodList,
                                                       hSelList = hSelList,
                                                       matingMethodVec = matingMethodVec,
                                                       allocateMethodVec = allocateMethodVec,
                                                       weightedAllocationMethodList = weightedAllocationMethodList,
                                                       traitNoRAList = traitNoRAList,
                                                       hList = hListNow,
                                                       minimumUnitAllocateVec = minimumUnitAllocateVec,
                                                       includeGVPVec = includeGVPVec,
                                                       targetGenGVPVec = targetGenGVPVec,
                                                       nNextPopVec = nNextPopVec,
                                                       performOCSVec = performOCSVec,
                                                       targetGenOCSVec = targetGenOCSVec,
                                                       HeStarRatioVec = HeStarRatioVec,
                                                       degreeOCSVec = degreeOCSVec,
                                                       includeGVPOCSVec = includeGVPOCSVec,
                                                       hOCSList = hOCSList,
                                                       weightedAllocationMethodOCSList = weightedAllocationMethodOCSList,
                                                       traitNoOCSList = traitNoOCSList,
                                                       nCrossesOCSVec = nCrossesOCSVec,
                                                       nameMethod = nameMethod,
                                                       nCores =  nCores,
                                                       overWriteRes = overWriteRes,
                                                       showProgress = showProgress,
                                                       returnMethod = returnMethod,
                                                       saveAllResAt = saveAllResAt,
                                                       evaluateGVMethod = evaluateGVMethod,
                                                       nTopEval = nTopEval,
                                                       traitNoEval = traitNoEval,
                                                       hEval = hEval,
                                                       summaryAllResAt = summaryAllResAt,
                                                       verbose = verbose)


            #### 2.2.4. Start simulation, summary results, and save simBs class object ####
            print("Start simulation....")

            checkFinishSimulation <- FALSE
            while (!checkFinishSimulation) {
              if (strategyName == "selectOPV") {
                mySimBsBVNow$nCores <- nCoresWhenOPV
              }

              if (strategyName == "selectEMBV") {
                mySimBsBVNow$nCores <- nCoresWhenEMBV
              }
              system.time(
                mySimBsBVNow$startSimulation()
              )

              if (!is.null(saveAllResAt)) {
                saveAllResAtSplit <- stringr::str_split(string = list.files(saveAllResAt),
                                                        pattern = "_")
                saveAllResAtSplitLast <- lapply(X = saveAllResAtSplit,
                                                FUN = function(saveAllResAtSplitVec) {
                                                  return(saveAllResAtSplitVec[length(saveAllResAtSplitVec)])
                                                })

                saveAllNumeric <- unique(sort(as.numeric(stringr::str_remove(saveAllResAtSplitLast, ".rds"))))

                checkFinishSimulation <- all(unlist(lapply(mySimBsBVNow$trueGVMatList, length)) == (nGenerationProceed + 1)) |
                  all((1:nIterSimulation) %in% saveAllNumeric)
              } else {
                checkFinishSimulation <- all(unlist(lapply(mySimBsBVNow$trueGVMatList, length)) == (nGenerationProceed + 1))
              }
            }

            if (strategyName %in% c("selectOPV", "selectEMBV")) {
              mySimBsBVNow$nCores <- nCores
            }

            if (all(unlist(lapply(mySimBsBVNow$trueGVMatList, length)) == (nGenerationProceed + 1))) {
              mySimBsBVNow$summaryResults()
            } else {
              print("Start summarization....")
              mySimBsBVNow$summaryAllResAt <- saveAllResAt
              mySimBsBVNow$nCores <- nCoresSummary
              mySimBsBVNow$summaryResults()
              mySimBsBVNow$nCores <- nCores
            }



            saveRDS(object = mySimBsBVNow, file = fileNameSimBsBVConv)


            rm(mySimBsBVNow)
            gc(reset = TRUE)
          }
          ed <- Sys.time()

          print(paste0("Computation time required: ", round(ed - st, 2), " ",
                       attr(ed - st, "units")))
        }
      }




      ##### 2.3. Perform breeding simulations for equal allocation #####
      print("Equal allocation")

      st <- Sys.time()
      simBsNameEqual <- paste0(simBsName, "-EQ")
      fileNameSimBsEqual <- paste0(dirSimBs, scriptID, "_", simBsNameEqual, "_simBs.rds")



      if (!file.exists(fileNameSimBsEqual)) {
        mySimBsBVEqual <- myBreedSimulatR::simBs$new(simBsName = simBsNameEqual,
                                                     bsInfoInit = myBsInfoInit,
                                                     breederInfoInit = myBreederInfoInit,
                                                     lociEffMethod = lociEffMethod,
                                                     trainingPopType = trainingPopType,
                                                     trainingPopInit = trainingPopInit,
                                                     trainingIndNamesInit = trainingIndNamesInit,
                                                     methodMLRInit = methodMLRInit,
                                                     multiTraitInit = multiTraitInit,
                                                     nIterSimulation = nIterSimulation,
                                                     nGenerationProceed = nGenerationProceed,
                                                     nRefreshMemoryEvery = nRefreshMemoryEvery,
                                                     updateBreederInfo = updateBreederInfo,
                                                     phenotypingInds = phenotypingInds,
                                                     nRepForPhenoInit = nRepForPhenoInit,
                                                     nRepForPheno = nRepForPheno,
                                                     updateModels = updateModels,
                                                     methodMLR = methodMLR,
                                                     multiTrait = multiTrait,
                                                     nSelectionWaysVec = nSelectionWaysVec,
                                                     selectionMethodList = selectionMethodList,
                                                     traitNoSelList = traitNoSelList,
                                                     blockSplitMethod = blockSplitMethod,
                                                     nMrkInBlock = nMrkInBlock,
                                                     minimumSegmentLength = minimumSegmentLength,
                                                     nSelInitOPVList = nSelInitOPVList,
                                                     nIterOPV = nIterOPV,
                                                     nProgeniesEMBVVec = nProgeniesEMBVVec,
                                                     nIterEMBV = nIterEMBV,
                                                     nCoresEMBV = nCoresEMBV,
                                                     clusteringForSelList = clusteringForSelList,
                                                     nClusterList = nClusterList,
                                                     nTopClusterList = nTopClusterList,
                                                     nTopEachList = nTopEachList,
                                                     nSelList = nSelList,
                                                     multiTraitsEvalMethodList = multiTraitsEvalMethodList,
                                                     hSelList = hSelList,
                                                     matingMethodVec = matingMethodVec,
                                                     allocateMethodVec =  rep("equalAllocation", nGenerationProceed),
                                                     weightedAllocationMethodList = rep(list("selectWBV"), nGenerationProceed),
                                                     traitNoRAList = rep(list(1), nGenerationProceed),
                                                     hList = rep(list(0), nGenerationProceed),
                                                     minimumUnitAllocateVec = minimumUnitAllocateVec,
                                                     includeGVPVec = rep(FALSE, nGenerationProceed),
                                                     targetGenGVPVec = targetGenGVPVec,
                                                     nNextPopVec = nNextPopVec,
                                                     performOCSVec = rep(FALSE, nGenerationProceed),
                                                     targetGenOCSVec = targetGenOCSVec,
                                                     HeStarRatioVec = HeStarRatioVec,
                                                     degreeOCSVec = degreeOCSVec,
                                                     includeGVPOCSVec = includeGVPOCSVec,
                                                     hOCSList = hOCSList,
                                                     weightedAllocationMethodOCSList = weightedAllocationMethodOCSList,
                                                     traitNoOCSList = traitNoOCSList,
                                                     nCrossesOCSVec = nCrossesOCSVec,
                                                     nameMethod = nameMethod,
                                                     nCores =  nCores,
                                                     overWriteRes = overWriteRes,
                                                     showProgress = showProgress,
                                                     returnMethod = returnMethod,
                                                     saveAllResAt = saveAllResAt,
                                                     evaluateGVMethod = evaluateGVMethod,
                                                     nTopEval = nTopEval,
                                                     traitNoEval = traitNoEval,
                                                     hEval = hEval,
                                                     summaryAllResAt = summaryAllResAt,
                                                     verbose = verbose)


        #### 2.3.1 Start simulation, summary results, and save simBs class object for equal allocation ####
        print("Start simulation....")

        checkFinishSimulation <- FALSE
        while (!checkFinishSimulation) {
          if (strategyName == "selectOPV") {
            mySimBsBVEqual$nCores <- nCoresWhenOPV
          }

          if (strategyName == "selectEMBV") {
            mySimBsBVEqual$nCores <- nCoresWhenEMBV
          }
          system.time(
            mySimBsBVEqual$startSimulation()
          )

          if (!is.null(saveAllResAt)) {
            saveAllResAtSplit <- stringr::str_split(string = list.files(saveAllResAt),
                                                    pattern = "_")
            saveAllResAtSplitLast <- lapply(X = saveAllResAtSplit,
                                            FUN = function(saveAllResAtSplitVec) {
                                              return(saveAllResAtSplitVec[length(saveAllResAtSplitVec)])
                                            })

            saveAllNumeric <- unique(sort(as.numeric(stringr::str_remove(saveAllResAtSplitLast, ".rds"))))

            checkFinishSimulation <- all(unlist(lapply(mySimBsBVEqual$trueGVMatList, length)) == (nGenerationProceed + 1)) |
              all((1:nIterSimulation) %in% saveAllNumeric)
          } else {
            checkFinishSimulation <- all(unlist(lapply(mySimBsBVEqual$trueGVMatList, length)) == (nGenerationProceed + 1))
          }
        }

        if (strategyName %in% c("selectOPV", "selectEMBV")) {
          mySimBsBVEqual$nCores <- nCores
        }

        if (all(unlist(lapply(mySimBsBVEqual$trueGVMatList, length)) == (nGenerationProceed + 1))) {
          mySimBsBVEqual$summaryResults()
        } else {
          print("Start summarization....")
          mySimBsBVEqual$summaryAllResAt <- saveAllResAt
          mySimBsBVEqual$nCores <- nCoresSummary
          mySimBsBVEqual$summaryResults()
          mySimBsBVEqual$nCores <- nCores
        }

        saveRDS(object = mySimBsBVEqual, file = fileNameSimBsEqual)


        rm(mySimBsBVEqual)
        gc(reset = TRUE)
      }
      ed <- Sys.time()

      print(paste0("Computation time required: ", round(ed - st, 2), " ",
                   attr(ed - st, "units")))




      ##### 2.4. Perform breeding simulations for optimized allocation with SToSOO #####
      print("Optimized allocation with StoSOO")

      st <- Sys.time()
      simBsNameStoSOO <- paste0(simBsName, "-ORA")
      fileNameSimBsStoSOO <- paste0(dirSimBs, scriptID, "_", simBsNameStoSOO, "_simBs.rds")




      if (!file.exists(fileNameSimBsStoSOO)) {

        if (selCriTypeName == "WBV") {
          selCriTypeNameSto <- "SimplestOpt"
        } else if (selCriTypeName == "WBV+GVP") {
          selCriTypeNameSto <- "IncludeGVPOpt"
        }

        if (nIterOptimization == 200) {
          nIterOptimizationSto <- 20000
        } else if (nIterOptimization == 100) {
          nIterOptimizationSto <- 5000
        }

        simBsNameSto <- paste0(simTypeName, "-", selectTypeName,
                               "-", selCriTypeNameSto, "-",
                               nGenerationProceed, "_Gen-50_BS-1_ME-",
                               nIterOptimizationSto, "_Eval-for-", traitName,
                               "-by-BRRSet-", nTrainingPopInit, "-", strategyName,
                               "-", lociEffMethod)
        dirSimBsSto <- paste0(dirTrainingPop, scriptIDSto, "_",
                              simBsNameSto, "/")
        fileNameOptimInfoSto <- paste0(dirSimBsSto, scriptIDSto, "_",
                                       simBsNameSto, "_simBs.rds")
        if (file.exists(fileNameOptimInfoSto)) {
          optimInfoSto <- readRDS(file = fileNameOptimInfoSto)
          hVecMat <- optimInfoSto$optimalHyperParamMatsList$Initial
          hVecSto0 <- hVecMat[nrow(hVecMat), ]
          hVecSto <- lowerBound + (upperBound - lowerBound) * hVecSto0
          hLensSto <- sapply(X = 1:nGenerationProceed,
                             FUN = function(genNow) {
                               length(traitNoRAList[[genNow]]) * (includeGVPVec[genNow] + length(weightedAllocationMethodList[[genNow]]))
                             }, simplify = TRUE)
          if (sameAcrossGeneration) {
            hListSto <- sapply(X = hLensSto,
                               FUN = function(hLen) {
                                 hVecSto[1:hLen]
                               }, simplify = FALSE)
          } else {
            hVecLenNow <- sum(hLensSto)
            hListSto <- split(x = hVecSto[1:hVecLenNow],
                              f = rep(1:nGenerationProceedSimulation,
                                      hLensSto[1:nGenerationProceedSimulation]))
          }



          mySimBsBVStoSOO <- myBreedSimulatR::simBs$new(simBsName = simBsNameConv,
                                                        bsInfoInit = myBsInfoInit,
                                                        breederInfoInit = myBreederInfoInit,
                                                        lociEffMethod = lociEffMethod,
                                                        trainingPopType = trainingPopType,
                                                        trainingPopInit = trainingPopInit,
                                                        trainingIndNamesInit = trainingIndNamesInit,
                                                        methodMLRInit = methodMLRInit,
                                                        multiTraitInit = multiTraitInit,
                                                        nIterSimulation = nIterSimulation,
                                                        nGenerationProceed = nGenerationProceed,
                                                        nRefreshMemoryEvery = nRefreshMemoryEvery,
                                                        updateBreederInfo = updateBreederInfo,
                                                        phenotypingInds = phenotypingInds,
                                                        nRepForPhenoInit = nRepForPhenoInit,
                                                        nRepForPheno = nRepForPheno,
                                                        updateModels = updateModels,
                                                        methodMLR = methodMLR,
                                                        multiTrait = multiTrait,
                                                        nSelectionWaysVec = nSelectionWaysVec,
                                                        selectionMethodList = selectionMethodList,
                                                        traitNoSelList = traitNoSelList,
                                                        blockSplitMethod = blockSplitMethod,
                                                        nMrkInBlock = nMrkInBlock,
                                                        minimumSegmentLength = minimumSegmentLength,
                                                        nSelInitOPVList = nSelInitOPVList,
                                                        nIterOPV = nIterOPV,
                                                        nProgeniesEMBVVec = nProgeniesEMBVVec,
                                                        nIterEMBV = nIterEMBV,
                                                        nCoresEMBV = nCoresEMBV,
                                                        clusteringForSelList = clusteringForSelList,
                                                        nClusterList = nClusterList,
                                                        nTopClusterList = nTopClusterList,
                                                        nTopEachList = nTopEachList,
                                                        nSelList = nSelList,
                                                        multiTraitsEvalMethodList = multiTraitsEvalMethodList,
                                                        hSelList = hSelList,
                                                        matingMethodVec = matingMethodVec,
                                                        allocateMethodVec = allocateMethodVec,
                                                        weightedAllocationMethodList = weightedAllocationMethodList,
                                                        traitNoRAList = traitNoRAList,
                                                        hList = hListSto,
                                                        minimumUnitAllocateVec = minimumUnitAllocateVec,
                                                        includeGVPVec = includeGVPVec,
                                                        targetGenGVPVec = targetGenGVPVec,
                                                        nNextPopVec = nNextPopVec,
                                                        performOCSVec = performOCSVec,
                                                        targetGenOCSVec = targetGenOCSVec,
                                                        HeStarRatioVec = HeStarRatioVec,
                                                        degreeOCSVec = degreeOCSVec,
                                                        includeGVPOCSVec = includeGVPOCSVec,
                                                        hOCSList = hOCSList,
                                                        weightedAllocationMethodOCSList = weightedAllocationMethodOCSList,
                                                        traitNoOCSList = traitNoOCSList,
                                                        nCrossesOCSVec = nCrossesOCSVec,
                                                        nameMethod = nameMethod,
                                                        nCores =  nCores,
                                                        overWriteRes = overWriteRes,
                                                        showProgress = showProgress,
                                                        returnMethod = returnMethod,
                                                        saveAllResAt = saveAllResAt,
                                                        evaluateGVMethod = evaluateGVMethod,
                                                        nTopEval = nTopEval,
                                                        traitNoEval = traitNoEval,
                                                        hEval = hEval,
                                                        summaryAllResAt = summaryAllResAt,
                                                        verbose = verbose)


          #### 2.4.1 Start simulation, summary results, and save simBs class object for StoSOO allocation ####
          print("Start simulation....")

          checkFinishSimulation <- FALSE
          while (!checkFinishSimulation) {
            if (strategyName == "selectOPV") {
              mySimBsBVStoSOO$nCores <- nCoresWhenOPV
            }

            if (strategyName == "selectEMBV") {
              mySimBsBVStoSOO$nCores <- nCoresWhenEMBV
            }
            system.time(
              mySimBsBVStoSOO$startSimulation()
            )

            if (!is.null(saveAllResAt)) {
              saveAllResAtSplit <- stringr::str_split(string = list.files(saveAllResAt),
                                                      pattern = "_")
              saveAllResAtSplitLast <- lapply(X = saveAllResAtSplit,
                                              FUN = function(saveAllResAtSplitVec) {
                                                return(saveAllResAtSplitVec[length(saveAllResAtSplitVec)])
                                              })

              saveAllNumeric <- unique(sort(as.numeric(stringr::str_remove(saveAllResAtSplitLast, ".rds"))))

              checkFinishSimulation <- all(unlist(lapply(mySimBsBVStoSOO$trueGVMatList, length)) == (nGenerationProceed + 1)) |
                all((1:nIterSimulation) %in% saveAllNumeric)
            } else {
              checkFinishSimulation <- all(unlist(lapply(mySimBsBVStoSOO$trueGVMatList, length)) == (nGenerationProceed + 1))
            }
          }

          if (strategyName %in% c("selectOPV", "selectEMBV")) {
            mySimBsBVStoSOO$nCores <- nCores
          }

          if (all(unlist(lapply(mySimBsBVStoSOO$trueGVMatList, length)) == (nGenerationProceed + 1))) {
            mySimBsBVStoSOO$summaryResults()
          } else {
            print("Start summarization....")
            mySimBsBVStoSOO$summaryAllResAt <- saveAllResAt
            mySimBsBVStoSOO$nCores <- nCoresSummary
            mySimBsBVStoSOO$summaryResults()
            mySimBsBVStoSOO$nCores <- nCores
          }

          saveRDS(object = mySimBsBVStoSOO, file = fileNameSimBsStoSOO)


          rm(mySimBsBVStoSOO)
          gc(reset = TRUE)
        } else {
          print("Not optimized by StoSOO yet...")
        }
        ed <- Sys.time()

        print(paste0("Computation time required: ", round(ed - st, 2), " ",
                     attr(ed - st, "units")))



      }


    }
  }
}
