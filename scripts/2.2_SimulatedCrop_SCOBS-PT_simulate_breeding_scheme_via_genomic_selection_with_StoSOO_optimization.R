############################################################################################################################
######  Title: 2.2_SimulatedCrop_SCOBS-PT_simulate_breeding_scheme_via_genomic_selection_with_StoSOO_optimization     ######
######  Author: Kosuke Hamazaki (hamazaki@ut-biomet.org)                                                              ######
######  Affiliation: Lab. of Biometry and Bioinformatics, The University of Tokyo                                     ######
######  Date: 2024/07/10 (Created), 2024/07/10 (Last Updated)                                                         ######
############################################################################################################################





###### 1. Settings ######
##### 1.0. Reset workspace ######
# rm(list=ls())



##### 1.1. Setting working directory to the "SCOBS" directory #####
cropName <- "SimulatedCrop"
project <- "SCOBS-PT"
os <- osVersion



scriptIDData <- "0.1"
scriptID <- "2.2"


##### 1.2. Setting some parameters #####
dirMidSCOBSBase <- "midstream/"


#### 1.2.1. Setting some parameters related to names of R6 class ####
scenarioNo <- 1
simName <- paste0("Scenario_", scenarioNo)
breederName <- "K. Hamazaki"

dirMidSCOBSScenario <- paste0(dirMidSCOBSBase, scriptIDData, "_", simName, "/")
if (!dir.exists(dirMidSCOBSScenario)) {
  dir.create(dirMidSCOBSScenario)
}


#### 1.2.2. Setting some parameters related to simulation ####
trialNo <- 1
trialName <- paste0("Trial_", trialNo)

dirMidSCOBSTrial <- paste0(dirMidSCOBSScenario, scriptIDData,
                           "_", project, "_", trialName, "/")
if (!dir.exists(dirMidSCOBSTrial)) {
  dir.create(dirMidSCOBSTrial)
}

nSimGeno <- 1
nSimPheno <- 1
# simPhenoNos <- 1
simPhenoNos <- 1:nSimPheno

nIterSimulation <- 10
nGenerationProceed <- 4
nGenerationProceedSimulation <- nGenerationProceed

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

# strategyNames <- c("selectBV", "selectWBV",
#                    "selectOHV", "selectOPV")
strategyNames <- c(
  # "selectBV"
  # ,
  "selectWBV"
  # ,
  # "selectOHV"
)

nTrainingPopInit <- "true"
# nTrainingPopInit <- "250"

lociEffMethods <- c("true", "estimated")
# lociEffMethods <- c("estimated")
trainingPopType <- "latest"
trainingPopInit <- 1
trainingIndNamesInit <- NULL


#### 1.2.3. Setting some parameters related to optimization of hyperparameters ####
setGoalAsFinalGeneration <- TRUE
performOptimization <- c(TRUE, rep(FALSE, nGenerationProceed - 1))
useFirstOptimizedValue <- TRUE
performRobustOptimization <- FALSE
lowerQuantile <- 0.25
sameAcrossGeneration <- FALSE
hMin <- 0
hMax <- 2
nIterSimulationForOneMrkEffect <- 50
nIterMrkEffectForRobustOptimization <- 1
nIterOptimization <- 5000
nIterSimulationPerEvaluation <- nIterSimulationForOneMrkEffect *
  nIterMrkEffectForRobustOptimization
nTotalIterForOneOptimization <- nIterOptimization * nIterSimulationPerEvaluation


simTypeName <- "RecurrentGS"
selectTypeName <- "SI2"  # "SI1"; "SI2"; "SI3"; "SI4"
optTypeName <- NULL  # NULL; "RobustOpt"
selCriTypeName <- "IncludeGVPOpt"  # "SimplestOpt"; "AllocateBVOpt";
# "IncludeGVPOpt"; "AllocateBVIncludeGVPOpt"; "IncludeTwoBVGVPOpt"

nTraits <- 1
traitName <- paste0("Trait_", 1:nTraits)
gpTypeName <- "BRRSet"    # "BRRSet"; "BayesBSet"
genName <- paste0(nGenerationProceed, "_Gen")

simBsNameBaseWoAllNIter <- paste(c(simTypeName, selectTypeName,
                                   optTypeName, selCriTypeName, genName),
                                 collapse = "-")
simBsNameBaseWoNIterOpt <- paste0(simBsNameBaseWoAllNIter, "-",
                                  nIterSimulationForOneMrkEffect, "_BS-",
                                  nIterMrkEffectForRobustOptimization, "_ME")


simBsNameBase <- paste0(simBsNameBaseWoNIterOpt, "-",
                        nIterOptimization, "_Eval-for-", traitName,
                        "-by-", gpTypeName)


nChildrenPerExpansion <- 3
nMaxEvalPerNode <- ceiling(x = nIterOptimization / (log(x = nIterOptimization) ^ 3))
maxDepth <- min(ceiling(x = sqrt(x = nIterOptimization / nMaxEvalPerNode)),
                floor(x = logb(x = 9e15, base = nChildrenPerExpansion)))
confidenceParam <- 1 / sqrt(x = nIterOptimization)
saveTreesAtBase <- "current_trees_backup"
# returnOptimalNodes <- seq(from = nIterOptimization / 100, to = nIterOptimization,
#                           by = nIterOptimization / 100)
# whenToSaveTrees <- seq(from = nIterOptimization / 100, to = nIterOptimization,
#                        by = nIterOptimization / 100)

nIterPerUpdate <- max(min(nIterOptimization / 100, 100), 1)
returnOptimalNodes <- seq(from = nIterPerUpdate, to = nIterOptimization,
                          by = nIterPerUpdate)
whenToSaveTrees <- seq(from = nIterPerUpdate, to = nIterOptimization,
                       by = nIterPerUpdate)


nTopEvalForOpt <- 5
rewardWeightVec <- c(rep(0, nGenerationProceedSimulation - 1), 1)
# discountedRate <- 0.8
# rewardWeightVec <- sapply(1:nGenerationProceedSimulation,
#                           function(genProceedNo) discountedRate ^ (genProceedNo - 1))
digitsEval <- 3



#### 1.2.4. Setting some parameters related to estimation of marker effects ####
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
}

alpha <- 0.5
nIter <- 20000
burnIn <- 5000
thin <- 5
bayesian <- TRUE


#### 1.2.5. Setting some parameters related to selection of parent candidates ####
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

if (selectTypeName == "SI1") {
  clusteringForSelList <- rep(list(FALSE), nGenerationProceed)
  nClusterList <- rep(list(1), nGenerationProceed)
  nTopClusterList <- rep(list(1), nGenerationProceed)
  nTopEachList <- rep(list(15), nGenerationProceed)
  nSelList <- rep(list(15), nGenerationProceed)
} else if (selectTypeName == "SI2") {
  clusteringForSelList <- rep(list(TRUE), nGenerationProceed)
  nClusterList <- rep(list(25), nGenerationProceed)
  nTopClusterList <- rep(list(25), nGenerationProceed)
  nTopEachList <- rep(list(1), nGenerationProceed)
  nSelList <- rep(list(25), nGenerationProceed)
} else if (selectTypeName == "SI3") {
  clusteringForSelList <- rep(list(TRUE), nGenerationProceed)
  nClusterList <- rep(list(10), nGenerationProceed)
  nTopClusterList <- rep(list(10), nGenerationProceed)
  nTopEachList <- rep(list(5), nGenerationProceed)
  nSelList <- rep(list(50), nGenerationProceed)
} else if (selectTypeName == "SI4") {
  clusteringForSelList <- rep(list(FALSE), nGenerationProceed)
  nClusterList <- rep(list(1), nGenerationProceed)
  nTopClusterList <- rep(list(1), nGenerationProceed)
  nTopEachList <- rep(list(50), nGenerationProceed)
  nSelList <- rep(list(50), nGenerationProceed)
}


multiTraitsEvalMethodList <- rep(list("sum"), nGenerationProceedSimulation)
hSelList <- rep(list(list(1)), nGenerationProceedSimulation)


#### 1.2.6. Setting some parameters related to mating and resource allocation ####
matingMethodVec <- rep("diallelWithSelfing", nGenerationProceedSimulation)
allocateMethodVec <- rep("weightedAllocation", nGenerationProceedSimulation)
traitNoRAList <- rep(list(1), nGenerationProceedSimulation)
hList <- rep(list(1), nGenerationProceedSimulation)
minimumUnitAllocateVec <- rep(5, nGenerationProceedSimulation)
if (!stringr::str_detect(string = simBsNameBase, pattern = "GVP")) {
  includeGVPVec <- rep(FALSE, nGenerationProceedSimulation)
} else {
  includeGVPVec <- rep(TRUE, nGenerationProceedSimulation)
}
nNextPopVec <- rep(250, nGenerationProceedSimulation)

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


#### 1.2.7. Setting some parameters related to mating and resource allocation ####
nCores <- 1
# nCoresPerOptimization <- 10
# nCoresPerOptimizationWhenOPV <- 10
# nCoresPerOptimizationWhenEMBV <- 5
# nCoresSummary <- 10

# nCoresPerOptimization <- 25
# nCoresPerOptimizationWhenOPV <- 25
# nCoresPerOptimizationWhenEMBV <- 17
# nCoresSummary <- 25

nCoresPerOptimization <- 50
nCoresPerOptimizationWhenOPV <- 50
nCoresPerOptimizationWhenEMBV <- 34
nCoresSummary <- 50

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


#### 1.2.8. Save parameters ####
fileParamsSCOBS <- paste0(dirMidSCOBSTrial, scriptID,
                          "_", project, "_", trialName,
                          "_", simBsNameBase,
                          "_all_parameters.RData")
save.image(fileParamsSCOBS)



##### 1.3. Import packages #####
require(myBreedSimulatR)
require(data.table)
require(MASS)
require(rrBLUP)
require(BGLR)
require(RAINBOWR)
require(ggplot2)
require(plotly)





##### 1.4. Project options #####
options(stringAsFactors = FALSE)


# simGenoNo <- 1
# simPhenoNo <- 1
# strategyName <- strategyNames[1]
for (simGenoNo in 1:nSimGeno) {
  genoName <- paste0("Geno_", simGenoNo)

  dirMidSCOBSGeno <- paste0(dirMidSCOBSTrial, scriptIDData,
                            "_", genoName, "/")


  for (simPhenoNo in simPhenoNos) {
    phenoName <- paste0("Pheno_", simPhenoNo)

    dirMidSCOBSPheno <- paste0(dirMidSCOBSGeno, scriptIDData,
                               "_", phenoName, "/")


    ###### 2. Load bsInfoInit & breederInfoInit & estimate marker effects ######
    simBsNameTrainingPop <- paste0(simBsNameBase, "-", nTrainingPopInit)

    dirMidSCOBSTrainingPop <- paste0(dirMidSCOBSPheno, scriptIDData,
                                     "_Model_with_", nTrainingPopInit, "/")
    if (!dir.exists(dirMidSCOBSTrainingPop)) {
      dir.create(dirMidSCOBSTrainingPop)
    }



    if (nTrainingPopInit == "true") {
      dirMidSCOBSTrainingPopFalse <- paste0(dirMidSCOBSPheno, scriptIDData,
                                            "_Model_with_", 250, "/")
      fileNameBsInfoInitRmInit <- paste0(dirMidSCOBSTrainingPopFalse, scriptIDData,
                                         "_bsInfoInit_remove_initial_population.rds")
      fileNameBreederInfoInitWithMrkEffRmInit <- paste0(dirMidSCOBSTrainingPopFalse, scriptIDData,
                                                        "_breederInfoInit_with_estimated_marker_effects_",
                                                        methodMLRInit, "_", 250,
                                                        "_individuals_remove_initial_population.rds")

      lociEffMethod <- lociEffMethods[1]
    } else {
      fileNameBsInfoInitRmInit <- paste0(dirMidSCOBSTrainingPop, scriptIDData,
                                         "_bsInfoInit_remove_initial_population.rds")
      fileNameBreederInfoInitWithMrkEffRmInit <- paste0(dirMidSCOBSTrainingPop, scriptIDData,
                                                        "_breederInfoInit_with_estimated_marker_effects_",
                                                        methodMLRInit, "_", nTrainingPopInit,
                                                        "_individuals_remove_initial_population.rds")

      lociEffMethod <- lociEffMethods[2]
    }
    myBsInfoInit <- readRDS(file = fileNameBsInfoInitRmInit)
    myBreederInfoInit <- readRDS(file = fileNameBreederInfoInitWithMrkEffRmInit)


    if (any(includeGVPVec)) {
      if (is.null(myBsInfoInit$lociInfo$recombBetweenMarkersList)) {
        myBsInfoInit$lociInfo$computeRecombBetweenMarkers()
        myBsInfoInit$traitInfo$lociInfo$computeRecombBetweenMarkers()
        myBsInfoInit$populations[[myBsInfoInit$generation]]$traitInfo$lociInfo$computeRecombBetweenMarkers()
      }
    }

    # mySimBsBVList <- list()
    for (strategyName in strategyNames) {
      print(paste0(genoName, "  ", phenoName, "  ",
                   strategyName, "  ", lociEffMethod))
      st <- Sys.time()
      simBsName <- paste0(simBsNameTrainingPop, "-", strategyName, "-", lociEffMethod)
      selectionMethodList <- rep(list(strategyName), nGenerationProceedSimulation)
      if (!any(stringr::str_detect(string = simBsNameBase, pattern = c("TwoBV", "AllocateBV", "AllocateWBV")))) {
        weightedAllocationMethodList <- rep(list(strategyName), nGenerationProceedSimulation)
      } else if (stringr::str_detect(string = simBsNameBase, pattern = "AllocateBV")) {
        weightedAllocationMethodList <- rep(list("selectBV"), nGenerationProceedSimulation)
      } else if (stringr::str_detect(string = simBsNameBase, pattern = "AllocateWBV")) {
        weightedAllocationMethodList <- rep(list("selectWBV"), nGenerationProceedSimulation)
      } else {
        weightedAllocationMethodList <- rep(list(strategyNames[1:2]), nGenerationProceedSimulation)
      }


      dirMidSCOBSBV <- paste0(dirMidSCOBSTrainingPop, scriptID, "_", simBsName, "/")
      dirMidSCOBSTree <- paste0(dirMidSCOBSBV, scriptID, "_",
                                saveTreesAtBase, "/")

      pastResFiles <- list.files(dirMidSCOBSTrainingPop)
      wherePastRes <- grep(pattern = simBsNameBaseWoNIterOpt,
                           x = pastResFiles)

      if (!dir.exists(dirMidSCOBSTree)) {
        dir.create(dirMidSCOBSTree, recursive = TRUE)
      }


      if (length(wherePastRes) >= 1) {
        pastResDirs <- pastResFiles[wherePastRes]

        nIterOptimizationsPastStr <- stringr::str_extract(string = pastResDirs,
                                                          pattern = "ME-.*_Eval")
        nIterOptimizationsPast <- abs(readr::parse_number(nIterOptimizationsPastStr))

        pastResDir <- paste0(dirMidSCOBSTrainingPop,
                             pastResDirs[which.max(nIterOptimizationsPast)], "/")

        pastResTreeDir <- paste0(pastResDir, scriptID, "_",
                                 saveTreesAtBase, "/")
        pastFilesToMoveRaw <- paste0(pastResTreeDir,
                                     list.files(pastResTreeDir))
        pastFilesToMoveRaw <- pastFilesToMoveRaw[grep(pattern = "Initial",
                                                      x = pastFilesToMoveRaw)]
        pastFilesToMove <- pastFilesToMoveRaw
        newFilesToMove <- stringr::str_replace_all(string = pastFilesToMove,
                                                   pattern = nIterOptimizationsPastStr[which.max(nIterOptimizationsPast)],
                                                   replacement = paste0("ME-", nIterOptimization, "_Eval"))


        file.copy(
          from = pastFilesToMove,
          to = newFilesToMove,
          overwrite = FALSE
        )

        if (max(nIterOptimizationsPast) > nIterOptimization) {
          fileNameOptimalParams <- newFilesToMove[grep(pattern = ".*.csv",
                                                       x = newFilesToMove)]
          optimalParams <- read.csv(file = fileNameOptimalParams, header = TRUE, row.names = 1)
          optimalIters <- readr::parse_number(rownames(optimalParams))

          newOptimalParams <- optimalParams[optimalIters <= nIterOptimization, ]

          write.csv(x = newOptimalParams, file = fileNameOptimalParams,
                    row.names = TRUE)
        }
      }

      saveAllResAt <- NULL
      summaryAllResAt <- NULL
      saveTreesAt <- paste0(dirMidSCOBSBV, scriptID, "_", saveTreesAtBase, "/")
      if (!dir.exists(saveTreesAt)) {
        dir.create(saveTreesAt)
      }
      saveTreeNameBase <- paste0(saveTreesAt, scriptID, "_", simBsName)


      fileNameSimBsBVNow <- paste0(dirMidSCOBSBV, scriptID, "_", simBsName, "_simBs.rds")

      if (!file.exists(fileNameSimBsBVNow)) {

        ###### 3. Simulate breeding scheme using simple GS with each BV ######
        ##### 3.1. Set simBs class object #####

        mySimBsBVNow <- myBreedSimulatR::simBsOpt$new(simBsName = simBsName,
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
                                                      nGenerationProceedSimulation = nGenerationProceedSimulation,
                                                      setGoalAsFinalGeneration = setGoalAsFinalGeneration,
                                                      performOptimization = performOptimization,
                                                      useFirstOptimizedValue = useFirstOptimizedValue,
                                                      performRobustOptimization = performRobustOptimization,
                                                      lowerQuantile = lowerQuantile,
                                                      sameAcrossGeneration = sameAcrossGeneration,
                                                      hMin = hMin,
                                                      hMax = hMax,
                                                      nTotalIterForOneOptimization = nTotalIterForOneOptimization,
                                                      nIterSimulationPerEvaluation = nIterSimulationPerEvaluation,
                                                      nIterSimulationForOneMrkEffect = nIterSimulationForOneMrkEffect,
                                                      nIterMrkEffectForRobustOptimization = nIterMrkEffectForRobustOptimization,
                                                      nIterOptimization = nIterOptimization,
                                                      nMaxEvalPerNode = nMaxEvalPerNode,
                                                      maxDepth = maxDepth,
                                                      nChildrenPerExpansion = nChildrenPerExpansion,
                                                      confidenceParam = confidenceParam,
                                                      returnOptimalNodes = returnOptimalNodes,
                                                      saveTreeNameBase = saveTreeNameBase,
                                                      whenToSaveTrees = whenToSaveTrees,
                                                      nTopEvalForOpt = nTopEvalForOpt,
                                                      rewardWeightVec = rewardWeightVec,
                                                      digitsEval = digitsEval,
                                                      nRefreshMemoryEvery = nRefreshMemoryEvery,
                                                      updateBreederInfo = updateBreederInfo,
                                                      phenotypingInds = phenotypingInds,
                                                      nRepForPhenoInit = nRepForPhenoInit,
                                                      nRepForPheno = nRepForPheno,
                                                      updateModels = updateModels,
                                                      updateBreederInfoSimulation = updateBreederInfoSimulation,
                                                      phenotypingIndsSimulation = phenotypingIndsSimulation,
                                                      nRepForPhenoSimulation = nRepForPhenoSimulation,
                                                      updateModelsSimulation = updateModelsSimulation,
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
                                                      minimumUnitAllocateVec = minimumUnitAllocateVec,
                                                      matingMethodVec = matingMethodVec,
                                                      allocateMethodVec = allocateMethodVec,
                                                      weightedAllocationMethodList = weightedAllocationMethodList,
                                                      traitNoRAList = traitNoRAList,
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
                                                      nCores = nCores,
                                                      nCoresPerOptimization = nCoresPerOptimization,
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


        ##### 3.2. Start simulation, summary results, and save simBs class object #####
        print("Start simulation....")

        if (strategyName == "selectOPV") {
          mySimBsBVNow$nCoresPerOptimization <- nCoresPerOptimizationWhenOPV
        }

        if (strategyName == "selectEMBV") {
          mySimBsBVNow$nCoresPerOptimization <- nCoresPerOptimizationWhenEMBV
        }
        system.time(
          mySimBsBVNow$startSimulation()
        )

        if (strategyName %in% c("selectOPV", "selectEMBV")) {
          mySimBsBVNow$nCoresPerOptimization <- nCoresPerOptimization
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



        saveRDS(object = mySimBsBVNow, file = fileNameSimBsBVNow)

        rm(mySimBsBVNow)
        gc(reset = TRUE)
        ed <- Sys.time()

        print(paste0("Computation time required: ", round(ed - st, 2), " ",
                     attr(ed - st, "units")))

      }
      gc(reset = TRUE)
    }
  }
}
