###################################################################################################################################################################
######  Title: 4.2_SimulatedCrop_SCOBS-PT_simulate_breeding_scheme_via_genomic_selection_with_simply_optimized_selection_criteria_plot_gradients_for_paper   ######
######  Author: Kosuke Hamazaki (hamazaki@ut-biomet.org)                                                                                                     ######
######  Affiliation: Lab. of Biometry and Bioinformatics, The University of Tokyo                                                                            ######
######  Date: 2024/09/05 (Created), 2024/09/09 (Last Updated)                                                                                                ######
###################################################################################################################################################################





###### 1. Settings ######
##### 1.0. Reset workspace ######
# rm(list=ls())



##### 1.1. Setting working directory to the "SCOBS" directory #####
cropName <- "SimulatedCrop"
project <- "SCOBS-PT"
os <- osVersion

scriptIDData <- "0.1"
scriptIDOpt0 <- "2.1"
scriptIDOpt <- "4.1"
scriptIDSto <- "2.2"
scriptIDConv <- "3.1"
scriptID <- "4.2"

overWriteRes <- FALSE



##### 1.2. Setting some parameters #####
dirMid <- "midstream/"


#### 1.2.1. Setting some parameters related to names of R6 class ####
scenarioNo <- 2
simName <- paste0("Scenario_", scenarioNo)
breederName <- "K. Hamazaki"

dirScenario <- paste0(dirMid, scriptIDData, "_", simName, "/")
if (!dir.exists(dirScenario)) {
  dir.create(dirScenario)
}


#### 1.2.2. Setting some parameters related to simulation ####
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




##### 1.4. Project options #####
options(stringAsFactors = FALSE)



##### 1.5 Setting some parameters related to optimization of hyperparameters #####
#### 1.5.1. Number of generations and simulations ####
# nIterSimulationForOneMrkEffect <- 50
# nIterMrkEffectForRobustOptimization <- 1
# nIterOptimizations <- c(20000, 5000)
simTypeName <- "RecurrentGS"



#### 1.2.4. Number of generations and simulations ####
simTypeName <- "RecurrentGS"

nGenerationProceeds <- 1:2
# nGenerationProceeds <- 4


# nGenerationProceed <- 1
for (nGenerationProceed in nGenerationProceeds) {
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

  nSimPheno <- 1
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
  selectTypeNames <- paste0("SI", c(1, 2, 4))

  # selectTypeName <- "SI1"      # str(args[3]): "SI1"; "SI2"; "SI3"; "SI4"
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

    if (nGenerationProceed == 1) {
      selCriTypeName <- "WBV+GVP"
    } else {
      selCriTypeName <- "WBV"
    }
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

    if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
      gvpInit <- "Zero"            # str(args[9]): "Mean", "Zero"
    } else {
      gvpInit <- "Mean"
    }


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
    # dirSimBs <- paste0(dirTrainingPop, scriptIDOpt, "_", simBsNameBase, "/")
    dirSimBs <- paste0(dirTrainingPop, scriptIDOpt, "_", simBsNameBase, "_check_gradient/")
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
    fileNameOptimInfoSol <- paste0(dirTrainingPop, scriptIDOpt0, "_", simBsNameBase, "/",
                                   scriptIDOpt0, "_current_results/",
                                   scriptIDOpt0, "_current_optimization_information.jb")
    fileNameOptimInfo <- paste0(dirSimBsMid, scriptIDOpt,
                                "_current_gradient_information.jb")



    if (file.exists(fileNameOptimInfo)) {
      st <- Sys.time()

      print("Gradient-based Optimized Resource Allocation")
      optimInfo <- joblib$load(filename = fileNameOptimInfo)
      logitHs <- sapply(X = 1:length(optimInfo$logitHTensorList),
                        FUN = function(iterNo) {
                          np$array(optimInfo$logitHTensorList[[iterNo]]$detach())
                        })
      hLists <- apply(X = lowerBound + (upperBound - lowerBound) / (1 + exp(-logitHs)),
                      MARGIN = 1, FUN = function(x) x, simplify = TRUE)
      logitHGrads <- sapply(X = 1:length(optimInfo$logitHTensorList),
                            FUN = function(iterNo) {
                              np$array(optimInfo$logitHTensorGradList[[iterNo]]$detach())
                            })
      hGradLists <- apply(X = lowerBound + (upperBound - lowerBound) / (1 + exp(-logitHGrads)),
                          MARGIN = 1, FUN = function(x) x, simplify = TRUE)
      outputs <- unlist(lapply(optimInfo$outputList, FUN = function(x) x$item()))
      logitHsMat <- t(logitHs)
      logitHsMatRo <- round(logitHsMat, 4)
      logitHGradsMat <- t(logitHGrads) * learningRate * (-1)



      nGrids <- 51
      # nGrids <- 26
      if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
        xLabel <- "Logit weights for WBV (Gen 0)"
        yLabel <- "Logit weights for GVP (Gen 0)"

        gridX <- seq(from = -2, to = 3, length = nGrids)
        gridY <- seq(from = -5, to = 0, length = nGrids)
      } else {
        xLabel <- "Logit weights for WBV (Gen 0)"
        yLabel <- "Logit weights for WBV (Gen 1)"

        gridX <- seq(from = -2, to = 3, length = nGrids)
        gridY <- seq(from = -2, to = 3, length = nGrids)
      }


      arrowMax <- sqrt(2) * 5 / (nGrids - 1)
      arrowLen0 <- sqrt(apply(logitHGradsMat ^ 2, 1, sum))
      arrowUnit <- arrowMax / max(arrowLen0)
      arrowStarts <- logitHsMat - logitHGradsMat * arrowUnit / 2
      arrowEnds <- logitHsMat + logitHGradsMat * arrowUnit / 2
      # xRange <- c(-2, 0)
      # yRange <- c(-2, 0)
      xRange <- range(gridX)
      yRange <- range(gridY)
      xRange[1] <- xRange[1] - max(logitHGradsMat * arrowUnit / 2)
      xRange[2] <- xRange[2] + max(logitHGradsMat * arrowUnit / 2)
      yRange[1] <- yRange[1] - max(logitHGradsMat * arrowUnit / 2)
      yRange[2] <- yRange[2] + max(logitHGradsMat * arrowUnit / 2)
      xRange2 <- xRange
      yRange2 <- yRange
      xRange2[1] <- xRange2[1] - 0.1
      xRange2[2] <- xRange2[2] + 0.1
      yRange2[1] <- yRange2[1] - 0.1
      yRange2[2] <- yRange2[2] + 0.1



      fileNameGra <- paste0(dirSimBs, scriptID, "_gradient_change_wide_range.png")
      png(filename = fileNameGra, width = 1100, height = 1000)
      op <- par(mar = c(8, 9.5, 4, 2),
                mgp = c(6, 2.5, 0))
      plot(logitHsMat, type = "n",
           xlim = xRange, ylim = yRange,
           xlab = xLabel,
           ylab = yLabel,
           cex.axis = 3,
           cex.lab = 4)
      # i <- 1
      for (i in 1:nrow(logitHsMat)) {
        arrows(x0 = arrowStarts[i, 1],
               y0 = arrowStarts[i, 2],
               x1 = arrowEnds[i, 1],
               y1 = arrowEnds[i, 2],
               lwd = 3,
               length = 0.1)
      }

      if (file.exists(fileNameOptimInfoSol)) {
        optimInfoSol <- joblib$load(filename = fileNameOptimInfoSol)
        logitHsSol <- sapply(X = 1:length(optimInfoSol$logitHTensorList),
                             FUN = function(iterNo) {
                               np$array(optimInfoSol$logitHTensorList[[iterNo]]$detach())
                             })
        if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
          logitHsSolFirst <- c(0, -log(-(upperBound - lowerBound) / lowerBound - 1))
        } else {
          logitHsSolFirst <- rep(0, nGenerationProceedSimulation)
        }

        logitHsSolLast <- logitHsSol[, ncol(logitHsSol)]

        points(x = logitHsSolFirst[1], y = logitHsSolFirst[2], col = 4, pch = 19, cex = 3)
        points(x = logitHsSolLast[1], y = logitHsSolLast[2], col = 3, pch = 19, cex = 3)
      }

      par(op)
      dev.off()


      # outputsMat <- matrix(data = outputs, nrow = nGrids, byrow = TRUE)
      outputsMat <- matrix(data = NA, nrow = nGrids, ncol = nGrids)
      # no <- 0
      for (i in 1:nGrids) {
        for (j in 1:nGrids) {
          # no <- no + 1
          whichOutPut <- which((logitHsMatRo[, 1] == round(gridX[i], 4)) &
                                 (logitHsMatRo[, 2] == round(gridY[j], 4)))

          if (length(whichOutPut) == 1) {
            # print(whichOutPut)
            outputsMat[i, j] <- outputs[whichOutPut]
          }
        }
      }


      fileNameOut <- paste0(dirSimBs, scriptID, "_outputs_change_wide_range.png")
      png(filename = fileNameOut, width = 1100, height = 1000)
      op <- par(mar = c(8, 9.5, 4, 2),
                mgp = c(6, 2.5, 0))
      image(x = gridX,
            y = gridY,
            z = outputsMat,
            xlim = xRange,
            ylim = yRange,
            xlab = xLabel,
            ylab = yLabel,
            cex.axis = 3,
            cex.lab = 4)
      par(op)
      dev.off()

      fileNameGraOut <- paste0(dirSimBs, scriptID, "_gradient_change_wide_range_with_outputs.png")
      png(filename = fileNameGraOut, width = 1100, height = 1000)
      op <- par(mar = c(8, 9.5, 4, 2),
                mgp = c(6, 2.5, 0))
      image(x = gridX,
            y = gridY,
            z = outputsMat,
            xlim = xRange2,
            ylim = yRange2,
            xlab = xLabel,
            ylab = yLabel,
            cex.axis = 3,
            cex.lab = 4)


      for (i in 1:nrow(logitHsMat)) {
        arrows(x0 = arrowStarts[i, 1],
               y0 = arrowStarts[i, 2],
               x1 = arrowEnds[i, 1],
               y1 = arrowEnds[i, 2],
               lwd = 3,
               length = 0.1)
      }

      if (file.exists(fileNameOptimInfoSol)) {
        optimInfoSol <- joblib$load(filename = fileNameOptimInfoSol)
        logitHsSol <- sapply(X = 1:length(optimInfoSol$logitHTensorList),
                             FUN = function(iterNo) {
                               np$array(optimInfoSol$logitHTensorList[[iterNo]]$detach())
                             })
        if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
          logitHsSolFirst <- c(0, -log(-(upperBound - lowerBound) / lowerBound - 1))
        } else {
          logitHsSolFirst <- rep(0, nGenerationProceedSimulation)
        }

        logitHsSolLast <- logitHsSol[, ncol(logitHsSol)]

        points(x = logitHsSolFirst[1], y = logitHsSolFirst[2], col = 4, pch = 19, cex = 3)
        points(x = logitHsSolLast[1], y = logitHsSolLast[2], col = 3, pch = 19, cex = 3)
      }
      par(op)
      dev.off()


      remained <- !apply(X = abs(round(logitHsMat * 5, 1)) %% 1 > 0,
            MARGIN = 1, FUN = any)

      logitHsMat <- logitHsMat[remained, ]
      logitHsMatRo <- logitHsMatRo[remained, ]
      logitHGradsMat <- logitHGradsMat[remained, ]
      outputs <- outputs[remained]

      # nGrids <- 51
      nGrids <- 26
      if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
        xLabel <- "Logit weights for WBV (Gen 0)"
        yLabel <- "Logit weights for GVP (Gen 0)"

        gridX <- seq(from = -2, to = 3, length = nGrids)
        gridY <- seq(from = -5, to = 0, length = nGrids)
      } else {
        xLabel <- "Logit weights for WBV (Gen 0)"
        yLabel <- "Logit weights for WBV (Gen 1)"

        gridX <- seq(from = -2, to = 3, length = nGrids)
        gridY <- seq(from = -2, to = 3, length = nGrids)
      }


      arrowMax <- sqrt(2) * 5 / (nGrids - 1)
      arrowLen0 <- sqrt(apply(logitHGradsMat ^ 2, 1, sum))
      arrowUnit <- arrowMax / max(arrowLen0)
      arrowStarts <- logitHsMat - logitHGradsMat * arrowUnit / 2
      arrowEnds <- logitHsMat + logitHGradsMat * arrowUnit / 2
      # xRange <- c(-2, 0)
      # yRange <- c(-2, 0)
      xRange <- range(gridX)
      yRange <- range(gridY)
      xRange[1] <- xRange[1] - max(logitHGradsMat * arrowUnit / 2)
      xRange[2] <- xRange[2] + max(logitHGradsMat * arrowUnit / 2)
      yRange[1] <- yRange[1] - max(logitHGradsMat * arrowUnit / 2)
      yRange[2] <- yRange[2] + max(logitHGradsMat * arrowUnit / 2)
      xRange2 <- xRange
      yRange2 <- yRange
      xRange2[1] <- xRange2[1] - 0.1
      xRange2[2] <- xRange2[2] + 0.1
      yRange2[1] <- yRange2[1] - 0.1
      yRange2[2] <- yRange2[2] + 0.1




      fileNameGra <- paste0(dirSimBs, scriptID, "_gradient_change_wide_range_2.png")
      png(filename = fileNameGra, width = 1100, height = 1000)
      op <- par(mar = c(8, 9.5, 4, 2),
                mgp = c(6, 2.5, 0))
      plot(logitHsMat, type = "n",
           xlim = xRange, ylim = yRange,
           xlab = xLabel,
           ylab = yLabel,
           cex.axis = 3,
           cex.lab = 4)
      # i <- 1
      for (i in 1:nrow(logitHsMat)) {
        arrows(x0 = arrowStarts[i, 1],
               y0 = arrowStarts[i, 2],
               x1 = arrowEnds[i, 1],
               y1 = arrowEnds[i, 2],
               lwd = 4,
               length = 0.1)
      }

      if (file.exists(fileNameOptimInfoSol)) {
        optimInfoSol <- joblib$load(filename = fileNameOptimInfoSol)
        logitHsSol <- sapply(X = 1:length(optimInfoSol$logitHTensorList),
                             FUN = function(iterNo) {
                               np$array(optimInfoSol$logitHTensorList[[iterNo]]$detach())
                             })
        if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
          logitHsSolFirst <- c(0, -log(-(upperBound - lowerBound) / lowerBound - 1))
        } else {
          logitHsSolFirst <- rep(0, nGenerationProceedSimulation)
        }

        logitHsSolLast <- logitHsSol[, ncol(logitHsSol)]

        points(x = logitHsSolFirst[1], y = logitHsSolFirst[2], col = 4, pch = 19, cex = 3)
        points(x = logitHsSolLast[1], y = logitHsSolLast[2], col = 3, pch = 19, cex = 3)
      }

      par(op)
      dev.off()


      # outputsMat <- matrix(data = outputs, nrow = nGrids, byrow = TRUE)
      outputsMat <- matrix(data = NA, nrow = nGrids, ncol = nGrids)
      # no <- 0
      for (i in 1:nGrids) {
        for (j in 1:nGrids) {
          # no <- no + 1
          whichOutPut <- which((logitHsMatRo[, 1] == round(gridX[i], 4)) &
                                 (logitHsMatRo[, 2] == round(gridY[j], 4)))

          if (length(whichOutPut) == 1) {
            # print(whichOutPut)
            outputsMat[i, j] <- outputs[whichOutPut]
          }
        }
      }


      fileNameOut <- paste0(dirSimBs, scriptID, "_outputs_change_wide_range_2.png")
      png(filename = fileNameOut, width = 1100, height = 1000)
      op <- par(mar = c(8, 9.5, 4, 2),
                mgp = c(6, 2.5, 0))
      image(x = gridX,
            y = gridY,
            z = outputsMat,
            xlim = xRange,
            ylim = yRange,
            xlab = xLabel,
            ylab = yLabel,
            cex.axis = 3,
            cex.lab = 4)
      par(op)
      dev.off()

      fileNameGraOut <- paste0(dirSimBs, scriptID, "_gradient_change_wide_range_with_outputs_2.png")
      png(filename = fileNameGraOut, width = 1100, height = 1000)
      op <- par(mar = c(8, 9.5, 4, 2),
                mgp = c(6, 2.5, 0))
      image(x = gridX,
            y = gridY,
            z = outputsMat,
            xlim = xRange2,
            ylim = yRange2,
            xlab = xLabel,
            ylab = yLabel,
            cex.axis = 3,
            cex.lab = 4)

      for (i in 1:nrow(logitHsMat)) {
        arrows(x0 = arrowStarts[i, 1],
               y0 = arrowStarts[i, 2],
               x1 = arrowEnds[i, 1],
               y1 = arrowEnds[i, 2],
               lwd = 4,
               length = 0.1)
      }

      if (file.exists(fileNameOptimInfoSol)) {
        optimInfoSol <- joblib$load(filename = fileNameOptimInfoSol)
        logitHsSol <- sapply(X = 1:length(optimInfoSol$logitHTensorList),
                             FUN = function(iterNo) {
                               np$array(optimInfoSol$logitHTensorList[[iterNo]]$detach())
                             })
        if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
          logitHsSolFirst <- c(0, -log(-(upperBound - lowerBound) / lowerBound - 1))
        } else {
          logitHsSolFirst <- rep(0, nGenerationProceedSimulation)
        }

        logitHsSolLast <- logitHsSol[, ncol(logitHsSol)]

        points(x = logitHsSolFirst[1], y = logitHsSolFirst[2], col = 4, pch = 19, cex = 3)
        points(x = logitHsSolLast[1], y = logitHsSolLast[2], col = 3, pch = 19, cex = 3)
      }
      par(op)
      dev.off()
    }
  }
}
