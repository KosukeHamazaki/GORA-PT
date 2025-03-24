###################################################################################################################################################################
######  Title: 3.2_SimulatedCrop_SCOBS-PT_simulate_breeding_scheme_via_genomic_selection_with_simply_optimized_selection_criteria_plot_figure_for_paper     ######
######  Author: Kosuke Hamazaki (hamazaki@ut-biomet.org)                                                                                                     ######
######  Affiliation: Lab. of Biometry and Bioinformatics, The University of Tokyo                                                                            ######
######  Date: 2024/07/16 (Created), 2024/07/16 (Last Updated)                                                                                                ######
###################################################################################################################################################################





###### 1. Settings ######
##### 1.0. Reset workspace ######
# rm(list=ls())



##### 1.1. Setting working directory to the "SCOBS" directory #####
cropName <- "SimulatedCrop"
project <- "SCOBS-PT"
os <- osVersion

scriptIDData <- "0.1"
scriptIDMainOpt <- "2.1"
scriptIDConv <- "3.1"
scriptID <- "3.2"

overWriteRes <- FALSE



##### 1.2. Setting some parameters #####
dirMid <- "midstream/"


#### 1.2.1. Setting some parameters related to names of R6 class ####
scenarioNo <- 1
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





##### 1.4. Project options #####
options(stringAsFactors = FALSE)



##### 1.5 Setting some parameters related to optimization of hyperparameters #####
#### 1.5.1. Number of generations and simulations ####
# nIterSimulationForOneMrkEffect <- 50
# nIterMrkEffectForRobustOptimization <- 1
# nIterOptimizations <- c(20000, 5000)
simTypeName <- "RecurrentGS"

# nGenerationProceeds <- 1:4
nGenerationProceeds <- 4

for (nGenerationProceed in nGenerationProceeds) {
  # nGenerationProceed <- 4      # int(args[1])
  nGenerationProceedSimulation <- nGenerationProceed
  nIterSimulationOpt <- 100       # int(args[2])
  nIterSimulation <- 10000

  #### 1.5.2. Traits ####
  nTraits <- 1
  traitNo <- 1
  traitName <- paste0("Trait_", traitNo)
  gpTypeName <- "TrueMrkEff"    # "BRRSet"; "BayesBSet"; "TrueMrkEff"

  strategyName <- "selectWBV"

  lociEffMethod <- "true"
  setGoalAsFinalGeneration <- TRUE
  sameAcrossGeneration <- FALSE


  #### 1.5.3. Parameter related to gradient-based optimization ####
  optimMethod <- "SGD"         # str(args[5]): "SGD", "Adam", "Adadelta"
  useScheduler <- "True"       # bool(args[6])

  learningRate <- 0.15         # float(args[7])
  betas <- c(0.1, 0.1)

  lowerBound <- -0.5
  upperBound <- 3.5


  #### 1.5.4. Changing Parameters ####
  if (nGenerationProceed == 4) {
    # nIterOptimizations <- c(200, 100)
    # nIterOptimizations <- 200
    nIterOptimizations <- 100
  } else {
    nIterOptimizations <- 200
  }
  selectTypeNames <- paste0("SI", c(1, 2, 4))
  selectTypeNames2 <- paste0("SI", c(1, 2, 3))
  selCriTypeNames0 <- c("WBV", "WBV+GVP")  # str(args[4]): "WBV"; "BV+WBV"; "WBV+GVP"; "BV+WBV+GVP"


  # nIterOptimization <- 100
  # nIterOptimization <- 200
  for (nIterOptimization in nIterOptimizations) {
    if (nIterOptimization == 200) {
      selCriTypeNames <- selCriTypeNames0
    } else {
      selCriTypeNames <- selCriTypeNames0[2]
    }

    traitName <- "Trait_1"

    geno1Name <- "Geno_1"
    dirGeno1 <- paste0(dirTrial, scriptIDData,
                       "_", geno1Name, "/")
    pheno1Name <- "Pheno_1"

    dirPheno1 <- paste0(dirGeno1, scriptIDData,
                        "_", pheno1Name, "/")

    dirTrainingPop1 <- paste0(dirPheno1, scriptIDData,
                              "_Model_with_true/")

    #### 1.5.4. Directory names ####
    fileNameGeneticGainList <- paste0(dirTrainingPop1, scriptID, "_",
                                      "all_change_in_genetic_gain_for_", 18,
                                      "_strategies_of_selection_intensity_", nGenerationProceed, "_Gen.rds")
    fileNameGeneticVarList <- paste0(dirTrainingPop1, scriptID, "_",
                                     "all_change_in_genetic_variance_for_", 18,
                                     "_strategies_of_selection_intensity_", nGenerationProceed, "_Gen.rds")
    fileNamePredAccList <- paste0(dirTrainingPop1, scriptID, "_",
                                  "all_change_in_prediction_accuracy_for_", 18,
                                  "_strategies_of_selection_intensity_", nGenerationProceed, "_Gen.rds")
    fileNameGeneticGainIterList <- paste0(dirTrainingPop1, scriptID, "_",
                                          "final_genetic_gain_of_10000_iterations_for_", 18,
                                          "_strategies_of_selection_intensity_", nGenerationProceed, "_Gen.rds")
    fileNameGeneticGainFinalTenList <- paste0(dirTrainingPop1, scriptID, "_",
                                              "genetic_gain_for_ten_traits_for_", 18,
                                              "_strategies_of_selection_intensity_", nGenerationProceed, "_Gen.rds")

    if (nIterOptimization == 100) {
      nSimPheno <- 10
      simPhenoNos <- 1:nSimPheno
      fileExists <- file.exists(fileNameGeneticGainFinalTenList)
    } else {
      nSimPheno <- 1
      simPhenoNos <- 1
      fileExists <- file.exists(fileNameGeneticGainList)
    }





    ###### 2. Start for loop for selectTypeNames ######
    if (overWriteRes | (!fileExists)) {
      geneticGainList <- list()
      geneticVarList <- list()
      predAccList <- list()
      geneticGainIterList <- list()
      geneticGainFinalTenList <- list()


      # simPhenoNo <- 1
      for (simPhenoNo in simPhenoNos) {
        phenoName <- paste0("Pheno_", simPhenoNo)
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



        #### 1.5.5. Changing parameters depending on strategies ####

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



            # if (!file.exists(fileNameSimEval)) {

            simBsList <- list()
            optEqNames <- c("GORA", "ORA", "EQ")

            for (optEqName in optEqNames) {
              if (optEqName == "GORA") {
                if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
                  gvpInits <- c("Mean", "Zero")
                } else {
                  gvpInits <- "Mean"
                }
              } else {
                if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
                  gvpInits <-  "Zero"
                } else {
                  gvpInits <- "Mean"
                }
              }
              # gvpInit <- "Zero"            # str(args[9]): "Mean", "Zero"
              for (gvpInit in gvpInits) {
                #### 1.5.6. Simulated breeding scheme name ####
                simBsNameBaseWoAllNIter <- paste0(simTypeName, "-", selectTypeName, "-",
                                                  selCriTypeName, "-", optimMethod, "_",
                                                  learningRate, "-Sche_", useScheduler, "-",
                                                  nGenerationProceed, "_Gen-InitGVP_", gvpInit)
                simBsNameBaseWoNIterOpt <- paste0(simBsNameBaseWoAllNIter, "-",
                                                  nIterSimulationOpt, "_BS")

                if (simPhenoNo == 1) {
                  simBsNameBaseDir <- paste0(simBsNameBaseWoNIterOpt, "-",
                                             200, "_Eval-for-",
                                             traitName, "-by-", gpTypeName)
                } else {
                  simBsNameBaseDir <- paste0(simBsNameBaseWoNIterOpt, "-",
                                             nIterOptimization, "_Eval-for-",
                                             traitName, "-by-", gpTypeName)
                }
                simBsNameBase <- paste0(simBsNameBaseWoNIterOpt, "-",
                                        nIterOptimization, "_Eval-for-",
                                        traitName, "-by-", gpTypeName)

                dirSCOBSBV <- paste0(dirTrainingPop, scriptIDMainOpt, "_", simBsNameBaseDir, "/")


                if (optEqName == "GORA") {
                  optEqNameNow <- paste0(optEqName, "-", stringr::str_sub(string = gvpInit, start = 1, end = 1))
                } else {
                  optEqNameNow <- optEqName
                }

                fileNameSimBsBVNow <- paste0(dirSCOBSBV, scriptIDConv, "_", simBsNameBase,
                                             "-", optEqName, "_simBs.rds")
                if (file.exists(fileNameSimBsBVNow)) {
                  simBsBVNow <- readRDS(file = fileNameSimBsBVNow)

                  simBsBVNow$simBsName <- optEqNameNow
                  simBsList[[optEqNameNow]] <- simBsBVNow
                }
              }
            }

            fileNameSimEval <- paste0(dirSCOBSBV, scriptID, "_",
                                      simBsNameBase, "_simEval.rds")
            mySimEval <- myBreedSimulatR::simEval$new(simEvalName = simBsNameBase,
                                                      simBsList = simBsList)
            saveRDS(object = mySimEval, file = fileNameSimEval)


            #   } else {
            #   mySimEval <- readRDS(file = fileNameSimEval)
            # }

            trueGVSummaryArrayList <- lapply(X = mySimEval$simBsList,
                                             FUN = function(simBsEach) {
                                               trueGVSummaryArray <- simBsEach$trueGVSummaryArray
                                               trueGVSummaryArrayNeeded <- trueGVSummaryArray[c(1, 5), , , ]
                                               trueGVSummaryArrayNeeded[1, , ] <-
                                                 (trueGVSummaryArrayNeeded[1, , ] - trueGVSummaryArray[2, , rep(1, nGenerationProceedSimulation + 1), ]) /
                                                 sqrt(trueGVSummaryArray[5, , rep(1, nGenerationProceedSimulation + 1), ])
                                               return(trueGVSummaryArrayNeeded)
                                             })


            #### 2.4.1. Optimized parameters #####
            optimizedParamsList <- lapply(X = mySimEval$simBsList,
                                          FUN = function(simBsEach) {
                                            hList <- simBsEach$hList
                                            optimizedParams <- do.call(what = rbind,
                                                                       args = hList)
                                            return(optimizedParams)
                                          })
            optimizedParamsList[[length(optimizedParamsList)]][] <- 0

            fileNameOptParams <- paste0(dirSCOBSBV, scriptID, "_", names(optimizedParamsList),
                                        "_optimized_parameteres.csv")

            saveOptimizedParams <- sapply(X = 1:length(optimizedParamsList),
                                          FUN = function(optimizedParamNo) {
                                            write.csv(x = optimizedParamsList[[optimizedParamNo]],
                                                      file = fileNameOptParams[optimizedParamNo])
                                          })



            ##### 2.5. Change over four generations #####
            #### 2.5.1. Change in genetic gain ####
            trueGVSummaryMeanList <- lapply(X = trueGVSummaryArrayList,
                                            FUN = function(trueGVSummaryArrayNeeded) {
                                              trueGVSummaryMean <- apply(X = trueGVSummaryArrayNeeded,
                                                                         MARGIN = c(1, 2), FUN = mean)

                                              return(trueGVSummaryMean)
                                            })
            geneticGainMat <- do.call(
              what = rbind,
              args = lapply(X = trueGVSummaryMeanList,
                            FUN = function(trueGVSummaryMean) {
                              trueGVSummaryMeanMax <- trueGVSummaryMean[1, ]
                              trueGVSummaryMeanGenGain <- trueGVSummaryMeanMax / trueGVSummaryMeanMax[1]

                              return(trueGVSummaryMeanGenGain)
                            })
            )


            if (nSimPheno > 1) {
              geneticGainFinalTenList[[selectTypeName]][[selCriTypeName]][[phenoName]] <- geneticGainMat[, ncol(geneticGainMat)]
            }


            if (nSimPheno == 1) {
              #### 2.5.2. Change in genetic variance ####
              geneticVarMat <- do.call(
                what = rbind,
                args = lapply(X = trueGVSummaryMeanList,
                              FUN = function(trueGVSummaryMean) {
                                trueGVSummaryMeanVar <- trueGVSummaryMean[2, ]

                                return(trueGVSummaryMeanVar / trueGVSummaryMeanVar[1])
                              })
              )


              #### 2.5.3. Change in prediction accuracy ####
              predAccMat <- do.call(
                what = rbind,
                args = lapply(X = mySimEval$simBsList,
                              FUN = function(simBsNow) {
                                corResIter <- sapply(X = 1:simBsNow$nIterSimulation,
                                                     FUN = function(simNo) {
                                                       trueGVMatListEach <- simBsNow$trueGVMatList[[simNo]]
                                                       estimatedGVMatListEach <- simBsNow$estimatedGVMatList[[simNo]]

                                                       corRes <- unlist(mapply(FUN = function(x1, x2) {
                                                         cor(x1, x2)[1, 1]
                                                       },
                                                       x1 = estimatedGVMatListEach,
                                                       x2 = trueGVMatListEach,
                                                       SIMPLIFY = FALSE))

                                                       return(corRes)
                                                     }, simplify = FALSE)

                                predAcc <- apply(
                                  X = do.call(what = rbind,
                                              args = corResIter),
                                  MARGIN = 2,
                                  FUN = mean,
                                  na.rm = TRUE
                                )


                                return(predAcc)
                              })
              )


              ##### 2.6. Evaluation in final generation #####
              #### 2.6.1. Evaluation by density ####
              geneticGainIter <- do.call(what = rbind,
                                         args = lapply(X = trueGVSummaryArrayList,
                                                       FUN = function(trueGVSummaryArrayNeeded) {
                                                         trueGVSummaryArrayNeeded[1, ncol(trueGVSummaryArrayNeeded), ] /
                                                           trueGVSummaryArrayNeeded[1, 1, ]
                                                       }))
            }




            if (nSimPheno == 1) {
              geneticGainList[[selectTypeName]][[selCriTypeName]] <- geneticGainMat
              geneticVarList[[selectTypeName]][[selCriTypeName]] <- geneticVarMat
              predAccList[[selectTypeName]][[selCriTypeName]] <- predAccMat
              geneticGainIterList[[selectTypeName]][[selCriTypeName]] <- geneticGainIter
            }


          }
        }

      }
    } else {
      if (nSimPheno == 1) {
        geneticGainList <- readRDS(file = fileNameGeneticGainList)
        geneticVarList <- readRDS(file = fileNameGeneticVarList)
        predAccList <- readRDS(file = fileNamePredAccList)
        geneticGainIterList <- readRDS(file = fileNameGeneticGainIterList)
      } else {
        geneticGainFinalTenList <- readRDS(file = fileNameGeneticGainFinalTenList)
      }
    }


    if (nSimPheno == 1) {
      # selectTypeCols <- c("grey40", "goldenrod1", "blue", "violetred",
      #                     "goldenrod1", "blue", "violetred")
      # selectTypeCols <- c("goldenrod1", "blue", "violetred",
      #                     "goldenrod4", "blue4", "violetred4",
      #                     "yellow", "lightblue", "rosybrown1")
      selectTypeCols20 <- c("#cc99ff", "#660099", "#0066cc", "#333333")
      selectTypeLtys <- c(4, 4, 4, 2, 2, 2, 1, 1, 1, 3, 3, 3)
      selectTypeLtys20 <- rep(c(4, 2, 1, 3), 3)
      selectTypeLwds <- 1.75


      names(geneticGainList) <- names(geneticGainIterList) <-
        names(geneticVarList) <- names(predAccList) <- selectTypeNames2


      if (overWriteRes | (!fileExists)) {
        saveRDS(object = geneticGainList,
                file = fileNameGeneticGainList)
        saveRDS(object = geneticVarList,
                file = fileNameGeneticVarList)
        saveRDS(object = predAccList,
                file = fileNamePredAccList)
        saveRDS(object = geneticGainIterList,
                file = fileNameGeneticGainIterList)
      }

    } else {
      # selectTypeCols <- c("grey80", "yellow", "lightblue", "lightgreen",
      #                           "orange1", "mediumpurple1", "rosybrown1")
      # selectTypeCols <- c("yellow", "lightblue", "rosybrown1",
      #                     "yellow", "lightblue", "rosybrown1",
      #                     "yellow", "lightblue", "rosybrown1")
      selectTypeCols0 <- c("#cc99ff", "#cc99ff", "#00ffcc", "#999999")


      names(geneticGainFinalTenList) <- selectTypeNames2

      if (overWriteRes | (!fileExists)) {
        saveRDS(object = geneticGainFinalTenList,
                file = fileNameGeneticGainFinalTenList)
      }
    }

    ##### 3. Draw plots for figures #####
    ##### 3.1. Some definitions #####
    for (selCriTypeName in selCriTypeNames) {
      if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {

        if (nSimPheno == 1) {
          optEqNamesNow <- c("GORA1", "GORA2", "ORA", "EQ")
          selectTypeCols2 <- selectTypeCols20
          selectTypeLtys2 <- selectTypeLtys20
        } else {
          optEqNamesNow <- c("GORA2", "ORA", "EQ")
          selectTypeCols <- selectTypeCols0[-2]
        }
      } else {
        optEqNamesNow <- c("GORA1", "EQ")

        if (nSimPheno == 1) {
          selectTypeCols2 <- selectTypeCols20[c(2, 4)]
          selectTypeLtys2 <- selectTypeLtys20[c(2, 4, 6, 8, 10, 12)]
        } else {
          selectTypeCols <- selectTypeCols0[c(2, 4)]
        }
      }

      strategyNamesAbbAllOrd <- paste0(rep(optEqNamesNow, each = length(selectTypeNames2)),
                                       "-", rep(selectTypeNames2, length(optEqNamesNow)))
      genNos <- paste0(0:nGenerationProceed)



      ##### 3.2. Results for one trait #####
      if (nSimPheno == 1) {
        geneticGainListNow <- lapply(X = geneticGainList,
                                     FUN = function(x) {
                                       x[[selCriTypeName]]
                                     })
        geneticVarListNow <- lapply(X = geneticVarList,
                                    FUN = function(x) {
                                      x[[selCriTypeName]]
                                    })
        predAccListNow <- lapply(X = predAccList,
                                 FUN = function(x) {
                                   x[[selCriTypeName]]
                                 })
        geneticGainIterListNow <- lapply(X = geneticGainIterList,
                                         FUN = function(x) {
                                           x[[selCriTypeName]]
                                         })


        #### 3.2.1. Change in genetic gain ####
        if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
          geneticGainGradOpt1Mat <- do.call(
            what = rbind,
            args = lapply(X = geneticGainListNow,
                          FUN = function(geneticGainStr) {
                            geneticGainStr[(nrow(geneticGainStr) - 3), ]
                          })
          )
          geneticGainGradOpt2Mat <- do.call(
            what = rbind,
            args = lapply(X = geneticGainListNow,
                          FUN = function(geneticGainStr) {
                            geneticGainStr[(nrow(geneticGainStr) - 2), ]
                          })
          )
          geneticGainStoOptMat <- do.call(
            what = rbind,
            args = lapply(X = geneticGainListNow,
                          FUN = function(geneticGainStr) {
                            geneticGainStr[(nrow(geneticGainStr) - 1), ]
                          })
          )
          geneticGainEqMat <- do.call(
            what = rbind,
            args = lapply(X = geneticGainListNow,
                          FUN = function(geneticGainStr) {
                            geneticGainStr[nrow(geneticGainStr), ]
                          })
          )
          geneticGainMatAll <- rbind(geneticGainGradOpt1Mat, geneticGainGradOpt2Mat,
                                     geneticGainStoOptMat, geneticGainEqMat)
        } else {
          geneticGainGradOpt1Mat <- do.call(
            what = rbind,
            args = lapply(X = geneticGainListNow,
                          FUN = function(geneticGainStr) {
                            geneticGainStr[(nrow(geneticGainStr) - 1), ]
                          })
          )

          geneticGainEqMat <- do.call(
            what = rbind,
            args = lapply(X = geneticGainListNow,
                          FUN = function(geneticGainStr) {
                            geneticGainStr[nrow(geneticGainStr), ]
                          })
          )

          geneticGainMatAll <- rbind(geneticGainGradOpt1Mat,
                                     geneticGainEqMat)
        }

        rownames(geneticGainMatAll) <- strategyNamesAbbAllOrd
        geneticGainMatAllOrd <- geneticGainMatAll

        # geneticGainDf <- data.frame(Strategy = rep(strategyNamesAbbAllOrd,
        #                                            ncol(geneticGainMatAllOrd)),
        #                             Population = rep(colnames(geneticGainMatAllOrd),
        #                                              each = nrow(geneticGainMatAllOrd)),
        #                             Value = c(geneticGainMatAllOrd))
        # geneticGainDf$Strategy <- factor(geneticGainDf$Strategy,
        #                                  levels = strategyNamesAbbAllOrd)
        # geneticGainDf$Population <- factor(geneticGainDf$Population,
        #                                    labels = paste0("Population-", 1:5))
        # geneticGainDf$Population <- factor(geneticGainDf$Population,
        #                                    labels = genNos)

        # pltChangeGain <- ggplot2::ggplot(data = geneticGainDf) +
        #   ggplot2::geom_line(mapping = aes(x = Population, y = Value,
        #                                    group = Strategy,
        #                                    color = Strategy),
        #                      lty = rep(selectTypeLtys, each = 5),
        #                      lwd = selectTypeLwds) +
        #   ggplot2::scale_color_manual(values = selectTypeCols) +
        #   ggplot2::theme(text = element_text(size = 96)) +
        #   ggplot2::xlab(label = "Generation number") +
        #   ggplot2::ylab(label = "Genetic gain") +
        #   ggplot2::ylim(c(1, 3.1))
        #
        # fileNamePltChangeGain <- paste0(dirMidSCOBSTrainingPop1, scriptID, "_",
        #                                 "change_in_genetic_gain_over_four_generations_for_9_strategies.png")
        # # png(filename = fileNamePltChangeGain, width = 1800, height = 1100)
        # png(filename = fileNamePltChangeGain, width = 1600, height = 1600)
        # print(pltChangeGain)
        # dev.off()
        #


        strategyNamesAbbAllOrdSplit <- stringr::str_split(string = strategyNamesAbbAllOrd, pattern = "-")
        allocateStrategies <- unlist(lapply(X = strategyNamesAbbAllOrdSplit, FUN = function(x) x[1]))
        selectionInts <- unlist(lapply(X = strategyNamesAbbAllOrdSplit, FUN = function(x) x[2]))


        geneticGainSplitDf <- data.frame(Strategy = rep(allocateStrategies,
                                                        ncol(geneticGainMatAllOrd)),
                                         SI = rep(selectionInts, ncol(geneticGainMatAllOrd)),
                                         Population = rep(colnames(geneticGainMatAllOrd),
                                                          each = nrow(geneticGainMatAllOrd)),
                                         Value = c(geneticGainMatAllOrd))

        geneticGainSplitDf$Strategy <- factor(geneticGainSplitDf$Strategy,
                                              levels = optEqNamesNow)
        # geneticGainSplitDf$Population <- factor(geneticGainSplitDf$Population,
        #                                    labels = paste0("Population-", 1:5))
        geneticGainSplitDf$Population <- factor(geneticGainSplitDf$Population,
                                                labels = genNos)

        pltChangeGainSplit <- ggplot2::ggplot(data = geneticGainSplitDf) +
          ggplot2::geom_line(mapping = aes(x = Population, y = Value,
                                           group = Strategy,
                                           color = Strategy),
                             lty = rep(selectTypeLtys2, each = nGenerationProceed + 1),
                             lwd = selectTypeLwds) +
          ggplot2::theme_bw() +
          ggplot2::facet_wrap(~ SI) +
          ggplot2::scale_color_manual(values = selectTypeCols2) +
          ggplot2::theme(text = element_text(size = 96)) +
          ggplot2::xlab(label = "Generation number") +
          ggplot2::ylab(label = "Genetic gain") +
          ggplot2::ylim(c(1, 3.5))

        fileNamePltChangeGainSplit <- paste0(dirTrainingPop1, scriptID, "_",
                                             "change_in_genetic_gain_over_four_generations_for_", length(optEqNamesNow),
                                             "_strategies_SI_split_", nGenerationProceed, "_Gen_", selCriTypeName, ".png")
        # png(filename = fileNamePltChangeGain, width = 1800, height = 1100)
        png(filename = fileNamePltChangeGainSplit, width = 2500, height = 1000)
        print(pltChangeGainSplit)
        dev.off()


        #### 3.2.2. Change in genetic variance ####
        if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
          geneticVarGradOpt1Mat <- do.call(
            what = rbind,
            args = lapply(X = geneticVarListNow,
                          FUN = function(geneticVarStr) {
                            geneticVarStr[(nrow(geneticVarStr) - 3), ]
                          })
          )
          geneticVarGradOpt2Mat <- do.call(
            what = rbind,
            args = lapply(X = geneticVarListNow,
                          FUN = function(geneticVarStr) {
                            geneticVarStr[(nrow(geneticVarStr) - 2), ]
                          })
          )
          geneticVarStoOptMat <- do.call(
            what = rbind,
            args = lapply(X = geneticVarListNow,
                          FUN = function(geneticVarStr) {
                            geneticVarStr[(nrow(geneticVarStr) - 1), ]
                          })
          )
          geneticVarEqMat <- do.call(
            what = rbind,
            args = lapply(X = geneticVarListNow,
                          FUN = function(geneticVarStr) {
                            geneticVarStr[nrow(geneticVarStr), ]
                          })
          )
          geneticVarMatAll <- rbind(geneticVarGradOpt1Mat, geneticVarGradOpt2Mat,
                                    geneticVarStoOptMat, geneticVarEqMat)
        } else {
          geneticVarGradOpt1Mat <- do.call(
            what = rbind,
            args = lapply(X = geneticVarListNow,
                          FUN = function(geneticVarStr) {
                            geneticVarStr[(nrow(geneticVarStr) - 1), ]
                          })
          )

          geneticVarEqMat <- do.call(
            what = rbind,
            args = lapply(X = geneticVarListNow,
                          FUN = function(geneticVarStr) {
                            geneticVarStr[nrow(geneticVarStr), ]
                          })
          )
          geneticVarMatAll <- rbind(geneticVarGradOpt1Mat,
                                    geneticVarEqMat)

        }
        rownames(geneticVarMatAll) <- strategyNamesAbbAllOrd
        geneticVarMatAllOrd <- geneticVarMatAll

        # geneticVarDf <- data.frame(Strategy = rep(strategyNamesAbbAllOrd,
        #                                           ncol(geneticVarMatAllOrd)),
        #                            Population = rep(colnames(geneticVarMatAllOrd),
        #                                             each = nrow(geneticVarMatAllOrd)),
        #                            Value = c(geneticVarMatAllOrd))
        # geneticVarDf$Strategy <- factor(geneticVarDf$Strategy,
        #                                 levels = strategyNamesAbbAllOrd)
        # # geneticVarDf$Population <- factor(geneticVarDf$Population,
        # #                                   labels = paste0("Population-", 1:5))
        # geneticVarDf$Population <- factor(geneticVarDf$Population,
        #                                   labels = genNos)
        #
        # pltChangeVar <- ggplot2::ggplot(data = geneticVarDf) +
        #   ggplot2::geom_line(mapping = aes(x = Population, y = Value,
        #                                    group = Strategy,
        #                                    color = Strategy),
        #                      lty = rep(selectTypeLtys, each = 5),
        #                      lwd = selectTypeLwds) +
        #   ggplot2::scale_color_manual(values = selectTypeCols) +
        #   ggplot2::theme(text = element_text(size = 96)) +
        #   ggplot2::xlab(label = "Generation number") +
        #   ggplot2::ylab(label = "Genetic variance") +
        #   ggplot2::ylim(c(0.6, 2.3))
        #
        # fileNamePltChangeVar <- paste0(dirMidSCOBSTrainingPop1, scriptID, "_",
        #                                "change_in_genetic_variance_over_four_generations_for_9_strategies.png")
        # # png(filename = fileNamePltChangeVar, width = 1800, height = 1100)
        # png(filename = fileNamePltChangeVar, width = 1600, height = 1600)
        # print(pltChangeVar)
        # dev.off()



        geneticVarSplitDf <- data.frame(Strategy = rep(allocateStrategies,
                                                       ncol(geneticVarMatAllOrd)),
                                        SI = rep(selectionInts, ncol(geneticVarMatAllOrd)),
                                        Population = rep(colnames(geneticVarMatAllOrd),
                                                         each = nrow(geneticVarMatAllOrd)),
                                        Value = c(geneticVarMatAllOrd))

        geneticVarSplitDf$Strategy <- factor(geneticVarSplitDf$Strategy,
                                             levels = optEqNamesNow)
        # geneticVarSplitDf$Population <- factor(geneticVarSplitDf$Population,
        #                                    labels = paste0("Population-", 1:5))
        geneticVarSplitDf$Population <- factor(geneticVarSplitDf$Population,
                                               labels = genNos)

        pltChangeVarSplit <- ggplot2::ggplot(data = geneticVarSplitDf) +
          ggplot2::geom_line(mapping = aes(x = Population, y = Value,
                                           group = Strategy,
                                           color = Strategy),
                             lty = rep(selectTypeLtys2, each = nGenerationProceed + 1),
                             lwd = selectTypeLwds) +
          ggplot2::theme_bw() +
          ggplot2::facet_wrap(~ SI) +
          ggplot2::scale_color_manual(values = selectTypeCols2) +
          ggplot2::theme(text = element_text(size = 96)) +
          ggplot2::xlab(label = "Generation number") +
          ggplot2::ylab(label = "Genetic variance") +
          ggplot2::ylim(c(0.1, 1.4))

        fileNamePltChangeVarSplit <- paste0(dirTrainingPop1, scriptID, "_",
                                            "change_in_genetic_variance_over_four_generations_for_", length(optEqNamesNow),
                                            "_strategies_SI_split_", nGenerationProceed, "_Gen_", selCriTypeName, ".png")
        # png(filename = fileNamePltChangeVar, width = 1800, height = 1100)
        png(filename = fileNamePltChangeVarSplit, width = 2500, height = 1000)
        print(pltChangeVarSplit)
        dev.off()


        # #### 3.2.3. Change in prediction accuracy ####
        # predAccGradOptMat <- do.call(
        #   what = rbind,
        #   args = lapply(X = predAccListNow,
        #                 FUN = function(predAccStr) {
        #                   predAccStr[(nrow(predAccStr) - 2), ]
        #                 })
        # )
        # predAccEqMat <- do.call(
        #   what = rbind,
        #   args = lapply(X = predAccListNow,
        #                 FUN = function(predAccStr) {
        #                   predAccStr[(nrow(predAccStr) - 1), ]
        #                 })
        # )
        # predAccStoOptMat <- do.call(
        #   what = rbind,
        #   args = lapply(X = predAccListNow,
        #                 FUN = function(predAccStr) {
        #                   predAccStr[nrow(predAccStr), ]
        #                 })
        # )
        # predAccMatAll <- rbind(predAccGradOptMat, predAccEqMat, predAccStoOptMat)
        # rownames(predAccMatAll) <- strategyNamesAbbAllOrd
        # predAccMatAllOrd <- predAccMatAll
        #
        # predAccDf <- data.frame(Strategy = rep(strategyNamesAbbAllOrd,
        #                                        ncol(predAccMatAllOrd)),
        #                         Population = rep(colnames(predAccMatAllOrd),
        #                                          each = nrow(predAccMatAllOrd)),
        #                         Value = c(predAccMatAllOrd))
        # predAccDf$Strategy <- factor(predAccDf$Strategy,
        #                              levels = strategyNamesAbbAllOrd)
        # # predAccDf$Population <- factor(predAccDf$Population,
        # #                                   labels = paste0("Population-", 1:5))
        # predAccDf$Population <- factor(predAccDf$Population,
        #                                labels = genNos)
        #
        # pltPredAcc <- ggplot2::ggplot(data = predAccDf) +
        #   ggplot2::geom_line(mapping = aes(x = Population, y = Value,
        #                                    group = Strategy,
        #                                    color = Strategy),
        #                      lty = rep(selectTypeLtys, each = 5),
        #                      lwd = selectTypeLwds) +
        #   ggplot2::scale_color_manual(values = selectTypeCols) +
        #   ggplot2::theme(text = element_text(size = 96)) +
        #   ggplot2::xlab(label = "Generation number") +
        #   ggplot2::ylab(label = "Prediction accuracy") +
        #   ggplot2::ylim(c(-0.05, 1))
        #
        # fileNamePltPredAcc <- paste0(dirMidSCOBSTrainingPop1, scriptID, "_",
        #                              "change_in_prediction_accuracy_over_four_generations_for_9_strategies.png")
        # # png(filename = fileNamePltPredAcc, width = 1800, height = 1100)
        # png(filename = fileNamePltPredAcc, width = 1600, height = 1600)
        # print(pltPredAcc)
        # dev.off()
        #
        #
        #
        # predAccSplitDf <- data.frame(Strategy = rep(allocateStrategies,
        #                                             ncol(predAccMatAllOrd)),
        #                              SI = rep(selectionInts, ncol(predAccMatAllOrd)),
        #                              Population = rep(colnames(predAccMatAllOrd),
        #                                               each = nrow(predAccMatAllOrd)),
        #                              Value = c(predAccMatAllOrd))
        #
        # predAccSplitDf$Strategy <- factor(predAccSplitDf$Strategy,
        #                                   levels = optEqNamesNow)
        # # predAccSplitDf$Population <- factor(predAccSplitDf$Population,
        # #                                    labels = paste0("Population-", 1:5))
        # predAccSplitDf$Population <- factor(predAccSplitDf$Population,
        #                                     labels = genNos)
        #
        # pltChangePredAccSplit <- ggplot2::ggplot(data = predAccSplitDf) +
        #   ggplot2::geom_line(mapping = aes(x = Population, y = Value,
        #                                    group = Strategy,
        #                                    color = Strategy),
        #                      lty = rep(selectTypeLtys2, each = 5),
        #                      lwd = selectTypeLwds) +
        #   ggplot2::facet_wrap(~ SI) +
        #   ggplot2::scale_color_manual(values = selectTypeCols2) +
        #   ggplot2::theme(text = element_text(size = 96)) +
        #   ggplot2::xlab(label = "Generation number") +
        #   ggplot2::ylab(label = "Prediction accuracy") +
        #   ggplot2::ylim(c(-0.05, 1))
        #
        # fileNamePltChangePredAccSplit <- paste0(dirMidSCOBSTrainingPop1, scriptID, "_",
        #                                         "change_in_prediction_accuracy_over_four_generations_for_9_strategies_SI_split.png")
        # # png(filename = fileNamePltChangeVar, width = 1800, height = 1100)
        # png(filename = fileNamePltChangePredAccSplit, width = 2500, height = 1000)
        # print(pltChangePredAccSplit)
        # dev.off()



        #### 3.2.4. CDF for final genetic gain ####
        if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
          geneticGainIterOpt1List <- lapply(
            X = geneticGainIterListNow,
            FUN = function(geneticGainIter) {
              geneticGainIter[(nrow(geneticGainIter) - 3), ]
            }
          )

          geneticGainIterOpt2List <- lapply(
            X = geneticGainIterListNow,
            FUN = function(geneticGainIter) {
              geneticGainIter[(nrow(geneticGainIter) - 2), ]
            }
          )

          geneticGainIterStoOptList <- lapply(
            X = geneticGainIterListNow,
            FUN = function(geneticGainIter) {
              geneticGainIter[(nrow(geneticGainIter) - 1), ]
            }
          )
          geneticGainIterEqList <- lapply(
            X = geneticGainIterListNow,
            FUN = function(geneticGainIter) {
              geneticGainIter[nrow(geneticGainIter), ]
            }
          )



          geneticGainIterAllList <- c(
            geneticGainIterOpt1List,
            geneticGainIterOpt2List,
            geneticGainIterStoOptList,
            geneticGainIterEqList
          )
        } else {
          geneticGainIterOpt1List <- lapply(
            X = geneticGainIterListNow,
            FUN = function(geneticGainIter) {
              geneticGainIter[(nrow(geneticGainIter) - 1), ]
            }
          )

          geneticGainIterEqList <- lapply(
            X = geneticGainIterListNow,
            FUN = function(geneticGainIter) {
              geneticGainIter[nrow(geneticGainIter), ]
            }
          )

          geneticGainIterAllList <- c(
            geneticGainIterOpt1List,
            geneticGainIterEqList
          )
        }
        names(geneticGainIterAllList) <- strategyNamesAbbAllOrd
        geneticGainIterAllOrdList <- geneticGainIterAllList

        # geneticGainIterDf <- data.frame(
        #   Strategy = rep(strategyNamesAbbAllOrd,
        #                  lapply(X = geneticGainIterAllOrdList,
        #                         FUN = length)),
        #   Value = unlist(geneticGainIterAllOrdList)
        # )
        # geneticGainIterDf$Strategy <- factor(geneticGainIterDf$Strategy,
        #                                      levels = strategyNamesAbbAllOrd)
        # pltCdf <- ggplot2::ggplot(data = geneticGainIterDf,
        #                           aes(x = Value, group = Strategy,
        #                               color = Strategy, linetype = Strategy)) +
        #   ggplot2::stat_ecdf(pad = TRUE,
        #                      lwd = selectTypeLwds) +
        #   ggplot2::scale_linetype_manual(breaks = strategyNamesAbbAllOrd,
        #                                  values = selectTypeLtys) +
        #   ggplot2::scale_color_manual(values = selectTypeCols) +
        #   ggplot2::theme(text = element_text(size = 96),
        #                  legend.position = "none") +
        #   ggplot2::xlim(c(-0.17, 3.7)) +
        #   ggplot2::xlab(label = "Genetic gain") +
        #   ggplot2::ylab(label = "CDF")
        #
        # fileNamePltCdf <- paste0(dirMidSCOBSTrainingPop1, scriptID, "_",
        #                          "density_plot_of_final_genetic_gain_for_9_strategies.png")
        # # png(filename = fileNamePltCdf, width = 1800, height = 1100)
        # png(filename = fileNamePltCdf, width = 2400, height = 1000)
        # print(pltCdf)
        # dev.off()




        geneticGainIterSplitDf <- data.frame(
          Strategy = rep(allocateStrategies,
                         lapply(X = geneticGainIterAllOrdList,
                                FUN = length)),
          SI = rep(selectionInts,
                   lapply(X = geneticGainIterAllOrdList,
                          FUN = length)),
          Value = unlist(geneticGainIterAllOrdList)
        )
        geneticGainIterSplitDf$Strategy <- factor(geneticGainIterSplitDf$Strategy,
                                                  levels = optEqNamesNow)
        pltCdfSplit <- ggplot2::ggplot(data = geneticGainIterSplitDf,
                                       aes(x = Value, group = Strategy,
                                           color = Strategy, linetype = Strategy)) +
          ggplot2::theme_bw() +
          ggplot2::facet_wrap(~ SI, nrow = 3, ncol = 1) +
          ggplot2::stat_ecdf(pad = TRUE,
                             lwd = selectTypeLwds) +
          ggplot2::scale_linetype_manual(breaks = allocateStrategies,
                                         values = selectTypeLtys) +
          ggplot2::scale_color_manual(values = selectTypeCols2) +
          ggplot2::theme(text = element_text(size = 96),
                         legend.position = "none") +
          ggplot2::xlim(c(0.7, 4.5)) +
          ggplot2::xlab(label = "Genetic gain") +
          ggplot2::ylab(label = "CDF")

        fileNamePltCdfSplit <- paste0(dirTrainingPop1, scriptID, "_",
                                      "density_plot_of_final_genetic_gain_for_", length(optEqNamesNow),
                                      "_strategies_SI_split_", nGenerationProceed, "_Gen_", selCriTypeName, ".png")
        # png(filename = fileNamePltCdf, width = 1800, height = 1100)
        png(filename = fileNamePltCdfSplit, width = 2000, height = 1800)
        print(pltCdfSplit)
        dev.off()


        ##### 3.3. Results for ten traits #####
      } else {
        #### 3.3.1. Boxplot for ten traits ####
        geneticGainFinalTenListNow <- lapply(X = geneticGainFinalTenList,
                                             FUN = function(x) {
                                               x[[selCriTypeName]]
                                             })

        if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
          # geneticGainFinalTenGradOpt1List <- lapply(
          #   X = geneticGainFinalTenListNow,
          #   FUN = function(geneticGainFinalTenStr) {
          #     do.call(
          #       what = rbind,
          #       args = lapply(
          #         X = geneticGainFinalTenStr,
          #         FUN = function(geneticGainFinalTenPheno) {
          #           geneticGainFinalTenOpt <-
          #             geneticGainFinalTenPheno[(length(geneticGainFinalTenPheno) - 3)]
          #
          #           return(geneticGainFinalTenOpt)
          #         }
          #       )
          #     )
          #   }
          # )
          geneticGainFinalTenGradOpt2List <- lapply(
            X = geneticGainFinalTenListNow,
            FUN = function(geneticGainFinalTenStr) {
              do.call(
                what = rbind,
                args = lapply(
                  X = geneticGainFinalTenStr,
                  FUN = function(geneticGainFinalTenPheno) {
                    geneticGainFinalTenOpt <-
                      geneticGainFinalTenPheno[(length(geneticGainFinalTenPheno) - 2)]

                    return(geneticGainFinalTenOpt)
                  }
                )
              )
            }
          )

          geneticGainFinalTenStoOptList <- lapply(
            X = geneticGainFinalTenListNow,
            FUN = function(geneticGainFinalTenStr) {
              do.call(
                what = rbind,
                args = lapply(
                  X = geneticGainFinalTenStr,
                  FUN = function(geneticGainFinalTenPheno) {
                    geneticGainFinalTenStoOpt <-
                      geneticGainFinalTenPheno[(length(geneticGainFinalTenPheno) - 1)]

                    return(geneticGainFinalTenStoOpt)
                  }
                )
              )
            }
          )

          geneticGainFinalTenEqList <- lapply(
            X = geneticGainFinalTenListNow,
            FUN = function(geneticGainFinalTenStr) {
              do.call(
                what = rbind,
                args = lapply(
                  X = geneticGainFinalTenStr,
                  FUN = function(geneticGainFinalTenPheno) {
                    geneticGainFinalTenEq <-
                      geneticGainFinalTenPheno[length(geneticGainFinalTenPheno)]

                    return(geneticGainFinalTenEq)
                  }
                )
              )
            }
          )

          geneticGainFinalTenAllList <- c(
            # geneticGainFinalTenGradOpt1List,
            geneticGainFinalTenGradOpt2List,
            geneticGainFinalTenStoOptList,
            geneticGainFinalTenEqList
          )
        } else {
          geneticGainFinalTenGradOpt1List <- lapply(
            X = geneticGainFinalTenListNow,
            FUN = function(geneticGainFinalTenStr) {
              do.call(
                what = rbind,
                args = lapply(
                  X = geneticGainFinalTenStr,
                  FUN = function(geneticGainFinalTenPheno) {
                    geneticGainFinalTenOpt <-
                      geneticGainFinalTenPheno[(length(geneticGainFinalTenPheno) - 1)]

                    return(geneticGainFinalTenOpt)
                  }
                )
              )
            }
          )

          geneticGainFinalTenEqList <- lapply(
            X = geneticGainFinalTenList,
            FUN = function(geneticGainFinalTenStr) {
              do.call(
                what = rbind,
                args = lapply(
                  X = geneticGainFinalTenStr,
                  FUN = function(geneticGainFinalTenPheno) {
                    geneticGainFinalTenEq <-
                      geneticGainFinalTenPheno[length(geneticGainFinalTenPheno)]

                    return(geneticGainFinalTenEq)
                  }
                )
              )
            }
          )

          geneticGainFinalTenAllList <- c(
            geneticGainFinalTenGradOpt1List,
            geneticGainFinalTenEqList
          )
        }
        names(geneticGainFinalTenAllList) <- strategyNamesAbbAllOrd



        strategyNamesAbbAllOrdSplit <- stringr::str_split(string = strategyNamesAbbAllOrd, pattern = "-")
        allocateStrategies <- unlist(lapply(X = strategyNamesAbbAllOrdSplit, FUN = function(x) x[1]))
        selectionInts <- unlist(lapply(X = strategyNamesAbbAllOrdSplit, FUN = function(x) x[2]))
        eachLen <- lapply(X = geneticGainFinalTenAllList, FUN = length)

        # geneticGainFinalTenOptDf <- data.frame(Strategy = rep(strategyNamesAbbAllOrd,
        #                                                       eachLen),
        #                                        Value = unlist(geneticGainFinalTenAllList))
        geneticGainFinalTenOptDf <- data.frame(Strategy = rep(allocateStrategies, eachLen),
                                               SI = rep(selectionInts, eachLen),
                                               Value = unlist(geneticGainFinalTenAllList))
        geneticGainFinalTenOptDf$Strategy <- factor(x = geneticGainFinalTenOptDf$Strategy,
                                                    levels = unique(allocateStrategies))

        pltBox <- ggplot2::ggplot(data = geneticGainFinalTenOptDf) +
          ggplot2::geom_boxplot(mapping = aes(x = Strategy, y = Value, fill = Strategy)) +
          facet_wrap(~ SI) +
          ggplot2::theme_bw() +
          ggplot2::scale_fill_manual(values = selectTypeCols) +
          ggplot2::theme(text = element_text(size = 72)) +
          ggplot2::ylab(label = "Genetic gain")



        fileNamePltBox <- paste0(dirGeno1, scriptID, "_",
                                 "comparison_of_final_genetic_gain_between_", length(optEqNamesNow),
                                 "_strategies_boxplot_", nGenerationProceed, "_Gen_", selCriTypeName, ".png")
        # png(filename = fileNamePltBox, width = 1800, height = 1100)
        png(filename = fileNamePltBox, width = 2400, height = 1000)
        print(pltBox)
        dev.off()


        pltViolin <- ggplot2::ggplot(data = geneticGainFinalTenOptDf) +
          ggplot2::geom_violin(mapping = aes(x = Strategy, y = Value, fill = Strategy)) +
          facet_wrap(~ SI) +
          ggplot2::theme_bw() +
          ggplot2::scale_fill_manual(values = selectTypeCols) +
          ggplot2::theme(text = element_text(size = 72)) +
          ggplot2::ylab(label = "Genetic gain")

        fileNamePltViolin <- paste0(dirGeno1, scriptID, "_",
                                    "comparison_of_final_genetic_gain_between_", length(optEqNamesNow),
                                    "_strategies_violin", nGenerationProceed, "_Gen_", selCriTypeName, ".png")
        # png(filename = fileNamePltViolin, width = 1800, height = 1100)
        png(filename = fileNamePltViolin, width = 2400, height = 1000)
        print(pltViolin)
        dev.off()


        if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
          geneticGainFinalTenOptList <- geneticGainFinalTenGradOpt2List
        } else {
          geneticGainFinalTenOptList <- geneticGainFinalTenGradOpt1List
        }



        # geneticGainFinalTenGradOptModiList$TBVGVP / tapply(X = geneticGainFinalTenGradOptModiList$EQ, INDEX = rep(1:10, 6), mean)
        geneticGainFinalTenOptEqDiff <- do.call(what = cbind,
                                                args = geneticGainFinalTenOptList) -
          do.call(what = cbind, args = geneticGainFinalTenEqList)
        # matrix(rep(tapply(X = unlist(geneticGainFinalTenEqList), INDEX = rep(1:10, 3), mean), 3), ncol = 3)
        boxplot(geneticGainFinalTenOptEqDiff)

        geneticGainFinalTenOptEqImp <- round((do.call(what = cbind,
                                                      args = geneticGainFinalTenOptList) /
                                                do.call(what = cbind, args = geneticGainFinalTenEqList) - 1) * 100, 3)
        colnames(geneticGainFinalTenOptEqImp) <- selectTypeNames2
        geneticGainFinalTenOptEqImpOrd <- geneticGainFinalTenOptEqImp


        fileNameGeneticGainFinalTenOptEqImpOrd <- paste0(dirGeno1, scriptID, "_",
                                                         "improvement_from_equal_allocation_for_gradient_optimized_strategies_",
                                                         nGenerationProceed, "_Gen_", selCriTypeName, ".csv")
        fwrite(x = data.frame(geneticGainFinalTenOptEqImpOrd),
               file = fileNameGeneticGainFinalTenOptEqImpOrd)


        if (selCriTypeName %in% c("WBV+GVP", "BV+WBV+GVP")) {
          geneticGainFinalTenStoOptEqDiff <- do.call(what = cbind,
                                                     args = geneticGainFinalTenStoOptList) -
            do.call(what = cbind, args = geneticGainFinalTenEqList)
          boxplot(geneticGainFinalTenStoOptEqDiff)

          geneticGainFinalTenStoOptEqImp <- round((do.call(what = cbind,
                                                           args = geneticGainFinalTenStoOptList) /
                                                     do.call(what = cbind, args = geneticGainFinalTenEqList) - 1) * 100, 3)
          colnames(geneticGainFinalTenStoOptEqImp) <- selectTypeNames2
          geneticGainFinalTenStoOptEqImpOrd <- geneticGainFinalTenStoOptEqImp


          fileNameGeneticGainFinalTenStoOptEqImpOrd <- paste0(dirGeno1, scriptID, "_",
                                                              "improvement_from_equal_allocation_for_StoSOO_optimized_strategies_",
                                                              nGenerationProceed, "_Gen_", selCriTypeName, ".csv")
          fwrite(x = data.frame(geneticGainFinalTenStoOptEqImpOrd),
                 file = fileNameGeneticGainFinalTenStoOptEqImpOrd)


          geneticGainFinalTenOptStoOptDiff <- do.call(what = cbind,
                                                      args = geneticGainFinalTenOptList) -
            do.call(what = cbind, args = geneticGainFinalTenStoOptList)
          boxplot(geneticGainFinalTenOptStoOptDiff)

          geneticGainFinalTenOptStoOptImp <- round((do.call(what = cbind,
                                                            args = geneticGainFinalTenOptList) /
                                                      do.call(what = cbind, args = geneticGainFinalTenStoOptList) - 1) * 100, 3)
          colnames(geneticGainFinalTenOptStoOptImp) <- selectTypeNames2
          geneticGainFinalTenOptStoOptImpOrd <- geneticGainFinalTenOptStoOptImp


          fileNameGeneticGainFinalTenOptStoOptImpOrd <- paste0(dirGeno1, scriptID, "_",
                                                               "improvement_from_StoOpt_for_optimized_strategies_",
                                                               nGenerationProceed, "_Gen_", selCriTypeName, ".csv")
          fwrite(x = data.frame(geneticGainFinalTenOptStoOptImpOrd),
                 file = fileNameGeneticGainFinalTenOptStoOptImpOrd)



          geneticGainFinalTenImpFromEq <- cbind(geneticGainFinalTenOptEqImpOrd, geneticGainFinalTenStoOptEqImpOrd)

          selectTypeColsImp <- selectTypeCols[-3]
          allocateStrategiesImp <- rep(optEqNamesNow[-3], each = 3)
          selectionIntsImp <- colnames(geneticGainFinalTenImpFromEq)
          eachLenImp <- apply(X = geneticGainFinalTenImpFromEq, MARGIN = 2, FUN = length)

          geneticGainFinalTenImpDf <- data.frame(Strategy = rep(allocateStrategiesImp, eachLenImp),
                                                 SI = rep(selectionIntsImp, eachLenImp),
                                                 Value = c(geneticGainFinalTenImpFromEq))
          geneticGainFinalTenImpDf$Strategy <- factor(x = geneticGainFinalTenImpDf$Strategy,
                                                      levels = unique(allocateStrategiesImp))

          pltBoxImp <- ggplot2::ggplot(data = geneticGainFinalTenImpDf) +
            ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
            ggplot2::geom_boxplot(mapping = aes(x = Strategy, y = Value, fill = Strategy)) +
            facet_wrap(~ SI) +
            ggplot2::theme_bw() +
            ggplot2::scale_fill_manual(values = selectTypeColsImp) +
            ggplot2::theme(text = element_text(size = 72)) +
            ggplot2::ylab(label = "Improvement from EQ (%)")

          fileNamePltBoxImp <- paste0(dirGeno1, scriptID, "_",
                                      "comparison_of_improvement_from_EQ_between_", length(optEqNamesNow),
                                      "_strategies_boxplot_", nGenerationProceed, "_Gen_", selCriTypeName, ".png")
          # png(filename = fileNamePltBoxImp, width = 1800, height = 1100)
          png(filename = fileNamePltBoxImp, width = 2400, height = 1000)
          print(pltBoxImp)
          dev.off()




          pltViolinImp <- ggplot2::ggplot(data = geneticGainFinalTenImpDf) +
            ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
            ggplot2::geom_violin(mapping = aes(x = Strategy, y = Value, fill = Strategy)) +
            facet_wrap(~ SI) +
            ggplot2::theme_bw() +
            ggplot2::scale_fill_manual(values = selectTypeColsImp) +
            ggplot2::theme(text = element_text(size = 72)) +
            ggplot2::ylab(label = "Improvement from EQ (%)")

          fileNamePltViolinImp <- paste0(dirGeno1, scriptID, "_",
                                         "comparison_of_improvement_from_EQ_between_", length(optEqNamesNow),
                                         "_strategies_violin_", nGenerationProceed, "_Gen_", selCriTypeName, ".png")

          # png(filename = fileNamePltViolinImp, width = 1800, height = 1100)
          png(filename = fileNamePltViolinImp, width = 2400, height = 1000)
          print(pltViolinImp)
          dev.off()
        }
      }
    }
  }
}
