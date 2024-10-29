##########################################################################################
######  Title: 0.1_SimulatedCrop_SCOBS-PT_generate_initial_population_for_pytorch   ######
######  Author: Kosuke Hamazaki (hamazaki@ut-biomet.org)                            ######
######  Affiliation: Lab. of Biometry and Bioinformatics, The University of Tokyo   ######
######  Date: 2020/12/12 (Created), 2024/06/24 (Last Updated)                       ######
##########################################################################################





###### 1. Settings ######
##### 1.0. Reset workspace ######
# rm(list=ls())



##### 1.1. Setting working directory to the "SCOBS" directory #####
cropName <- "SimulatedCrop"
project <- "SCOBS-PT"
os <- osVersion

if (stringr::str_detect(string = os, pattern = "Ubuntu")) {
  setwd("/home/hamazaki/SCOBS-PT/")
}

isRproject <- function(path = getwd()) {
  files <- list.files(path)

  if (length(grep(".Rproj", files)) >= 1) {
    out <- TRUE
  } else {
    out <-  FALSE
  }
  return(out)
}


scriptID <- "0.1"


##### 1.2. Setting some parameters #####
dirMidSCOBSBase <- "midstream/"


#### 1.2.1. Setting some parameters related to names of R6 class ####
scenarioNo <- 3
simName <- paste0("Scenario_", scenarioNo)
breederName <- "K. Hamazaki"

dirMidSCOBSScenario <- paste0(dirMidSCOBSBase, scriptID, "_", simName, "/")
if (!dir.exists(dirMidSCOBSScenario)) {
  dir.create(dirMidSCOBSScenario)
}


#### 1.2.2. Setting some parameters related to simulation & specie ####
trialNo <- 1
trialName <- paste0("Trial_", trialNo)

dirMidSCOBSTrial <- paste0(dirMidSCOBSScenario, scriptID,
                           "_", project, "_", trialName, "/")
if (!dir.exists(dirMidSCOBSTrial)) {
  dir.create(dirMidSCOBSTrial)
}


nSimGeno <- 1
nSimPheno <- 10

simGeno <- TRUE
simPheno <- TRUE

nChr <- 10
lChrEach <- 200
lChr <- rep(lChrEach, nChr)
nLociEach <- 82
nLoci <- rep(nLociEach, nChr)
effPopSize <- 2000

methodDist <- "euclidean"
# nTrainingPopInits <- c(2000, 1000, 500, 250)
nTrainingPopInits <- 250
nIndsPopInit <- 250

seedSimHaplo <- sample(x = 1e09, size = 1)
seedSimRM <- sample(x = 1e09, size = 1)
seedSimRS <- sample(x = 1e09, size = 1)
seedSimMC <- sample(x = 1e09, size = 1)

#### 1.2.3. Setting some parameters related to estimation of marker effects ####
methodMLRInit <- "BRR"
multiTraitInit <- FALSE

alpha <- 0.5
nIter <- 20000
burnIn <- 5000
thin <- 5
bayesian <- TRUE



#### 1.2.4. Setting some parameters related to traits ####
nMarkersEach <- 2
nMarkers <- rep(nMarkersEach, nChr)
nTraits <- 1
nQTLsEachChr <- 80
nQTLs <- rep(nQTLsEachChr, nChr)
qtlOverlap <- FALSE
nOverlap <- 0
effCor <- 0
propDomi <- 0
interactionMean <- 0

herit <- 0.7
envSpecificEffects <- 0
residCor <- 0

nRepInit <- 3
multiTraitsAsEnvs <- FALSE


#### 1.2.5. Setting some parameters fixed across scenarios ####
nCoreMax <- 5

ploidy <- 2
mutRate <- 1e-08
recombRate <- 1e-06
chrNames <- myBreedSimulatR::.charSeq(prefix = "Chr", seq = seq(nChr))
recombRateOneVal <- FALSE
verbose <- TRUE

actionTypeEpiSimple <- TRUE

founderIsInitPop <- FALSE
nGenerationRM <- 300
nGenerationRSM <- 15
nGenerationRM2 <- 3
propSelRS <- 0.1
popNameBase <- "G"

initGenotyping <- TRUE
mafThres <- 0.05
heteroThres <- 1

multiTraitsAsEnvs <- FALSE
includeIntercept <- TRUE

estimateGV <- TRUE
estimatedGVMethod <- "lme4"


#### 1.2.6. Save parameters ####
fileParamsSCOBS <- paste0(dirMidSCOBSTrial, scriptID,
                          "_", project, "_", trialName,
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




###### 2. Create simulation & specie information ######
##### 2.1. Create simulation information #####
fileNameSimInfo <- paste0(dirMidSCOBSTrial, scriptID,
                          "_simInfo.rds")
if (!file.exists(fileNameSimInfo)) {
  mySimInfo <- myBreedSimulatR::simInfo$new(simName = simName,
                                            simGeno = simGeno,
                                            simPheno = simPheno,
                                            # nSimGeno = nSimGeno,
                                            # nSimPheno = nSimPheno,
                                            nCoreMax = nCoreMax,
                                            # nCorePerGeno = 1,
                                            # nCorePerPheno = 3,
                                            # saveDataFileName = NULL
  )

  saveRDS(object = mySimInfo, file = fileNameSimInfo)
} else {
  mySimInfo <- readRDS(file = fileNameSimInfo)
}


simGenoNo <- 1
simPhenoNo <- 1
for (simGenoNo in 1:nSimGeno) {
  genoName <- paste0("Geno_", simGenoNo)
  specName <- paste0(simName, "-", trialName,
                     "-", genoName)

  dirMidSCOBSGeno <- paste0(dirMidSCOBSTrial, scriptID,
                            "_", genoName, "/")
  if (!dir.exists(dirMidSCOBSGeno)) {
    dir.create(dirMidSCOBSGeno)
  }

  ##### 2.2. Create specie information #####
  fileNameSpecie <- paste0(dirMidSCOBSGeno, scriptID,
                           "_specie.rds")
  if (!file.exists(fileNameSpecie)) {
    mySpec <- myBreedSimulatR::specie$new(nChr = nChr,
                                          lChr = lChr,
                                          specName = specName,
                                          ploidy = ploidy,
                                          mutRate = mutRate,
                                          recombRate = recombRate,
                                          chrNames = chrNames,
                                          nLoci = nLoci,
                                          recombRateOneVal = recombRateOneVal,
                                          effPopSize = effPopSize,
                                          simInfo = mySimInfo,
                                          verbose = verbose)
    saveRDS(object = mySpec, file = fileNameSpecie)
  } else {
    mySpec <- readRDS(file = fileNameSpecie)
  }


  ###### 3. Generate marker genotype and initial population (bsInfo) ######
  ##### 3.1. Create lociInfo object #####
  fileNameLociInfo <- paste0(dirMidSCOBSGeno, scriptID,
                             "_lociInfo.rds")
  if (!file.exists(fileNameLociInfo)) {
    myLociInfo <- myBreedSimulatR::lociInfo$new(genoMap = NULL,
                                                specie = mySpec)
    myLociInfo$computeRecombBetweenMarkers()

    saveRDS(object = myLociInfo, file = fileNameLociInfo)
  } else {
    myLociInfo <- readRDS(file = fileNameLociInfo)
  }
  print(paste0("Geno: ", simGenoNo, " ;  Generate marker genotype !!"))




  ##### 3.2. Create bsInfo object #####
  bsName <- paste0(specName, "-BreedingScheme")
  print("create bsInfo object")

  myBsInfoInitCommon <- myBreedSimulatR::bsInfo$new(bsName = bsName,
                                                    simInfo = mySimInfo,
                                                    specie = mySpec,
                                                    lociInfo = myLociInfo,
                                                    traitInfo = NULL,
                                                    geno = NULL,
                                                    haplo = NULL,
                                                    founderIsInitPop = founderIsInitPop,
                                                    nGenerationRM = nGenerationRM,
                                                    nGenerationRSM = nGenerationRSM,
                                                    nGenerationRM2 = nGenerationRM2,
                                                    propSelRS = propSelRS,
                                                    seedSimHaplo = seedSimHaplo,
                                                    seedSimRM = seedSimRM,
                                                    seedSimRS = seedSimRS,
                                                    seedSimMC = seedSimMC,
                                                    popNameBase = popNameBase,
                                                    initIndNames = NULL,
                                                    herit = herit,
                                                    envSpecificEffects = envSpecificEffects,
                                                    residCor = residCor,
                                                    verbose = verbose)


  ###### 4. Generate QTL information and corresponding objects for initial breeding scheme and breeder ######
  for (simPhenoNo in 1:nSimPheno) {
    phenoName <- paste0("Pheno_", simPhenoNo)
    print(phenoName)

    dirMidSCOBSPheno <- paste0(dirMidSCOBSGeno, scriptID,
                               "_", phenoName, "/")
    if (!dir.exists(dirMidSCOBSPheno)) {
      dir.create(dirMidSCOBSPheno)
    }

    ##### 3.2. Create traitInfo object #####
    fileNameTraitInfo <- paste0(dirMidSCOBSPheno, scriptID,
                                "_traitInfo.rds")

    if (!file.exists(fileNameTraitInfo)) {
      myTraitInfo <- myBreedSimulatR::traitInfo$new(lociInfo = myLociInfo,
                                                    nMarkers = nMarkers,
                                                    nTraits = nTraits,
                                                    nQTLs = nQTLs,
                                                    actionTypeEpiSimple = actionTypeEpiSimple,
                                                    qtlOverlap = qtlOverlap,
                                                    nOverlap = nOverlap,
                                                    effCor = effCor,
                                                    propDomi = propDomi,
                                                    interactionMean = interactionMean)

      saveRDS(object = myTraitInfo, file = fileNameTraitInfo)
    } else {
      myTraitInfo <- readRDS(file = fileNameTraitInfo)
    }

    myBsInfoInit <- myBsInfoInitCommon$clone(deep = FALSE)
    myBsInfoInit$bsName <- paste0(specName, "-", phenoName)
    myBsInfoInit$traitInfo <- myTraitInfo
    myBsInfoInit$populations[[1]]$traitInfo <- myTraitInfo
    for (indNo in 1:length(myBsInfoInit$populations[[1]]$inds)) {
      myBsInfoInit$populations[[1]]$inds[[indNo]]$traitInfo <- myTraitInfo
    }
    print(myTraitInfo$qtlPos[[1]][1:5])


    ##### 4.2. Create breederInfo object #####
    print("create breederInfo object")
    myBreederInfoInit <- myBreedSimulatR::breederInfo$new(breederName = breederName,
                                                          bsInfo = myBsInfoInit,
                                                          mrkNames = NULL,
                                                          initGenotyping = initGenotyping,
                                                          initGenotypedIndNames = NULL,
                                                          mafThres = mafThres,
                                                          heteroThres = heteroThres,
                                                          multiTraitsAsEnvs = multiTraitsAsEnvs,
                                                          includeIntercept = includeIntercept,
                                                          verbose = FALSE)



    ##### 4.3. Perform phenotyping #####
    myBreederInfoInit$phenotyper(bsInfo = myBsInfoInit,
                                 generationOfInterest = NULL,
                                 nRep = nRepInit,
                                 estimateGV = estimateGV,
                                 estimatedGVMethod = estimatedGVMethod)


    genoMatInit <- myBsInfoInit$populations[[myBsInfoInit$generation]]$genoMat
    if (simPhenoNo == 1) {
      skipClustering <- FALSE
    }  else {
      # skipClustering <- FALSE
      skipClustering <- sum(abs(genoMatInitOld - genoMatInit)) == 0
      print(sum(abs(genoMatInitOld - genoMatInit)) == 0)
    }
    genoMatInitOld <- genoMatInit

    if (!skipClustering) {
      distGenoMatInit <- Rfast::Dist(x = genoMatInit,
                                     method = methodDist)
      rownames(distGenoMatInit) <- colnames(distGenoMatInit) <- rownames(genoMatInit)
      distGenoInit <- as.dist(distGenoMatInit)

      indNamesInit <- myBreederInfoInit$populationsFB[[myBreederInfoInit$generation]]$indNames

      if (nIndsPopInit != effPopSize) {
        kmedoidsRes <- cluster::pam(x = distGenoInit,
                                    k = nIndsPopInit,
                                    pamonce = 5)
        selCandsInit <- indNamesInit[indNamesInit %in% kmedoidsRes$medoids]
      } else {
        selCandsInit <- indNamesInit
      }

      if (simPhenoNo >= 2) {
        selCandsCommon <- Reduce(f = intersect,
                                 x = list(selCandsInitOld, selCandsInit))

        print(length(selCandsCommon))
      }
      selCandsInitOld <- selCandsInit
    }

    print("create crossInfo object")
    crossInfoInit <- myBreedSimulatR::crossInfo$new(parentPopulation = myBsInfoInit$populations[[myBsInfoInit$generation]],
                                                    nSelectionWays = 1,
                                                    selectionMethod = "userSpecific",
                                                    traitNoSel = 1,
                                                    userSI = NULL,
                                                    lociEffects = NULL,
                                                    blockSplitMethod = "nMrkInBlock",
                                                    nMrkInBlock = 1,
                                                    minimumSegmentLength = 5,
                                                    nSelInitOPV = nIndsPopInit,
                                                    nIterOPV = 1e04,
                                                    nProgeniesEMBV = 50,
                                                    nIterEMBV = 5,
                                                    nCoresEMBV = 1,
                                                    clusteringForSel = FALSE,
                                                    nCluster = 1,
                                                    nTopCluster = 1,
                                                    nTopEach = nIndsPopInit,
                                                    nSel = nIndsPopInit,
                                                    multiTraitsEvalMethod = "sum",
                                                    hSel = 1,
                                                    matingMethod = "nonCross",
                                                    allocateMethod = "equalAllocation",
                                                    weightedAllocationMethod = NULL,
                                                    nProgenies = NULL,
                                                    traitNoRA = 1,
                                                    h = 0.1,
                                                    minimumUnitAllocate = 1,
                                                    includeGVP = FALSE,
                                                    nNextPop = nIndsPopInit,
                                                    nPairs = NULL,
                                                    nameMethod = "individualBase",
                                                    indNames = NULL,
                                                    seedSimRM = NULL,
                                                    seedSimMC = NULL,
                                                    selCands = selCandsInit,
                                                    crosses = NULL,
                                                    verbose = TRUE)


    myBsInfoInit$nextGeneration(crossInfo = crossInfoInit)

    myBreederInfoInit$getNewPopulation(bsInfo = myBsInfoInit,
                                       generationNew = NULL,
                                       genotyping = TRUE,
                                       genotypedIndNames = NULL)


    fileNameBsInfoInit <- paste0(dirMidSCOBSPheno, scriptID,
                                 "_bsInfoInit.rds")
    # saveRDS(object = myBsInfoInit, file = fileNameBsInfoInit)

    fileNameBreederInfoInit <- paste0(dirMidSCOBSPheno, scriptID,
                                      "_breederInfoInit.rds")
    # saveRDS(object = myBreederInfoInit, file = fileNameBreederInfoInit)

    trainingIndNamesInits <- list()
    for (nTrainingPopInit in nTrainingPopInits) {
      print(paste0("Geno: ", simGenoNo, "  Pheno: ", simPhenoNo,
                   "  TrainingPop: ", nTrainingPopInit))
      dirMidSCOBSTrainingPop <- paste0(dirMidSCOBSPheno, scriptID,
                                       "_Model_with_", nTrainingPopInit, "/")
      if (!dir.exists(dirMidSCOBSTrainingPop)) {
        dir.create(dirMidSCOBSTrainingPop)
      }


      myBsInfoInitNow <- myBsInfoInit$clone(deep = FALSE)
      myBreederInfoInitNow <- myBreederInfoInit$clone(deep = FALSE)

      if (!skipClustering) {
        if (nTrainingPopInit != effPopSize) {
          kmedoidsResTraining <- cluster::pam(x = distGenoInit,
                                              k = nTrainingPopInit,
                                              pamonce = 5)
          trainingIndNamesInit <- indNamesInit[indNamesInit %in% kmedoidsResTraining$medoids]
        } else {
          trainingIndNamesInit <- indNamesInit
        }
        trainingIndNamesInits[[paste0("Model_", nTrainingPopInit)]] <- trainingIndNamesInit

        if (simPhenoNo >= 2) {
          trainingIndNamesCommon <- Reduce(f = intersect,
                                           x = list(trainingIndNamesInitsOld[[paste0("Model_", nTrainingPopInit)]],
                                                    trainingIndNamesInit))

          print(length(trainingIndNamesCommon))
        }
      } else {
        trainingIndNamesInit <- trainingIndNamesInitsOld[[paste0("Model_", nTrainingPopInit)]]
      }
      myBreederInfoInitNow$estimateMrkEff(trainingPop = 1,
                                          trainingIndNames = trainingIndNamesInit,
                                          methodMLR = methodMLRInit,
                                          multiTrait = multiTraitInit,
                                          alpha = alpha,
                                          nIter = nIter,
                                          burnIn = burnIn,
                                          thin = thin,
                                          bayesian = bayesian)

      myBreederInfoInitNow$estimateGVByMLR(trainingPop = 1,
                                           trainingIndNames = trainingIndNamesInit,
                                           testingPop = length(myBreederInfoInitNow$populationsFB),
                                           methodMLR = methodMLRInit,
                                           multiTrait = multiTraitInit,
                                           alpha = alpha,
                                           nIter = nIter,
                                           burnIn = burnIn,
                                           thin = thin,
                                           bayesian = bayesian)

      myBreederInfoInitNow$estimatedMrkEffInfo[[paste0("G2_", methodMLRInit)]] <-
        myBreederInfoInitNow$estimatedMrkEffInfo[[paste0("G1_", methodMLRInit)]]
      myBreederInfoInitNow$estimatedMrkEffInfo[[paste0("G1_", methodMLRInit)]] <- NULL

      fileNameBreederInfoInitWithMrkEff <- paste0(dirMidSCOBSTrainingPop, scriptID,
                                                  "_breederInfoInit_with_estimated_marker_effects_",
                                                  methodMLRInit, "_", nTrainingPopInit, "_individuals.rds")
      # saveRDS(object = myBreederInfoInitNow, file = fileNameBreederInfoInitWithMrkEff)



      myBsInfoInitNow$removeInitialPop()
      myBreederInfoInitNow$removeInitialPop()

      myBsInfoInitNow$populations[[1]]$crossInfo <- NULL
      myBreederInfoInitNow$populationsFB[[1]]$crossInfo <- NULL

      myBreederInfoInitNow$estimatedGVByMLRInfo[[1]]$mrkEffMat <- NULL
      myBreederInfoInitNow$estimatedGVByMLRInfo[[1]]$totalEstimatedGVByRep <- NULL
      myBreederInfoInitNow$estimatedGVByMLRInfo[[1]]$totalEstimatedGVByMLR <- NULL
      myBreederInfoInitNow$estimatedGVByMLRInfo[[1]]$plot <- NULL
      myBreederInfoInitNow$estimatedMrkEffInfo[[1]]$mrkEstRes <- NULL
      myBreederInfoInitNow$estimatedMrkEffInfo[[1]]$plot <- NULL


      myBsInfoInitNow$generation <-
        myBsInfoInitNow$populations[[1]]$generation <- 1
      names(myBsInfoInitNow$populations)[1] <-
        myBsInfoInitNow$populations[[1]]$name <- paste0(myBsInfoInitNow$popNameBase, "_",
                                                        myBsInfoInitNow$generation)

      myBreederInfoInitNow$generation <-
        myBreederInfoInitNow$populationsFB[[1]]$generation <- 1
      names(myBreederInfoInitNow$populationsFB)[1] <-
        names(myBreederInfoInitNow$estimatedGVByMLRInfo)[1] <-
        myBsInfoInitNow$populations[[1]]$name <- paste0(myBreederInfoInitNow$popNameBase, "_",
                                                        myBreederInfoInitNow$generation)
      names(myBreederInfoInitNow$estimatedMrkEffInfo)[1] <-  paste0(myBreederInfoInitNow$popNameBase, "_",
                                                                    myBreederInfoInitNow$generation,
                                                                    "_", methodMLRInit)


      print(names(myBsInfoInitNow$populations[[1]]$inds)[1:10])

      fileNameBsInfoInitRmInit <- paste0(dirMidSCOBSTrainingPop, scriptID,
                                         "_bsInfoInit_remove_initial_population.rds")
      saveRDS(object = myBsInfoInitNow, file = fileNameBsInfoInitRmInit)

      fileNameBreederInfoInitWithMrkEffRmInit <- paste0(dirMidSCOBSTrainingPop, scriptID,
                                                        "_breederInfoInit_with_estimated_marker_effects_",
                                                        methodMLRInit, "_", nTrainingPopInit,
                                                        "_individuals_remove_initial_population.rds")
      saveRDS(object = myBreederInfoInitNow, file = fileNameBreederInfoInitWithMrkEffRmInit)


      ### For PyTorch
      dirPyTorch <- paste0(dirMidSCOBSTrainingPop, "PyTorch/")
      if (!dir.exists(dirPyTorch)) {
        dir.create(dirPyTorch)
      }

      founderHaplo <- myBsInfoInitNow$populations[[1]]$haploArray
      founderHaplo1 <- founderHaplo[, , 1]
      founderHaplo2 <- founderHaplo[, , 2]
      fileNameFounderHaplo1 <- paste0(dirPyTorch, "founderHaplo1.csv")
      fileNameFounderHaplo2 <- paste0(dirPyTorch, "founderHaplo2.csv")
      write.csv(x = founderHaplo1, file = fileNameFounderHaplo1,
                row.names = TRUE)
      write.csv(x = founderHaplo2, file = fileNameFounderHaplo2,
                row.names = TRUE)


      lociEffectsTrue <- myBsInfoInitNow$lociEffects
      genoMap <- myBsInfoInitNow$lociInfo$genoMap
      mrkPos <- myBsInfoInitNow$traitInfo$mrkPos - 1

      fileNameLociEffectsTrue <- paste0(dirPyTorch, "lociEffectsTrue.csv")
      fileNameGenoMap <- paste0(dirPyTorch, "genoMap.csv")
      fileNameMrkPos <- paste0(dirPyTorch, "mrkPos.csv")

      write.csv(x = lociEffectsTrue, file = fileNameLociEffectsTrue,
                row.names = TRUE)
      write.csv(x = genoMap, file = fileNameGenoMap,
                row.names = TRUE)
      write.csv(x = mrkPos, file = fileNameMrkPos,
                row.names = TRUE)

      recombRatesList <- myBsInfoInitNow$lociInfo$recombBetweenMarkersList
      for (chrNo in 1:nChr) {
        recombRates <- recombRatesList[[chrNo]]

        fileNameRecombRatesChr <- paste0(dirPyTorch, "recombRates",
                                         chrNo - 1, ".csv")
        write.csv(x = recombRates, file = fileNameRecombRatesChr,
                  row.names = TRUE)
      }

    }
    trainingIndNamesInitsOld <- trainingIndNamesInits
  }
}

