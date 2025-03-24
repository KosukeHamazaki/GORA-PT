#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -pe OpenMP 40
#$ -jc pcc-skl.168h
##$ -jc pcc-large.168h
##$ -jc pcc-normal.168h 
#$ -m a
#$ -M kosuke.hamazaki@riken.jp
nGenerationProceed=4
nIterSimulation=100
selectTypeName="SI3"
selCriTypeName="WBV+GVP"
optimMethod="SGD"
useScheduler="True"
learningRate=0.15
nIterOptimization=100
gvpInit="Zero"

source ~/.bashrc
source ~/.bash_profile
ulimit -s unlimited
#export OMP_NUM_THREADS=40
#export OMP_STACKSIZE=9GB
##singularity exec --bind /home:/home,/home/hamazaki/tmp:/tmp /home/hamazaki/R_environment.sif Rscript /home/hamazaki/SCOBS-PT/scripts/0.1.1_SimulatedCrop_SCOBS-PT_generate_initial_population_for_pytorch_for_raiden.R > /home/hamazaki/SCOBS-PT/midstream/0.1.1_SimulatedCrop_SCOBS-PT_generate_initial_population_for_pytorch_for_raiden_log.txt 2>&1
/home/hamazaki/miniconda3/bin/python /home/hamazaki/SCOBS-PT/scripts/2.3.1_SimulatedCrop_SCOBS-PT_optimize_breeding_strategies_via_pytorch_for_conv_for_raiden.py $nGenerationProceed $nIterSimulation $selectTypeName $selCriTypeName $optimMethod $useScheduler $learningRate $nIterOptimization $gvpInit > /home/hamazaki/SCOBS-PT/midstream/2.3.1_SimulatedCrop_SCOBS-PT_optimize_breeding_strategies_via_pytorch_for_conv_for_raiden_log.txt 
##2>&1
# nGenerationProceed nIterSimulation selectTypeName selCriTypeName optimMethod useScheduler learningRate nIterOptimization gvpInit
