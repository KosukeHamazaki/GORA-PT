#!/bin/sh
#$ -S /bin/bash
#$ -cwd
#$ -pe OpenMP 40
#$ -jc pcc-skl.168h
##$ -jc pcc-large.168h
##$ -jc pcc-large.24h
##$ -jc pcc-normal.168h 
#$ -m a
#$ -M kosuke.hamazaki@riken.jp

source ~/.bashrc
source ~/.bash_profile
ulimit -s unlimited
#export OMP_NUM_THREADS=40
#export OMP_STACKSIZE=9GB
singularity exec --bind /home:/home,/home/hamazaki/tmp:/tmp /home/hamazaki/R_environment.sif Rscript /home/hamazaki/SCOBS-PT/scripts/0.1.1_SimulatedCrop_SCOBS-PT_generate_initial_population_for_pytorch_for_raiden.R > /home/hamazaki/SCOBS-PT/midstream/0.1.1_SimulatedCrop_SCOBS-PT_generate_initial_population_for_pytorch_for_raiden_log.txt 2>&1
# /home/hamazaki/anaconda3/bin/python /home/hamazaki/SCOBS-PT/scripts/2.1.1_SimulatedCrop_SCOBS-PT_optimize_breeding_strategies_via_pytorch_for_raiden.py 3 100 SI4 WBV+GVP SGD False 0.15 500 > output.txt 2>&1
# nGenerationProceed nIterSimulation selectTypeName selCriTypeName optimMethod useScheduler learningRate nIterOptimization