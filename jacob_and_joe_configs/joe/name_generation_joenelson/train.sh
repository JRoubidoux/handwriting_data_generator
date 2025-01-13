#!/bin/bash

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=6   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=8
#SBATCH --mem-per-cpu=10GB   # memory per CPU core
#SBATCH -J "Train Paris Name Fields"   # job name
#SBATCH --mail-user=jnel8982@byu.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=standby
#SBATCH --requeue


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
/grphome/fslg_census/nobackup/archive/envs/torch_2/bin/python /grphome/fslg_census/nobackup/archive/projects/paris_french_census/name_generation_joenelson/train_paris_name_fields.py 200 /grphome/fslg_census/nobackup/archive/projects/paris_french_census/name_generation_joenelson/model