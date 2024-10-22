#!/bin/bash
#PBS -P sj53
#PBS -l ncpus=1
#PBS -l mem=2GB
#PBS -l jobfs=2GB
#PBS -q copyq
#PBS -l walltime=02:00:00
#PBS -l storage=scratch/sj53
#PBS -l wd
#PBS -o out11.txt
#PBS -e err11.txt

zip -r results_nci.zip /scratch/sj53/jt3998/RL4T1D/experiments/nci
