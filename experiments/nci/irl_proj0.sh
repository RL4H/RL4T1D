#!/bin/bash
#PBS -P sj53
#PBS -q gpuvolta
#PBS -l walltime=24:00:00
#PBS -l mem=24GB
#PBS -l jobfs=0
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -M u7116534@anu.edu.au
#PBS -l storage=scratch/sj53
#PBS -o out_proj0.txt
#PBS -e err_proj0.txt
#PBS -l software=python

module load pytorch/1.9.0

python3 /scratch/sj53/jt3998/RL4T1D/experiments/run_irl.py --patient_id 0 --n_expert 100 --l_expert 256 --i_irl 50 --i_update_init 100 --i_update 25 --n_sim 100 --l_sim 256
wait
