#!/bin/bash
#PBS -P sj53
#PBS -q gpuvolta
#PBS -l walltime=48:00:00
#PBS -l mem=24GB
#PBS -l jobfs=0
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -M u7116534@anu.edu.au
#PBS -l storage=scratch/sj53
#PBS -o out_p4s2.txt
#PBS -e err_p4s2.txt
#PBS -l software=python

module load pytorch/1.9.0
cd /scratch/sj53/jt3998/RL4T1D/experiments

python3 new_run_irl.py experiment.name=p4s2 agent=irl hydra/job_logging=disabled env.patient_id=4 experiment.seed=2 agent.n_training_workers=16 agent.total_interactions=200000 irl_file=irl_p4s2
wait