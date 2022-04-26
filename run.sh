#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --cpus-per-task 2
#SBATCH --mem=20000M
##SBATCH -p mri2016

module add apps/python/3.8.5

PYTHONUSERBASE=/home/c/chamilton4/Desktop/CISProject/

python3 trainCirce.py
