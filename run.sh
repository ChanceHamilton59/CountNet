#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --cpus-per-task 2
#SBATCH --mem=20000M

module add apps/python/3.8.5

PYTHONUSERBASE=/path/to/workspace/on/circe/ #Must change for your directory

python3 trainCirce.py
