#!/bin/bash

conda activate nesy-mammography

srun -u python dataHandling.py
