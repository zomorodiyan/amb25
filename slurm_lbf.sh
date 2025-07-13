#!/bin/bash
#SBATCH --mem=64G              # ajust
#SBATCH -p general             # Partition (queue)
#SBATCH -q public              # QoS
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 4                   # Total MPI tasks
#SBATCH -c 1                   # Cores per task
#SBATCH -t 0-00:00:20          # Time in D-HH:MM:SS (1 min for test)
#SBATCH --job-name=openfoam-lbf-test
#SBATCH --output=job_output.log
#SBATCH --error=job_error.log

# Load Apptainer (if your cluster uses modules)
#module load apptainer

# Define variables
CASE_DIR=/home/mzomoro1/code/AMB/
SIF_PATH=/home/mzomoro1/code/AMB/lbf.sif

# Step 1: blockMesh
apptainer exec $SIF_PATH bash -c "source /opt/OpenFOAM/OpenFOAM-10/etc/bashrc && cd $CASE_DIR && blockMesh"

# Step 2: setFields
apptainer exec $SIF_PATH bash -c "source /opt/OpenFOAM/OpenFOAM-10/etc/bashrc && cd $CASE_DIR && setFields"

# Step 3: decomposePar
apptainer exec $SIF_PATH bash -c "source /opt/OpenFOAM/OpenFOAM-10/etc/bashrc && cd $CASE_DIR && decomposePar"

# Step 4: Run solver
srun --mpi=pmi2 apptainer exec $SIF_PATH bash -c "source /opt/OpenFOAM/OpenFOAM-10/etc/bashrc && cd $CASE_DIR && laserbeamFoam -parallel"

# Step 5: Reconstruct (optional)
#apptainer exec $SIF_PATH bash -c "source /opt/OpenFOAM/OpenFOAM-10/etc/bashrc && cd $CASE_DIR && reconstructPar"

