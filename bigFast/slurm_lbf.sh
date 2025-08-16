#!/bin/bash
#SBATCH --mem=128G
#SBATCH -p htc
#SBATCH -q public
#SBATCH -N 1
#SBATCH -n 128                # Total MPI tasks
#SBATCH -c 1                  # Cores per task
#SBATCH -t 4-00:00:00         # Adjust time limit
#SBATCH --job-name=amb25-bigFast
#SBATCH --output=job_output.log
#SBATCH --error=job_error.log

# Define variables
CASE_DIR=/home/mzomoro1/amb25/bigFast
SIF_PATH=/home/mzomoro1/amb25/lbf-he.sif
LOG_TIME=$CASE_DIR/log.laserbeamFoam.time

# Clean previous timing file
> $LOG_TIME

# Function to run a command inside the container
foam_exec() {
    CMD=$1
    apptainer exec $SIF_PATH bash -c "source /opt/OpenFOAM/OpenFOAM-10/etc/bashrc && cd $CASE_DIR && $CMD"
}

echo "Copying 'initial' to 0..."
foam_exec "rm -rf 0 && cp -r initial 0"

echo "Running blockMesh..."
foam_exec "blockMesh"

echo "Running setFields..."
foam_exec "setFields"

echo "Running decomposePar..."
foam_exec "decomposePar"

# ----------------------------
# Run simulation with timing
# ----------------------------
echo "Running laserbeamFoam..."
/usr/bin/time -f "\n=== laserbeamFoam Timing ===\nElapsed time: %E\nUser time: %U\nCPU usage: %P" \
srun --mpi=pmi2 apptainer exec $SIF_PATH bash -c "source /opt/OpenFOAM/OpenFOAM-10/etc/bashrc && cd $CASE_DIR && laserbeamFoam -parallel" \
> $CASE_DIR/log.laserbeamFoam 2>> $LOG_TIME

# ----------------------------
# Reconstruction with timing
# ----------------------------
echo "Running reconstructParMesh..."
/usr/bin/time -f "\n=== reconstructParMesh Timing ===\nElapsed time: %E\nUser time: %U\nCPU usage: %P" \
apptainer exec $SIF_PATH bash -c "source /opt/OpenFOAM/OpenFOAM-10/etc/bashrc && cd $CASE_DIR && reconstructParMesh -constant" \
> $CASE_DIR/log.reconstructParMesh 2>> $LOG_TIME

echo "Running reconstructPar..."
/usr/bin/time -f "\n=== reconstructPar Timing ===\nElapsed time: %E\nUser time: %U\nCPU usage: %P" \
apptainer exec $SIF_PATH bash -c "source /opt/OpenFOAM/OpenFOAM-10/etc/bashrc && cd $CASE_DIR && reconstructPar" \
> $CASE_DIR/log.reconstructPar 2>> $LOG_TIME

# Clean up processors and create case.foam
foam_exec "rm -r processor* && touch case.foam"

echo "Simulation complete."

