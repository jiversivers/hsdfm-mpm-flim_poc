#!/bin/bash

#SBATCH --job-name=mclut_generator
#SBATCH --partition=comp06
#SBATCH --output=mclut_gen.txt
#SBATCH --error=mclut_gen.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdivers@uark.edu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --qos=gpu

export OMP_NUM_THREADS=1

# load required module
module purge
module load python/anaconda-3.14

# Activate venv
conda activate /home/jdivers/.conda/envs/mclut
echo $SLURM_JOB_ID

cd $SLURM_SUBMIT_DIR || exit
# input files needed for job

mkdir /scratch/$SLURM_JOB_ID
rsync -avq $SLURM_SUBMIT_DIR/generate_lut.py /scratch/$SLURM_JOB_ID
wait

cd /scratch/$SLURM_JOB_ID/ || exit

echo "Python script initiating..."
python3 generate_lut.py

rsync -av -q /scratch/$SLURM_JOB_ID/ $SLURM_SUBMIT_DIR/
echo "LUT saved at $HOME"

# check if rsync succeeded
if [ $? -ne 0 ]; then
  echo "Error: Failed to sync files back to original directory. Check /scratch/$SLURM_JOB_ID/ for output files."
  exit 1
fi