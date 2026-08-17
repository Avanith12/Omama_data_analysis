#!/bin/bash
#SBATCH --job-name=hq_deepsight
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=a.kanamarlapudi001@umb.edu # Where to send mail
#SBATCH -A aicore
#SBATCH -q aicore
#SBATCH -p AICORE_A100
#SBATCH -w chimera12
#SBATCH --gres=gpu:A100:1
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=128G
#SBATCH -t 3-00:00
#SBATCH --output=/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/deepsight/logs/array_%A-%a.out
#SBATCH --error=/hpcstor6/scratch01/a/a.kanamarlapudi001/HQ_StudyLevel_Data/deepsight/logs/array_%A-%a.err
#SBATCH --array=1-12

. /etc/profile

# check cpu number per task, should be equal to -n
echo "using $SLURM_CPUS_ON_NODE CPUs"
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID"
echo `date`

eval "$(conda shell.bash hook)"
conda activate O

cd /home/a.kanamarlapudi001/projects/omama-proj/avanith/original_data/HQ/deepsight/scripts
python hq_deepsight_executor.py $SLURM_ARRAY_TASK_ID

echo "Finish Run"
echo "end time is `date`"
