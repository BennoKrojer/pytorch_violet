#!/bin/bash
#SBATCH --job-name=clip
#SBATCH --output=logs/%A.log
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
echo 	"Arguments:	$@"
echo -n	"Date:		"; date
echo 	"JobId:		$SLURM_JOBID"
echo	"Node:		$HOSTNAME"
echo	"Nodelist:	$SLURM_JOB_NODELIST"

# activate conda env
module purge >/dev/null 2>&1
module load python/3.9
source violet_env/bin/activate
# Export env variables
export PYTHONBUFFERED=1

python -u $@ --job_id "$SLURM_JOBID"
