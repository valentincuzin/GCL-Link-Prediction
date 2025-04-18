#!/bin/bash
#SBATCH --gres=gpu:1              # Number of GPUs
#SBATCH --ntasks 2                # Number of cpus
#SBATCH --mem 8G                  # Size of cpu memory
#SBATCH --time 0-24:00:00         # Max duration days-hours:minutes:seconds
#SBATCH --mail-user=valentin.cuzin-rambaud@etu.univ-lyon1.fr  # Where to send mail
#SBATCH --mail-type=ALL           # Mail events (stop/start job)

# next line is important to stop the submission script at the 1st error
set -e

# activate recent Python
module load Programming_Languages/python/3.12.2

# activate Python environment
source  pytorch-env/bin/activate

# run Python script
python3  my_pytorch_code.py