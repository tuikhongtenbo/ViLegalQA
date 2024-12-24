#!/bin/sh
#SBATCH --job-name=VinaLLaMA-task-1           # Job name
#SBATCH --gres=gpu:1                          # 1 GPU
#SBATCH -N 1                                  # 1 node
#SBATCH --ntasks=1                            # Total task
#SBATCH --cpus-per-task=2                     # CPU each task
#SBATCH --mem=50G                             # Max mem
#SBATCH -o Vinallama.out                      # Output file

# GPU
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export HF_HUB_ENABLE_CACHE=True
export CUDA_VISIBLE_DEVICES=7     

# Install pakages in requirements.txt
pip install --upgrade pip setuptools wheel    
pip install -r /data/npl/MRC/ViLegalQA/Task1/requirements.txt  

# Running section
python /data/npl/MRC/ViLegalQA/Task1/main.py \
  --test_data_path="/data/npl/MRC/ViLegalQA/Task1/test_data.json" \
  --output_path="/data/npl/MRC/ViLegalQA/Task1/results_vinallama.json" \
  --model_name="/data/npl/MRC/ViLegalQA/Task1/Vinallama-2.7-chat"