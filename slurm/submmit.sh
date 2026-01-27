#!/bin/bash
#SBATCH -o job.%j.out         # 输出文件的名称，%j 会被替换为作业ID
#SBATCH -J myGpuJob           # 作业名称
#SBATCH --partition=gpu       # 指定使用 gpu 分区
#SBATCH --nodes=1             # 请求 1 个节点
#SBATCH --ntasks-per-node=1   # 在该节点上运行 1 个任务
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1# !!! 关键：请求分配 1 个 GPU 资源 !!!

echo "Job ID: $SLURM_JOB_ID"
echo "Job is running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

echo "Loading modules..."
module avail cudnn
module load gcc/9.2.0
module load cuda11.2/toolkit/11.2.0
module load cudnn8.1-cuda11.2/8.1.1.33
# openmpi 可能不是必须的，除非你的Python包需要它来编译
# module load openmpi/gcc/64/1.10.7 

# --- 激活 Conda 环境 (在Slurm脚本中必须这样做) ---
echo "Activating conda environment..."
# 首先，初始化 Conda 的 Shell 功能
eval "$(conda shell.bash hook)"
# 然后再激活你的环境
conda activate keras

# 检查 Python 和 GPU 工具的路径，确保环境正确
echo "Python executable: $(which python)"
echo "NVIDIA SMI output:"
nvidia-smi

# --- 运行您的程序 ---
echo "Starting the training script..."
# 使用 python 直接运行即可，srun 在这里不是必须的，因为你只请求了1个任务
python /home/lizhaolab/ct_seg/scripts/train_bbunet.py

echo "Job finished."