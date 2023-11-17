# SCAGE

## What is SCAGE?、

SCAGE is a self-conformation-aware pre-training framework for molecular property prediction reveals the quantitate structure-activity relationship like human experts

## Some commad line 

### 0. fix some bugs

```bash
export LD_LIBRARY_PATH="/home2/s439850/anaconda3/envs/SAGE/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home2/s439850/anaconda3/envs/SAGEHH/lib:$LD_LIBRARY_PATH"
```

### 1. Pre-training

```bash
CUDA_VISIBLE_DEVICES=3 python pretrain.py
```

```bash
python -m torch.distributed.launch --nproc_per_node=4 pretrain_dis.py
```

### 2. Fine-tuning

```bash
unset http_proxy
unset https_proxy

nnictl create --config ./config.yaml -p 3325
```