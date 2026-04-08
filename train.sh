#!/bin/bash
#base
MODEL=mamba_vision_T
DATA_PATH_TRAIN="/home/soybean_dataset_spilt/train"
DATA_PATH_VAL="/home/soybean_dataset_spilt/val"
BS=8
EXP=my_experiment
LR=0.001
WD=5e-5
DR=0
DATA_DIR="/home/soybean_dataset_spilt"
PRETRAIN_DIR="/home/MambaVision/mambavision_tiny_1k.pth.tar"
#PRETRAIN_DIR="/home/MambaVision/output/train/my_experiment/20260107-103630-mamba_vision_T-224/model_best.pth.tar"

torchrun --nproc_per_node=1 train.py --input-size 3 224 224 --crop-pct=0.875 --data_dir=$DATA_DIR\
 --train-split=$DATA_PATH_TRAIN --val-split=$DATA_PATH_VAL --model $MODEL  --initial-checkpoint $PRETRAIN_DIR --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR
#base_vmamba
MODEL=mamba_vision_T
DATA_PATH_TRAIN="/home/soybean_dataset_spilt/train"
DATA_PATH_VAL="/home/soybean_dataset_spilt/val"
BS=8
EXP=my_experiment
LR=0.001
WD=5e-5
DR=0
DATA_DIR="/home/soybean_dataset_spilt"
PRETRAIN_DIR="/home/MambaVision/mambavision_tiny_1k.pth.tar"

torchrun --nproc_per_node=1 train.py --input-size 3 224 224 --crop-pct=0.875 --data_dir=$DATA_DIR\
 --train-split=$DATA_PATH_TRAIN --val-split=$DATA_PATH_VAL --model $MODEL --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR

#diffusemix
MODEL=mamba_vision_T
DATA_PATH_TRAIN="/home/blended"
DATA_PATH_VAL="/home/soybean_dataset_spilt/val"
BS=8
EXP=my_experiment
LR=0.001
WD=5e-5
DR=0
DATA_DIR="/home/soybean_dataset_spilt"
PRETRAIN_DIR="/home/MambaVision/mambavision_tiny_1k.pth.tar"

torchrun --nproc_per_node=1 train.py --input-size 3 224 224 --crop-pct=0.875 --data_dir=$DATA_DIR\
 --train-split=$DATA_PATH_TRAIN --val-split=$DATA_PATH_VAL --model $MODEL  --initial-checkpoint $PRETRAIN_DIR --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR
#2
MODEL=mamba_vision_T
DATA_PATH_TRAIN="/home/blended"
DATA_PATH_VAL="/home/soybean_dataset_spilt/val"
BS=8
EXP=my_experiment
LR=0.001
WD=5e-5
DR=0
DATA_DIR="/home/soybean_dataset_spilt"
PRETRAIN_DIR="/home/MambaVision/output/train/my_experiment/20250526-095339-mamba_vision_T-224/model_best.pth.tar"

torchrun --nproc_per_node=1 train.py --input-size 3 224 224 --crop-pct=0.875 --data_dir=$DATA_DIR\
 --train-split=$DATA_PATH_TRAIN --val-split=$DATA_PATH_VAL --model $MODEL  --initial-checkpoint $PRETRAIN_DIR --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR



#soynet_base
MODEL=mamba_vision_T
DATA_PATH_TRAIN="/home/soynet_spilt/train"
DATA_PATH_VAL="/home/soynet_spilt/val"
BS=8
EXP=my_experiment
LR=0.001
WD=5e-5
DR=0
DATA_DIR="/home/soynet_spilt"
PRETRAIN_DIR="/home/MambaVision/mambavision_tiny_1k.pth.tar"

torchrun --nproc_per_node=1 train.py --input-size 3 224 224 --crop-pct=0.875 --data_dir=$DATA_DIR\
 --train-split=$DATA_PATH_TRAIN --val-split=$DATA_PATH_VAL --model $MODEL  --initial-checkpoint $PRETRAIN_DIR --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR




#soybean_leaf_dataset_base
MODEL=mamba_vision_T
DATA_PATH_TRAIN="/home/soybean_leaf_dataset_spilt/train"
DATA_PATH_VAL="/home/soybean_leaf_dataset_spilt/val"
BS=8
EXP=my_experiment
LR=0.001
WD=5e-5
DR=0
DATA_DIR="/home/soybean_leaf_dataset_spilt"
PRETRAIN_DIR="/home/MambaVision/mambavision_tiny_1k.pth.tar"

torchrun --nproc_per_node=1 train.py --input-size 3 224 224 --crop-pct=0.875 --data_dir=$DATA_DIR\
 --train-split=$DATA_PATH_TRAIN --val-split=$DATA_PATH_VAL --model $MODEL  --initial-checkpoint $PRETRAIN_DIR --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR



#soybean_leaf_dataset_base
MODEL=mamba_vision_T
DATA_PATH_TRAIN="/home/soybean_leaf_dataset_spilt/train"
DATA_PATH_VAL="/home/soybean_leaf_dataset_spilt/val"
BS=8
EXP=my_experiment
LR=0.001
WD=5e-5
DR=0
DATA_DIR="/home/soybean_leaf_dataset_spilt"
PRETRAIN_DIR="/home/MambaVision/output/train/my_experiment/soybean_leaf_dataset_inceptionnext_pre/model_best.pth.tar"


torchrun --nproc_per_node=1 train.py --input-size 3 224 224 --crop-pct=0.875 --data_dir=$DATA_DIR\
 --train-split=$DATA_PATH_TRAIN --val-split=$DATA_PATH_VAL --model $MODEL  --initial-checkpoint $PRETRAIN_DIR --amp --weight-decay ${WD} --drop-path ${DR} --batch-size $BS --tag $EXP --lr $LR

