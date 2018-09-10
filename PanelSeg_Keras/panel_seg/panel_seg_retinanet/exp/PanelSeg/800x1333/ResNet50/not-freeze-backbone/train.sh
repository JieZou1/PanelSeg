#!/bin/bash
	
#SBATCH --workdir=/slurm_storage/jzou/programs/PanelSeg/PanelSeg_Keras/panel_seg/panel_seg_retinanet/exp/PanelSeg/800x1333/ResNet50/not-freeze-backbone
#SBATCH --output=/slurm_storage/jzou/programs/PanelSeg/PanelSeg_Keras/panel_seg/panel_seg_retinanet/exp/PanelSeg/800x1333/ResNet50/not-freeze-backbone/slurm_%j.out
#SBATCH --error=/slurm_storage/jzou/programs/PanelSeg/PanelSeg_Keras/panel_seg/panel_seg_retinanet/exp/PanelSeg/800x1333/ResNet50/not-freeze-backbone/slurm_%j.error
#SBATCH --job-name=panel_seg_retinanet_train_panelseg_800_1333_resnet50_not-freeze-backbone
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx1
#SBATCH --mem-bind=local

export LD_LIBRARY_PATH=/slurm_storage/public/cuda9.0/lib64

# env
which python
python /slurm_storage/jzou/programs/PanelSeg/PanelSeg_Keras/panel_seg/panel_seg_retinanet/bin/train.py\
    --dataset_type csv\
    --classes /slurm_storage/jzou/programs/PanelSeg/PanelSeg_Keras/panel_seg/panel_seg_retinanet/exp/PanelSeg/mapping.csv\
    --l_classes /slurm_storage/jzou/programs/PanelSeg/PanelSeg_Keras/panel_seg/panel_seg_retinanet/exp/PanelSeg/label_mapping.csv\
    --annotations /slurm_storage/jzou/programs/PanelSeg/PanelSeg_Keras/panel_seg/panel_seg_retinanet/exp/PanelSeg/train_slurm.csv\
    --val-annotations /slurm_storage/jzou/programs/PanelSeg/PanelSeg_Keras/panel_seg/panel_seg_retinanet/exp/PanelSeg/eval_slurm.csv\
    --epoch 50\
    --backbone resnet50\
    --image-min-side 800\
    --image-max-side 1333
