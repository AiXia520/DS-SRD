# DS-SRD: A Unified Framework for Structured Representation Distillation

To improve the representation performance of smaller models, representation distillation has been investigated to transfer structured knowledge from a larger model (teacher) to a smaller model (student). Existing works aim to maximize a lower bound on mutual information to transfer global structure knowledge, thus ignoring the local structured semantic transfer from teacher representation. We propose a unified framework for Structured Representation Distillation with the Double Student training mechanism, named DS-SRD, which focuses on transferring global and local structured consistent representation between teacher and student. The motivation is that a good teacher network could construct a well-structured feature space in terms of global and local dependencies. DS-SRD makes the student mimic better structured semantic relations from the teacher, thus improving not only the representation performance but also can easily extend to different distillation tasks: supervised representation distillation and self-supervised representation distillation. In addition, we also designed a simple CNN-Transformer structure based on DS-SRD to make the CNN encoder attentive via transformer guidance. Through extensive experiments, we validate the effectiveness of our method compared to various supervised and self-supervised representation distillation baselines. 

# Self-supervised representation distillation
### Environment (Python 3.8.12)
```
# Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh

# Install required packages
pip install pytorch-gpu
pip install scikit-learn
pip install numpy
pip install scipy
```
# Data
- CIFAR-10/100 will automatically be downloaded.
- For ImageNet, please refer to the [[PyTorch ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet)]. The folder structure should be like `data/imagenet/train/n01440764/`
- For speech commands, run `bash speech_commands/download_speech_commands_dataset.sh`.
- For tabular datasets, download [[covtype.data.gz](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz)] and [[HIGGS.csv.gz](https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz)], and place them in `data/`. They are processed when first loaded.

### Training:
```
CUDA_VISIBLE_DEVICES=0 python main_lincls_ws.py 'data' --dataset cifar10  -a resnet18 --start-eval 80 --lr 10 --pretrained '/home/xyl/mywork/imix-distill/save/cifar10/resnet34_byol_m0.999_proj_mlpbn1_pred_mlpbn1_shufflebn_imixup1.0_lr0.125_wd1.0e-04_bsz256_ep2000_cos_warm_trial2/checkpoint_2000.pth'
```

# Supervised representation distillation

## Running

1. Fetch the pretrained teacher models by:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
which will download and save the models to `save/models`

2. Run distillation by following commands
    ```
     CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --model_t resnet32x4 --model_s resnet8x4 --trial 1
    CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --model_t wrn_40_2 --model_s wrn_16_2 --trial 1
    
    CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --model_t resnet32x4 --model_s ShuffleV1 --trial 1
    CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --model_t wrn_40_2 --model_s ShuffleV1 --trial 1
    ```

# DS-SRD with data augmentation

## Running

```
   CUDA_VISIBLE_DEVICES=0 python train_student_ds_srd.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0.8 -c 0.8 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode identity --project kd__wrn_40_2wrn_16_2__cifar100__identity --warmup_epoch 10 --save_entropy_log_step 0 --low-dim 128 --epochs 240

CUDA_VISIBLE_DEVICES=0 python train_student_ds_srd.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0.8 -c 0.8 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode flip --project kd__wrn_40_2wrn_16_2__cifar100__flip --warmup_epoch 10 --save_entropy_log_step 0 --low-dim 128 --epochs 240

CUDA_VISIBLE_DEVICES=0 python train_student_ds_srd.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0.8 -c 0.8 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode crop+flip --project kd__wrn_40_2wrn_16_2__cifar100__cropflip --warmup_epoch 10 --save_entropy_log_step 0 --low-dim 128 --epochs 240

CUDA_VISIBLE_DEVICES=0 python train_student_ds_srd.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0.8 -c 0.8 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutout --project kd__wrn_40_2wrn_16_2__cifar100__cutout --warmup_epoch 10 --save_entropy_log_step 0 --low-dim 128 --epochs 240

CUDA_VISIBLE_DEVICES=0 python train_student_ds_srd.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0.8 -c 0.8 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode autoaugment --project kd__wrn_40_2wrn_16_2__cifar100__autoaugment --warmup_epoch 10 --save_entropy_log_step 0 --low-dim 128 --epochs 240

CUDA_VISIBLE_DEVICES=0 python train_student_ds_srd.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0.8 -c 0.8 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode mixup --project kd__wrn_40_2wrn_16_2__cifar100__mixup --warmup_epoch 10 --save_entropy_log_step 0 --low-dim 128 --epochs 240

CUDA_VISIBLE_DEVICES=0 python train_student_ds_srd.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0.8 -c 0.8 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix --project kd__wrn_40_2wrn_16_2__cifar100__cutmix --warmup_epoch 10 --save_entropy_log_step 0 --low-dim 128 --epochs 240

CUDA_VISIBLE_DEVICES=0 python train_student_ds_srd.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0.8 -c 0.8 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion student_entropy --project kd__wrn_40_2wrn_16_2__cifar100__cutmix_pick_Sentropy --warmup_epoch 10 --save_entropy_log_step 0 --low-dim 128 --epochs 240

CUDA_VISIBLE_DEVICES=0 python train_student_ds_srd.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --distill crd --model_s wrn_16_2 -r 0.1 -a 0.9 -b 0.8 -c 0.8 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix_pick --mix_n_run 2 --cutmix_pick_criterion teacher_entropy --project kd__wrn_40_2wrn_16_2__cifar100__cutmix_pick_Tentropy --warmup_epoch 10 --save_entropy_log_step 0 --low-dim 128 --epochs 240

```
Our code is largely borrowed from [RepDistiller]、[Good-DA-in-KD]


