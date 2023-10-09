

CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --model_t resnet32x4 --model_s resnet8x4 --trial 1
CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --model_t wrn_40_2 --model_s wrn_16_2 --trial 1

CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --model_t resnet32x4 --model_s ShuffleV1 --trial 1
CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --model_t wrn_40_2 --model_s ShuffleV1 --trial 1




v100
CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --batch_size 256 --model_t resnet32x4 --model_s resnet8x4 --trial 2

CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --batch_size 64 --model_t wrn_40_2 --model_s wrn_16_2 --trial 2

CUDA_VISIBLE_DEVICES=2 python train_student_ds.py --batch_size 256 --model_t resnet32x4 --model_s ShuffleV1 --trial 2

CUDA_VISIBLE_DEVICES=2 python train_student_ds.py --batch_size 128 --model_t wrn_40_2 --model_s ShuffleV1 --trial 2


CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --model_t resnet56 --model_s resnet20 --trial 1

CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --model_t resnet110 --model_s resnet20 --trial 1

CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --model_t wrn_40_2 --model_s wrn_40_1 --trial 1



CUDA_VISIBLE_DEVICES=0 python train_student_ds.py --batch_size 256 --model_t resnet32x4 --model_s resnet8x4 --trial 3



V100(使用特征)
CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 -a 1 -b 0.8 --trial 1

CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_16_2 --low-dim 128 -a 1 -b 0.8 --trial 1

CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --model_s resnet20 --low-dim 64 -a 1 -b 0.8 --trial 1

CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --model_s resnet20 --low-dim 64 -a 1 -b 0.8 --trial 1

CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --model_s resnet32 --low-dim 64 -a 1 -b 0.8 --trial 1

CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s vgg8 --low-dim 512 -a 1 -b 0.8 --trial 1

参数改变：
CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_16_2 --low-dim 128 -a 1 -b 5 -c 1 --trial 2 --batch_size 64

CUDA_VISIBLE_DEVICES=2 python train_student_pcl.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 -a 1 -b 5 -c 1 --trial 4 --batch_size 64

CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --low-dim 100 -a 1 -b 5 -c 1 --trial 3 --batch_size 64

CUDA_VISIBLE_DEVICES=2 python train_student_pcl.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --model_s resnet20 --low-dim 64 -a 1 -b 5 -c 1 --trial 2 --batch_size 64

CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --model_s resnet20 --low-dim 64 -a 1 -b 5 -c 1 --trial 2 --batch_size 64

CUDA_VISIBLE_DEVICES=1 python train_student_pcl.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --model_s resnet32 --low-dim 64 -a 1 -b 5 -c 1 --trial 2 --batch_size 64



//ablation
CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 -a 1 -b 0 -c 0.8 --trial 1



logit
CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --low-dim 100 -a 1 -b 0.8 -c 0.1 --trial 1

CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV2 --low-dim 100 -a 1 -b 5 -c 1 --trial 2

CUDA_VISIBLE_DEVICES=2 python train_student_pcl.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --low-dim 100 -a 1 -b 5 -c 1 --trial 1

CUDA_VISIBLE_DEVICES=2 python train_student_pcl.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --low-dim 100 -a 1 -b 5 -c 1 --trial 1 --batch_size 64

CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --low-dim 100 -a 1 -b 10 -c 1 --trial 1

CUDA_VISIBLE_DEVICES=2 python train_student_pcl.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --model_s vgg8 --low-dim 100 -a 1 -b 5 -c 1 --trial 1 --batch_size 64

参数改变：
CUDA_VISIBLE_DEVICES=1 python train_student_pcl.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --low-dim 100 -a 1 -b 5 -c 1 --trial 3 --batch_size 64








pcl2--mean student teacher

CUDA_VISIBLE_DEVICES=0 python train_student_pcl2.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 -a 1 -b 0.8 --trial 5

CUDA_VISIBLE_DEVICES=0 python train_student_pcl2.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 -a 1 -b 0.8 --trial 6




V100（使用最后一层）
CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --low-dim 100 -a 1 -b 0.8 --trial 2

CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 --low-dim 100 -a 1 -b 0.8 --trial 4


python create_caltech101_data_files.py -i /path/to/caltech101/ -o /output_path/caltech101 -d


74.86
CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 -a 1 -b 1 -c 1 --trial 2 --low-dim 100




CUDA_VISIBLE_DEVICES=0 python  train.py --dataset cifar100 --clip_model ViT-B/16


CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 train.py --dataset cifar100 --clip_model ViT-B/16 --batch_size 32



**************dkd****


CUDA_VISIBLE_DEVICES=2 python train_student_pcl.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_16_2 --low-dim 128 -a 1 -b 5 -c 1 --trial 10 --batch_size 64 --dkd

CUDA_VISIBLE_DEVICES=2 python train_student_pcl.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --low-dim 100 -a 1 -b 5 -c 1 --trial 10 --batch_size 64 --dkd


CUDA_VISIBLE_DEVICES=2 python train_student_pcl.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_40_1 --low-dim 100 -a 1 -b 5 -c 1 --trial 10 --batch_size 64 --dkd --dkd_beta 6.0


CUDA_VISIBLE_DEVICES=1 python train_student_pcl.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --model_s resnet20 --low-dim 64 -a 1 -b 0.8 --trial 10 --batch_size 64 --dkd --dkd_beta 2.0


CUDA_VISIBLE_DEVICES=2 python train_student_pcl.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --model_s resnet20 --low-dim 64 -a 1 -b 0.8 --trial 10 --batch_size 64 --dkd --dkd_alpha 2.0 --dkd_beta 1.0

CUDA_VISIBLE_DEVICES=2 python train_student_pcl.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --model_s resnet32 --low-dim 64 -a 1 -b 0.8 -c 1 --trial 10 --batch_size 64 --dkd --dkd_alpha 2.0 --dkd_beta 1.0



CUDA_VISIBLE_DEVICES=2 python train_student_pcl.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --low-dim 100 -a 1 -b 5 -c 1 --trial 10 --batch_size 64 --dkd --dkd_beta 6.0

CUDA_VISIBLE_DEVICES=2 python train_student_pcl.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --model_s MobileNetV2 --low-dim 100 -a 1 -b 5 -c 1 --trial 10 --batch_size 64 --dkd --dkd_beta 8.0


CUDA_VISIBLE_DEVICES=1 python train_student_pcl.py --path_t ./save/models/ResNet50_vanilla/ckpt_epoch_240.pth --model_s vgg8 --low-dim 100 -a 1 -b 5 -c 1 --trial 10 --batch_size 64 --dkd --dkd_beta 8.0

CUDA_VISIBLE_DEVICES=1 python train_student_pcl.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s ShuffleV2 --low-dim 100 -a 1 -b 5 -c 1 --trial 10 --batch_size 64 --dkd --dkd_beta 8.0

CUDA_VISIBLE_DEVICES=1 python train_student_pcl.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s ShuffleV1 --low-dim 100 -a 1 -b 5 -c 1 --trial 10 --batch_size 64 --dkd --dkd_beta 8.0

CUDA_VISIBLE_DEVICES=1 python train_student_pcl.py --path_t ./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --model_s resnet8x4 -a 1 -b 5 -c 1 --trial 10 --batch_size 64 --dkd --dkd_alpha 8.0 --dkd_beta 1.0

CUDA_VISIBLE_DEVICES=1 python train_student_pcl.py --path_t ./save/models/vgg13_vanilla/ckpt_epoch_240.pth --model_s vgg8 --low-dim 100 -a 1 -b 5 -c 1 --trial 10 --batch_size 64 --dkd --dkd_alpha 6.0 --dkd_beta 1.0

CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/resnet56_vanilla/ckpt_epoch_240.pth --model_s resnet20 --low-dim 64 -a 1 -b 0.8 --trial 10 --batch_size 64 --dkd --dkd_alpha 2.0 --dkd_beta 1.0

CUDA_VISIBLE_DEVICES=0 python train_student_pcl.py --path_t ./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth --model_s wrn_16_2 --low-dim 128 -a 1 -b 5 -c 1 --trial 10 --batch_size 64 --dkd --dkd_alpha 6.0 --dkd_beta 1.0

