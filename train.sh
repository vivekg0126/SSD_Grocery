source activate pytorch
export CUDA_VISIBLE_DEVICES=2

#python train.py -model_save checkpoint_45mAP.pth.tar -lr 0.002 -batch_size 30 \
#      -num_workers 3 -data /home/badri/workspace_vk/gr_ds

nohup python train.py -model_save ckpt4.pth.tar -lr 0.002 -batch_size 32  \
      -num_workers 2  -data /home/badri/workspace_vk/gr_ds > nohup3.out &