#!/bin/bash


#kernprof -v -l train.py --batch_size 32 --epoch 50 --save_path ckpt_PASE_50h_newMI_noW_QRNN_100addrev_dev-version \
python -u -W ignore train.py --batch_size 32 --epoch 50 --save_path ckpt_PASE_50h_QRNN_100addrev_25overlap_5reg_r3 \
       	--num_workers 25 --warmup 10000000 --net_cfg cfg/workers5reg_r3.cfg \
	--fe_cfg cfg/PASE_dense_QRNN.cfg --do_eval --data_cfg data/librispeech_data_50h.cfg \
	--min_lr 0.0005 --fe_lr 0.0005 --data_root data/LibriSpeech_50h/all/ \
	--dtrans_cfg cfg/distortions/100addrev_25overlap.cfg \
	--stats data/librispeech_50h_stats_5.pkl --lr_mode poly \
	--chunk_size 24000 --random_scale True --tensorboard True \
	--fe_opt radam --min_opt radam

# --log_grad_keys rnn \

