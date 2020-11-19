#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

exp_name = 'exp_pase_frames-167_bs-256_lr-0.005_ep-100'
log_fname = '/home/dylan/ss-speaker-rec/sanity_check/logs/' + exp_name + '.csv'
save_dir = '/home/dylan/ss-speaker-rec/sanity_check/plots'
os.makedirs(save_dir, exist_ok=True)

data = np.genfromtxt(log_fname, delimiter=',', skip_header=1)

train_acc = data[:,0]
val_acc = data[:,1]
train_loss = data[:,2]
val_loss = data[:,3]

plt.plot(list(range(1, len(train_acc) + 1)), train_acc, label='Training Accuracy')
plt.plot(list(range(1, len(val_acc) + 1)), val_acc, label='Validation Accuracy')
plt.title(f'{exp_name} Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
save_path = os.path.join(save_dir, f'acc_{exp_name}.png')
if not os.path.exists(save_path):
    plt.savefig(save_path)
else:
    print(f'A file already exists at {save_path}')
plt.close()

plt.plot(list(range(1, len(train_loss) + 1)), train_loss, label='Training Loss')
plt.plot(list(range(1, len(val_loss) + 1)), val_loss, label='Validation Loss')
plt.title(f'{exp_name} Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
save_path = os.path.join(save_dir, f'loss_{exp_name}.png')
if not os.path.exists(save_path):
    plt.savefig(save_path)
else:
    print(f'A file already exists at {save_path}')
plt.close()

