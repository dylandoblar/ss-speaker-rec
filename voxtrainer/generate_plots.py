#! /usr/bin/python
# -*- encoding: utf-8 -*-
# generates plots from the output log file of a training job

import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


def gen_plots(scores_fname, out_dir, exp_name):
    if exp_name:
        exp_name = f'{exp_name} '
    else:
        exp_name = ''
    epoch = []
    train_prec = []
    train_loss = []
    val_eer = []
    with open(scores_fname) as f:
        reader = csv.reader(f)
        for row in reader:
            epoch.append(int(row[0].split(' ')[-1]))
            train_prec.append(float(row[2].strip().split(' ')[-1]))
            train_loss.append(float(row[3].strip().split(' ')[-1]))
            val_eer.append(float(row[4].strip().split(' ')[-1]))

    # make output dir
    os.makedirs(out_dir, exist_ok=True)

    # training loss
    plt.plot(epoch, train_loss, label='Training Loss')
    plt.title(f'{exp_name}Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    save_path = os.path.join(out_dir, 'train_loss.png')
    if not os.path.exists(save_path):
        plt.savefig(save_path)
    else:
        print(f'A file already exists at {save_path}')
    plt.close()

    # validation EER
    min_val_eer = min(val_eer)
    plt.plot(epoch, val_eer, color='green', label='Validation EER')
    plt.hlines(min_val_eer, min(epoch), max(epoch), colors='red', linestyles='dashed',
               label=f'Minimum Validation EER: {round(min_val_eer, 2)}%')
    plt.title(f'{exp_name}Validation EER')
    plt.xlabel('Epochs')
    plt.ylabel('EER (%)')
    plt.legend()
    # plt.show()
    save_path = os.path.join(out_dir, 'val_eer.png')
    if not os.path.exists(save_path):
        plt.savefig(save_path)
    else:
        print(f'A file already exists at {save_path}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training loss and validation EER')
    parser.add_argument('--scores_fname', help='fname of log file  with scores')
    parser.add_argument('--out_dir', help='directory name to store plots')
    parser.add_argument('--exp_name', help='name of experiment (used in plot title)')
    args = parser.parse_args()

    gen_plots(args.scores_fname, args.out_dir, args.exp_name)
