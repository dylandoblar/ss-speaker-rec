#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# from hparam import hparam as hp
from data_load import VoxCelebClassificationDataset
from models import LinearNet, OneHiddenLayerReluNet


def main():
    ''' Train a linear model for k-way classification on all speakers '''

    data_path = '/mnt/disks/data/pase_representations'
    log_dir = '/home/dylan/ss-speaker-rec/sanity_check/logs'
    # data_path = '/mnt/disks/data/filterbanks'
    use_pase = 'pase' in data_path
    num_frames = 167 if use_pase else 180
    batch_size = 256 # 128 # 64 #  512
    num_epochs = 100
    lr = 5e-3
    num_workers = 5 # num workers for the dataloaders
    exp_name = f"exp_{'pase' if use_pase else 'fbank'}_frames-{num_frames}_bs-{batch_size}_lr-{lr}_ep-{num_epochs}"

    full_dataset = VoxCelebClassificationDataset(data_path, num_frames, use_pase)

    train_size = int(0.9 * len(full_dataset))
    debug_size = 0
    # debug_size = int(0.93 * len(full_dataset)) # uncomment if debug set desired
    val_size = len(full_dataset) - train_size - debug_size
    if debug_size != 0:
        train_dataset, val_dataset, debug_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size, debug_size],
            # generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            # generator=torch.Generator().manual_seed(42)
        )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True
    )

    if False: # profile dataloader speed
        start_time = time.time()
        for batch_idx, (Xs, ys) in enumerate(train_loader):
            if batch_idx == 4:
                break
        t = (time.time()-start_time)/5
        print(f"t : {t}")
        print(f"batch_size : {batch_size}")
        print(f"num_workers : {num_workers}")
        print(f"t/batch_size : {t/batch_size}")
        return None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device : {device}")
    print(f"torch.cuda.get_device_name(0) : {torch.cuda.get_device_name(0)}")

    model = LinearNet(num_feature=full_dataset[0][0].size()[0], num_class=len(full_dataset.speaker_ids))

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    # Start logging training stats
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, exp_name+'.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['train_acc', 'val_acc', 'train_loss', 'val_loss'])


    print("Begin training.")
    for e in tqdm(range(1, num_epochs+1)):

        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for X_train_batch, y_train_batch in tqdm(train_loader):
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()


        # VALIDATION    
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        # log training stats
        with open(os.path.join(log_dir, exp_name+'.csv'), 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                train_epoch_acc/len(train_loader),
                val_epoch_acc/len(val_loader),
                train_epoch_loss/len(train_loader),
                val_epoch_loss/len(val_loader),
            ])

        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    return acc


if __name__ == "__main__":
    main()
