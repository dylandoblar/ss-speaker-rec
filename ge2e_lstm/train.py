#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import random
import torch
import torch.optim as optim
import numpy as np

from hparam import hparam as hp
from data_load import *
from speech_embedder_net import *


def get_dataloaders(N, M, num_frames):
    train_dataset = VoxCelebDataset(hp.data.train_path, M, num_frames)
    val_dataset = VoxCelebDataset(hp.data.test_path, M, num_frames)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=N, shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=N, shuffle=True, num_workers=4, drop_last=True)
    
    return (train_loader, val_loader, train_dataset, val_dataset)


def compute_EER(model, device, mel_db_batch, N, M):
    assert M % 2 == 0

    enrollment, verification = \
        torch.split(mel_db_batch, int(mel_db_batch.size(1)/2), dim=1)
    enrollment = torch.reshape(enrollment, \
                    (N*M//2, enrollment.size(2), enrollment.size(3)))
    verification = torch.reshape(verification, \
                    (N*M//2, verification.size(2), verification.size(3)))

    perm = random.sample(range(0, verification.size(0)), verification.size(0))
    unperm = list(perm)
    for i,j in enumerate(perm):
        unperm[j] = i

    enrollment = enrollment.to(device)
    verification = verification.to(device)

    verification = verification[perm]
    enrollment_emb = model(enrollment)
    verification_emb = model(verification)
    verification_emb = verification_emb[unperm]

    enrollment_emb = torch.reshape(enrollment_emb, \
                        (N, M//2, enrollment_emb.size(1)))
    verification_emb = torch.reshape(verification_emb, \
                        (N, M//2, verification_emb.size(1)))

    enrollment_centroids = get_centroids(enrollment_emb)
    sim_matrix = get_cossim(verification_emb, enrollment_centroids)

    # Calculate EER
    diff = 1; EER = 0; EER_thresh = 0; EER_FAR = 0; EER_FRR = 0
    
    for thres in [0.01*i+0.5 for i in range(50)]:
        sim_matrix_thresh = sim_matrix>thres
        
        FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(N))])
        /(N-1.0)/(float(M/2))/N)

        FRR = (sum([M/2-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(N))])
        /(float(M/2))/N)
        
        # Save threshold when FAR = FRR (= EER)
        if diff > abs(FAR-FRR):
            diff = abs(FAR-FRR)
            EER = (FAR+FRR)/2
            EER_thresh = thres
            EER_FAR = FAR
            EER_FRR = FRR

    return (EER, EER_thresh, EER_FAR, EER_FRR)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run():
    # Hyperparameters
    num_epochs = hp.train.epochs
    N = hp.train.N    # batch size
    M = hp.train.M    # number of utterances per speaker
    num_frames = hp.train.num_frames  # number of STFT frames to use per d-vector
    
    lr = hp.train.lr  # learning rate
    lr_step = hp.train.lr_step  # step size for learning rate scheduler
    
    output_interval = hp.train.output_interval
    ckpt_interval = hp.train.checkpoint_interval

    # Detect if there is a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda:0"):
        print("Using GPU.")
    else:
        print("Using CPU.")

    # Initialize model and send to device
    model = SpeechEmbedder()
    model = model.to(device)
    if hp.train.restore:
        print("Loading model parameters from", hp.model.model_path)
        model.load_state_dict(torch.load(hp.model.model_path))
    else:
        print("Training model from scratch.")

    # Initialize train and val data loaders
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(N, M, num_frames)
    print("Data loaders initialized.")

    # Set up loss function and optimizer
    criterion = GE2ELoss(device)
    optimizer = optim.SGD([
                    {'params': model.parameters()},
                    {'params': criterion.parameters()}
                ], lr=lr)

    # Decrease learning rate by factor of 0.5 periodically
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.5)

    # Create directory to save checkpoints
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)

    best_val_loss = np.inf
    for epoch in range(1, num_epochs+1):
        print('\nEpoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        model.train()

        # Training phase
        training_loss = 0.0
        for batch_id, mel_db_batch in enumerate(train_loader):
            mel_db_batch = mel_db_batch.to(device)
            mel_db_batch = torch.reshape(mel_db_batch, \
                (N*M, mel_db_batch.size(2), mel_db_batch.size(3)))

            perm = random.sample(range(0, N*M), N*M)
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Pass the inputs through the model
            embeddings = model(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (N, M, embeddings.size(1)))

            # Backpropagate loss
            loss = criterion(embeddings)
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
            optimizer.step()

            training_loss += loss.item()
            
            # Print training loss statistics for the current epoch
            if (batch_id+1) % output_interval == 0:
                msg = "{0}\tBatch: {1}/{2}\tBatch Loss: {3:.4f}\tAv Running Loss: {4:.4f}".format(
                        time.ctime(), batch_id+1, len(train_dataset)//N,
                        loss, training_loss/(batch_id+1))
                print(msg)

        scheduler.step()

        # Print average training loss per batch
        print("\n  Average training loss per batch: {:.4f}".format(
            training_loss/(len(train_dataset)//N)))

        # Validation and evaluation phase
        val_loss = 0.0
        running_EER = 0.0
        model.eval()
        for batch_id, mel_db_batch in enumerate(val_loader):
            # Compute loss on validation set
            val_mel_db_batch = mel_db_batch.to(device)

            val_mel_db_batch = torch.reshape(val_mel_db_batch, \
                (N*M, val_mel_db_batch.size(2), val_mel_db_batch.size(3)))

            perm = random.sample(range(0, N*M), N*M)
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i
            val_mel_db_batch = val_mel_db_batch[perm]

            # Pass the inputs through the model
            embeddings = model(val_mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (N, M, embeddings.size(1)))

            loss = criterion(embeddings)
            val_loss += loss.item()

            # Compute EER for batch
            EER, EER_thresh, EER_FAR, EER_FRR = compute_EER(model, device, mel_db_batch, N, M)
            running_EER += EER

        # Calculate and print average validation loss per batch
        avg_val_loss = val_loss/(len(val_dataset)//N)
        print("  Average validation loss per batch: {:.4f}".format(avg_val_loss))

        # Print average EER per batch
        avg_EER = running_EER/(len(val_dataset)//N)
        print("  EER on validation set: {:.4f}".format(avg_EER))

        # Save model checkpoint if best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if hp.train.checkpoint_dir is not None:
                model.eval().cpu()
                ckpt_filename = "best_epoch_" + str(epoch) + ".model"
                ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_filename)
                torch.save(model.state_dict(), ckpt_model_path)
                model.to(device).train()

        # Model checkpoints
        if hp.train.checkpoint_dir is not None and epoch % ckpt_interval == 0:
            model.eval().cpu()
            ckpt_filename = "ckpt_epoch_" + str(epoch) + ".model"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_filename)
            torch.save(model.state_dict(), ckpt_model_path)
            model.to(device).train()
            current_lr = get_lr(optimizer)
            print(f"Saved model checkpoint for epoch {epoch} to {ckpt_model_path}, \
                   with learning rate {current_lr}.")

    # Save final model
    model.eval().cpu()
    save_filename = "final_epoch_" + str(epoch) + ".model"
    save_model_path = os.path.join(hp.train.checkpoint_dir, save_filename)
    torch.save(model.state_dict(), save_model_path)
    print("\nTraining done. Final model parameters saved at", save_model_path)


if __name__ == "__main__":
    run()

