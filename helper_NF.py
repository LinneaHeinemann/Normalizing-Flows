import time
import os
import sys
import copy
import numpy as np
import torch
import torch.optim as optim
import argparse
import torch.nn.functional as F
import glob
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib import pyplot as plt
import jammy_flows
from scipy.stats import norm


def normalize(labels, p):
    """
    Normalize the input labels using percentile-based scaling.

    This function scales the input labels to a range of [0, 1] based on the specified percentiles.
    The scaling is done by computing the percentiles of the labels and then normalizing the labels
    using these percentile values.

    Parameters:
    labels (np.ndarray): The input labels to be normalized.
    p (float): The percentile value used for scaling. The function uses the p-th and (1-p)-th percentiles
               for normalization.

    Returns:
    tuple: A tuple containing the normalized labels and the range used for normalization.
           - normalized_labels (np.ndarray): The normalized labels.
           - ranges (np.ndarray): The range used for normalization, which includes the p-th and (1-p)-th percentiles.
    """
    ranges = np.percentile(labels, [100 * p, 100 * (1 - p)], axis=0)
    labels = (labels - ranges[0]) / (ranges[1] - ranges[0])
    return labels, ranges

# Function to denormalize the labels back to their original scale
def denormalize(labels, ranges):
    """
    Denormalize the input labels using the specified range.

    This function denormalizes the input labels using the specified range values.
    The denormalization is done by scaling the labels back to the original range
    using the provided range values.

    Parameters:
        labels (np.ndarray): The normalized labels to be  denormalized.
        ranges (np.ndarray): The range values used for normalization.

    Returns:
        np.ndarray: The denormalized labels.
    """
    return labels * (ranges[1] - ranges[0]) + ranges[0]

def denormalize_std(uncertainty, ranges):
    """
    Denormalizes the given uncertainty predictions using the provided range.

    It is different to the denormalization of the labels which also includes a shift.

    Parameters
    ----------
    uncertainty : array-like
        The normalized uncertainty to be denormalized.
    ranges : array-like
        A two-element array-like object where the first element is the minimum value
        and the second element is the maximum value of the original range.
    Returns
    -------
    array-like
        The denormalized uncertainty.
    """

    return uncertainty * (ranges[1] - ranges[0])



def get_normalized_data(data_path, return_SNR=False):
    """
    Load and normalize spectra and label data from the given path.
    Parameters
    ----------
    data_path : str
        The path to the directory containing the spectra and labels data files.
    Returns
    -------
    spectra : numpy.ndarray
        The normalized spectra data.
    labels : numpy.ndarray
        The normalized labels data (t_eff, log_g, fe_h).
    spectra_length : int
        The length of the spectra.
    n_labels : int
        The number of labels used (should be 3).
    labelNames : list of str
        The names of the labels used (t_eff, log_g, fe_h).
    ranges : numpy.ndarray
        The ranges used for normalization of the labels.
    """

    # Load the spectra data
    spectra = np.load(f"{data_path}/spectra.npy")
    spectra_length = spectra.shape[1]

    # Load the labels data
    # labels: mass, age, l_bol, dist, t_eff, log_g, fe_h
    labelNames = ["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h"]
    labels = np.load(f"{data_path}/labels.npy")
    SNR = labels[:, -1]
    labels = labels[:, :-1]

    # We only use the labels: t_eff, log_g, fe_h
    labelNames = labelNames[-3:]
    labels = labels[:, -3:]
    n_labels = labels.shape[1]

    labels, ranges = normalize(labels, 0.05)

    # Normalize spectra
    spectra = np.log(np.maximum(spectra, 0.2))

    if return_SNR:
        return spectra, labels, spectra_length, n_labels, labelNames, ranges, SNR
    return spectra, labels, spectra_length, n_labels, labelNames, ranges

def plot_pdf(model, test_loader, device, filename="pdf.png"):
    model.eval()

    x, y = next(iter(test_loader))

    x = x.to(device)
    y = y.to(device)

    model.visualize_pdf(
        input_data=x,
        filename=filename,
        batch_index=0,
        truth=y[0].detach().cpu()
    )


def plot_loss(train_losses, val_losses, title="Loss"):
    plt.figure(figsize=(6,4))

    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()

def nf_loss(inputs, batch_labels, model):
    """
    Computes the loss for a normalizing flow model.

    Parameters
    ----------
    inputs : torch.Tensor
        The input data to the model.
    batch_labels : torch.Tensor
        The labels corresponding to the input data.
    model : torch.nn.Module
        The normalizing flow model used for evaluation.
    Returns
    -------
    torch.Tensor
        The computed loss value.
    """
    log_pdfs = model.log_pdf_evaluation(batch_labels, inputs) # get the probability of the labels given the input data
    loss = -log_pdfs.mean() # take the negative mean of the log probabilities
    return loss


def test_model(model, test_loader, device):
    model.eval()
    model = model.to(device)

    total_nll = 0.0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            log_pdf = model.log_pdf_evaluation(y, x)
            total_nll += -log_pdf.mean().item()

    test_nll = total_nll / len(test_loader)
    print("TEST NLL:", test_nll)

    return test_nll

def train_model(model, train_loader, val_loader, device, epochs, lr, patience=10):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_no_improvement = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nf_loss(x, y, model)

            if not torch.isfinite(loss):  # skip bad batches bc some were nan 
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent exploding gradients
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation 
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = nf_loss(x, y, model)

                if not torch.isfinite(loss):  # skip bad batches
                    continue

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train NLL: {train_loss:.4f} | Val NLL: {val_loss:.4f}")

        #Early stopping 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return train_losses, val_losses



def evaluate_uncertainty(model, test_loader, device):
    model.eval()
    model.to(device)

    all_means = []
    all_stds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)  #  3 means + 3 stds

            means = output[:, :3]
            stds = output[:, 3:]

            all_means.append(means.cpu())
            all_stds.append(stds.cpu())
            all_targets.append(y.cpu())

    means = torch.cat(all_means)
    stds = torch.cat(all_stds)
    targets = torch.cat(all_targets)

    # Calibration 
    within_1sigma = ((targets >= means - stds) & (targets <= means + stds)).float()
    within_2sigma = ((targets >= means - 2*stds) & (targets <= means + 2*stds)).float()

    print("Coverage (±1σ):", within_1sigma.mean().item())
    print("Coverage (±2σ):", within_2sigma.mean().item())

    # Error vs uncertainty 
    errors = torch.abs(means - targets)

    corr = torch.corrcoef(torch.stack([
        errors.view(-1),
        stds.view(-1)
    ]))[0,1]

    print("Correlation(error, std):", corr.item())

    # Standardized residuals 
    z = (targets - means) / stds

    print("z mean:", z.mean().item())
    print("z std:", z.std().item())

    return means, stds, targets