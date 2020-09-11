# from time import time
from tqdm import tqdm
# from torch import nn
import numpy as np
import itertools
import torch
import json
import re
import os


# Utility: make grid search
def grid_search(train_dl, test_dl, num_epochs, sep=r'__', verbose=True, device=torch.device('cpu'), **kwargs):
    """ Make grid search

    Args
    train_dl (DataLoader)       Train dataloader
    test_dl (DataLoader)        Test dataloader
    num_epochs (int)            Number of training epochs
    sep (str)                   Characters used to split instance and parameter
    verbose (bool)              Whether to print verbose output or not
    device (torch.device)       Device to use in training and test

    Return
    (list)                      List of train losses per parameter combination
    (list)                      List of train time per parameter combination
    (list)                      List of test losses per parameter combination
    (list)                      List of test time per parameter combination
    (list)                      List of parameters combinations
    """
    # Make cartesian product of all the keys: list of arguments combinations
    kwargs = list(dict(zip(kwargs.keys(), x)) for x in itertools.product(*kwargs.values()))
    # Initialize train losses and time list
    train_losses, train_times = list(), list()
    # Initialize test losses and time list
    test_losses, test_times = list(), list()

    # Define iterator
    iterator = tqdm(range(len(kwargs)), disable=(not verbose),
                    desc='Making grid search')
    # Loop through each argument combination
    for i in iterator:

        # Retrieve network constructor
        net_init = kwargs[i]['net']
        # Retrieve  optimizer constructor
        optim_init = kwargs[i]['optim']
        # Retrieve loss function constructo
        loss_fn_init = kwargs[i]['loss_fn']

        # Define network arguments prefix
        net_prefix = r'^net__'
        # Get network keyword arguments
        net_kwargs = {
            # Remove starting `net__` prefix
            re.sub(net_prefix, '', kw): kwargs[i][kw]
            # Loop through each keyword argument
            for kw in kwargs[i]
            # Match only arguments starting with `net__` prefix
            if re.search(net_prefix, kw)
        }

        # Define optimizer arguments prefix
        optim_prefix = r'^optim__'
        # Get optimizer keyword arguments
        optim_kwargs = {
            # Remove starting `optim__` prefix
            re.sub(optim_prefix, '', kw): kwargs[i][kw]
            # Loop through each keyword argument
            for kw in kwargs[i]
            # Match only arguments starting with `optim__` prefix
            if re.search(optim_prefix, kw)
        }

        # Define loss function arguments prefix
        loss_fn_prefix = r'^loss_fn__'
        # Get loss_function keyword arguments
        loss_fn_kwargs = {
            # Remove starting `loss_fn__` prefix
            re.sub(loss_fn_prefix, '', kw): kwargs[i][kw]
            # Loop through each keyword argument
            for kw in kwargs[i]
            # Match only arguments startung with `loss_fn__` prefix
            if re.search(loss_fn_prefix, kw)
        }

        # Define current network
        net = net_init(**net_kwargs).to(device)
        # Define optimizer
        optim = optim_init(net.parameters(), **optim_kwargs)
        # Define loss function
        loss_fn = loss_fn_init(**loss_fn_kwargs)

        # Set newtork in training mode
        net.train()
        # Initialize current train loss and time
        train_losses_, train_times_ = [], []
        # Start training
        for e in range(num_epochs):
            # Train the network, retrieve mean loss and total time for current epoch
            epoch_loss, epoch_time = net.train_epoch(
                train_dl=train_dl,
                loss_fn=loss_fn,
                optimizer=optim,
                device=device
            )
            # Store epoch loss: scale it according to number of epochs
            train_losses_ += [epoch_loss]
            # Store epoch time: sum it with other epochs
            train_times_ += [epoch_time]
        # Store current parameters train loss and time
        train_losses.append(train_losses_)
        train_times.append(train_times_)

        # Set network in evaluation mode
        net.eval()
        # Disabe gradient evaluation for test purposes
        with torch.no_grad():
            # Test the network, retrieve mean loss and total time for current epoch
            test_loss, test_time = net.test_epoch(
                test_dl=test_dl,
                loss_fn=loss_fn,
                device=device
            )
            # Store test loss
            test_losses.append(test_loss)
            # Store test time
            test_times.append(test_time)

        # Verbose log
        if verbose:
            # Initialize output message
            msg = '\nParameters ({:d}): ({:s})\n\n'.format(i + 1, ', '.join([
                # Print current parameters
                '{:s}: {:s}'.format(str(kw), str(kwargs[i][kw]))
                # loop thriugh each parameter in current combination
                for kw in kwargs[i]
            ]))
            msg += 'Train loss {:.3f} '.format(train_losses_[-1])
            msg += 'in {:.3f} seconds (last)\n'.format(train_times_[-1])
            msg += 'Test loss {:.3f} '.format(test_loss)
            msg += 'in {:.3f} seconds\n'.format(test_time)
            # Write out message
            iterator.write(msg)

    # Return eiter parameters, losses and times lists
    return train_losses, train_times, test_losses, test_times, kwargs


# Utility: save epochs to disk
def save_epochs(path, train_losses=list(), train_times=list(), test_losses=list(), test_times=list(), **kwargs):
    """ Save epochs as .json

    Args
    path (str)              Path to output file
    # params (list)           List of parameters combinations
    train_losses (list)     List of mean loss per train epoch
    train_times (list)      List of time taken per train epoch
    test_losses (list)      List of mean loss per test epoch
    test_times (list)       List of time taken per test epoch

    Raise
    (OSError)               In case saving file was not possible
    """
    # Open output file
    with open(path, 'w') as file:
        # Write json file
        json.dump({
            # # Save parameters name and value as strings
            # 'params': [
            #     {str(kw): str(params[i][kw]) for kw in params[i]}
            #     for i in range(len(params))
            # ],
            # Other parameters to save
            **kwargs,
            # Save lists (eventually override less important parameters)
            'train_losses': train_losses,
            'train_times': train_times,
            'test_losses': test_losses,
            'test_times': test_times
        }, file)


# Utility: load epochs from disk
def load_epochs(path):
    """ Load epochs from .json file

    Args
    path (str)              Path to file holding epochs values

    Return
    (list)                  List of mean loss per train epoch
    (list)                  List of time taken per train epoch
    (list)                  List of mean loss per test epoch
    (list)                  List of time taken per test epoch
    (dict)                  Dictionary mapping other column names to lists

    Raise
    (FileNotFoundError)         In case file does not exists
    (OSError)                   In case it was not possible opening file
    """
    # # Initialize parameters list
    # params = list()
    # Initialize lists of training losses and times
    train_losses, train_times = list(), list()
    # Initialize lists of test losses and times
    test_losses, test_times = list(), list()
    # Open input file
    with open(path, 'r') as file:
        # Load json file
        epochs_dict = json.load(file)
        # # Retrieve lists of values
        # params = epochs_dict.get('params', list())
        train_losses = epochs_dict.pop('train_losses', list())
        train_times = epochs_dict.pop('train_times', list())
        test_losses = epochs_dict.pop('test_losses', list())
        test_times = epochs_dict.pop('test_times', list())
    # Return retrieved values
    return train_losses, train_times, test_losses, test_times, epochs_dict


# Utility: make training and testing
def train_test_epochs(
    net, train_dl, test_dl, loss_fn, optimizer, num_epochs, save_after,
    seed, encode_fn, decode_fn, decision_how='argmax', crop_len=100,
    net_path='', epochs_path='', verbose=True, device=torch.device('cpu')
):
    """ Load, train and test a network

    Args
    net (nn.Module)             Network to be trained and tested
    train_dl (DataLoader)       Train examples iterator
    test_dl (DataLoader)        Test exapmples iterator
    loss_fn (?)                 Function used to compute loss
    optimizer (toch.optim)      Optimizer updating network weight
    num_epochs (int)            Number of epochs to loop through
    save_after (int)            Number of epochs after which a checkupoint
                                must be saved
    seed (str)                  Test seed string, used in sample generation
    encode_fn (function)        Function encoding words to indices
    decode_fn (function)        Function decoding indices to words
    decision_how (str)          Wether to use argmax or softmax in sample
                                text generation
    crop_len (int)              Number of previous items used in prediction
    net_path (str)              Path to network weights file (.pth)
    epochs_path (str)           Path to epochs values file (.tsv)
    verbose (bool)              Wether to show verbose output or not
    device (torch.device)       Device holding network weights

    Return
    (list)      Train losses (mean per epoch)
    (list)      Test losses (mean per epoch)
    (list)      Train time (per epoch)
    (list)      Test time (per epoch)
    """
    # If file path is set, load network from file
    if os.path.isfile(net_path):
        # Load pretrained weigths and optimizer state
        net, _ = net.from_file(path=net_path, optimizer=optimizer)
    # Move network to device
    net.to(device)

    # Initialize train and test loss (per epoch)
    train_losses, test_losses = list(), list()
    # Initialize train and test times (per epoch)
    train_times, test_times = list(), list()

    # Eventually load epochs from file
    if os.path.isfile(epochs_path):
        # Load epochs values
        _, train_losses, train_times, test_losses, test_times = load_epochs(
            path=epochs_path
        )

    # Initialize best evaluation loss
    best_loss = float('inf')
    # Define iterator
    iterator = tqdm(range(1, num_epochs + 1), disable=(not verbose), desc='Training')
    # Loop through each epoch
    for e in iterator:
        # Set network in training mode
        net.train()
        # Train the network, retrieve mean loss and total time for current epoch
        train_loss, train_time = net.train_epoch(
            train_dl=train_dl,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        # Store train loss
        train_losses.append(train_loss)
        # Store train time
        train_times.append(train_time)

        # Set network in evaluation mode
        net.eval()
        # Disable gradient computation
        with torch.no_grad():
            # Test the network, retrieve mean loss and total time for current epoch
            test_loss, test_time = net.test_epoch(
                test_dl=test_dl,
                loss_fn=loss_fn,
                device=device
            )
            # Store test loss
            test_losses.append(test_loss)
            # Store test time
            test_times.append(test_time)

        # Case test loss is lower than previous best
        if (test_loss < best_loss) and (test_loss < train_loss):
            # Update new best loss
            best_loss = test_loss
            # Save best model
            net.to_file(path=net_path, optimizer=optimizer)

        # Check if current epoch is a checkpoint
        if e % save_after == 0:
            # Verbose output
            if verbose:
                # Initialize output message
                msg = 'Epoch nr. {:d}:\n'.format(e)
                msg += 'Train loss (mean) {:.3f} '.format(train_loss)
                msg += 'in {:.3f} seconds\n'.format(train_time)
                msg += 'Test loss (mean) {:.3f} '.format(test_loss)
                msg += 'in {:.3f} seconds\n\n'.format(test_time)
                # Add sampled text
                msg += 'Sampled text (with seed "{:s}"):\n'.format(seed)
                msg += net.generate_text(
                    seed, 250, encode_fn=encode_fn, decode_fn=decode_fn,
                    decision_how=decision_how, crop_len=crop_len
                )
                # Write out message
                # iterator.write(msg)
                print(msg)

            # # Save model weights
            # net.to_file(path=net_path, optimizer=optimizer)
            # Save epochs train and test values
            save_epochs(
                train_losses=train_losses,
                train_times=train_times,
                test_losses=test_losses,
                test_times=test_times,
                path=epochs_path
            )
