from time import time
from torch import nn
import numpy as np
import itertools
import torch
import json
import re
import os


class Charlie(nn.Module):

    # Constructor
    def __init__(self, input_size, hidden_units, layers_num, hidden_type='LSTM', dropout_prob=0.3):
        """ Constructor

        Args
        input_size (int)        Number of features in a single input vector
        hidden_units (int)      Number of hidden units in a single recurrent
                                hidden layer
        layers_num (int)        Number of stacked hidden layers
        hidden_type (str)       Type of hidden layer: LST or GRU
        dropout_prob (float)    Probability for dropout, must be between [0, 1]

        Raise
        (ValueError)            In case hidden type is not LSTM or GRU
        """
        # Call the parent Constructor
        super().__init__()

        # Initialize hidden layer class
        rnn = self.get_recurrent(hidden_type)

        # Define recurrent layer
        self.rnn = rnn(
            # Define size of the one-hot-encoded input
            input_size=input_size,
            # Define size of a single recurrent hidden layer
            hidden_size=hidden_units,
            # Define number of stacked recurrent hidden layers
            num_layers=layers_num,
            # Set dropout probability
            dropout=dropout_prob,
            # Set batch size as first dimension
            batch_first=True
        )

        # Define output layer
        self.out = nn.Linear(hidden_units, input_size)

        # Save architecture
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.layers_num = layers_num
        self.hidden_type = hidden_type
        self.dropout_prob = dropout_prob

    # Retrieve hidden layer class
    def get_recurrent(self, hidden_type):
        """ Retrieve hidden layer class

        Args
        hidden_type (str)       Name of hidden layer class

        Return
        (nn.Module)             Recurrent layer class

        Raise
        (ValueError)            In case hidden type is not valid
        """
        # Case hidden type is LSTM
        if hidden_type == 'LSTM':
            return nn.LSTM

        # Case hidden type is GRU
        if hidden_type == 'GRU':
            return nn.GRU

        # Otherwise: raise error
        raise ValueError(' '.join([
            'Error: given recurrent neural network type can be',
            'either LSTM or GRU: {} given instead'.format(hidden_type)
        ]))

    # Make forward step
    def forward(self, x, state=None):
        """ Make forward step

        Args
        x (torch.Tensor)        Tensor containing input vectors with size
                                (batch size, window size, number of features)

        state (?)               TODO

        Return
        (torch.Tensor)          Predicted new values (shifted input vector)
        (?)                     TODO
        """
        # Recurrent (hidden) layer output
        x, state = self.rnn(x, state)
        # Decision (linear) layer output
        x = self.out(x)
        # Return both new x and state
        return x, state

    def to_file(self, path, optimizer=None):
        # Save state dictionary
        torch.save({
            # Save optimizer state
            'optim': optimizer.state_dict() if optimizer else None,
            # Model state
            'model': {
                # Store model architecture
                'input_size': self.input_size,
                'hidden_units': self.hidden_units,
                'layers_num': self.lcayers_num,
                'hidden_type': self.hidden_type,
                'dropout_prob': self.dropout_prob,
                # Store model weights
                'weights': self.state_dict()
            },
        }, path)

    @classmethod
    def from_file(cls, path, optimizer=None):
        # Load state dictionary
        state_dict = torch.load(path)

        # Case optimizer has been given
        if optimizer:
            # Load optimizer state
            optimizer.load_state_dict(state_dict.get('optim'))

        # Load model dictionary
        model_dict = state_dict.get('model')
        # Retrieve model weigths
        model_weights = model_dict.pop('weights')
        # Instantiate new Neural Network class
        network = cls(**model_dict)
        # Load model parameters
        network.load_state_dict(model_weights)

        # Return the network and other loaded parameters
        return network, optimizer

    def train_batch(self, batch, loss_fn, optimizer):
        """ Train batch of input data

        Args
        batch (torch.Tensor)        Float tensor representing input data
        loss_fn (nn.Module)         Loss function, used to compute train loss
        optimizer (nn.optimizer)    Optimizer used to find out best weights
        """
        # Take out target variable (last character) from characters window
        labels_ohe = batch[:, -1, :]
        # Retrieve label as integer (the only position whose value is 1)
        labels_int = labels_ohe.argmax(dim=1)

        # Remove the labels from the input tensor
        input_ohe = batch[:, :-1, :]

        # Eventually clear previous recorded gradients
        optimizer.zero_grad()
        # Make forward pass
        output_ohe, _ = self(input_ohe)

        # Evaluate loss only for last output
        loss = loss_fn(output_ohe[:, -1, :], labels_int)
        # Backward pass
        loss.backward()
        # Update
        optimizer.step()
        # Return average batch loss
        return float(loss.data)

    def test_batch(self, batch, loss_fn):
        # Take out target variable (last character) from characters window
        labels_ohe = batch[:, -1, :]
        # Retrieve label as integer (the only position whose value is 1)
        labels_int = labels_ohe.argmax(dim=1)

        # Remove the labels from the input tensor
        input_ohe = batch[:, :-1, :]
        # Make forward pass
        output_ohe, _ = self(input_ohe)

        # Evaluate loss only for last output
        loss = loss_fn(output_ohe[:, -1, :], labels_int)
        # Return average batch loss
        return float(loss.data)

    def train_epoch(self, train_dl, loss_fn, optimizer, device=torch.device('cpu')):
        """ Train a single epoch

        Args
        train_dl (DataLoader)   Test batches iterator
        loss_fn (?)             Function used to compute loss
        optimizer (?)           Optimizer updating network weights
        device (troch.device)   Device holding network weights

        Return
        (float)     Mean loss
        (float)     Time taken
        """
        # Initialize timers
        time_beg, time_end = time(), 0.0
        # Initialize epoch loss
        epoch_losses = list()
        # Loop through each batch in current epoch
        for batch in train_dl:
            # Update network
            batch_loss = self.train_batch(
                batch=batch.to(device),
                loss_fn=loss_fn,
                optimizer=optimizer
            )
            # Store batch loss
            epoch_losses.append(batch_loss)
        # Update timers
        time_end = time()
        # Return mean loss and total time
        return sum(epoch_losses) / len(epoch_losses), time_end - time_beg

    def test_epoch(self, test_dl, loss_fn, device=torch.device('cpu')):
        """ Test a single epoch

        Args
        test_dl (DataLoader)    Test batches iterator
        loss_fn (?)             Function used to compute loss
        device (troch.device)   Device holding network weights

        Return
        (float)                 Mean loss
        (float)                 Time taken (seconds)
        """
        # Initialize timers
        time_beg, time_end = time(), 0.0
        # Initialize epoch loss
        epoch_losses = list()
        # Loop through each batch in current epoch
        for batch in test_dl:
            # Update network
            batch_loss = self.test_batch(
                batch=batch.to(device),
                loss_fn=loss_fn
            )
            # Store batch loss
            epoch_losses.append(batch_loss)
        # Update timers
        time_end = time()
        # Return mean loss and total time
        return sum(epoch_losses) / len(epoch_losses), time_end - time_beg


class Worden(Charlie):

    # Constructor
    def __init__(
        self, vocab_size, embedding_dim, hidden_units, layers_num,
        hidden_type='LSTM', trained_embeddings=None, freeze_embeddings=False,
        dropout_prob=0.3
    ):
        """ Constructor

        Args
        vocab_size (int)                    Number of words in vocabulary
        embedding_dim (int)                 Number of attributes in each
                                            word vector
        input_size (int)                    Number of features in a single
                                            input vector
        hidden_units (int)                  Number of hidden units in a single
                                            recurrent hidden layer
        layers_num (int)                    Number of stacked hidden layers
        hidden_type (str)                   Type of hidden layer: LST or GRU
        trained_embeddings (torch.Float)    Pre-trained embeddings tensor
        freeze_embeddings (bool)            Wether embeddings have to be
                                            trained or not
        dropout_prob (float)                Probability for dropout, must be
                                            between [0, 1]

        Raise
        (ValueError)                        In case hidden type is not LSTM
                                            or GRU
        (ValueError)                        In case given embedding shape and
                                            pretrained embeddings does not
                                            coincide
        """
        # Call parent constructor
        super().__init__(
            input_size=embedding_dim,
            hidden_units=hidden_units,
            layers_num=layers_num,
            hidden_type=hidden_type,
            dropout_prob=dropout_prob
        )

        # Case pretrained embedding layer has been defined
        if trained_embeddings:
            # Load embedding layer from pretrained
            self.embed = nn.Embedding.from_pretrained(
                embeddings=trained_embeddings,
                freeze=freeze_embeddings
            )

            # Check that given pre-trained embeddings and given sizes match
            if (vocab_size, embedding_dim) != trained_embeddings.shape:
                # Raise new exception
                raise ValueError(' '.join([
                    'Error: given pre-trained embeddings',
                    'have shape {}'.format(trained_embeddings.shape),
                    'while the expected shape',
                    'is {}'.format((vocab_size, embedding_dim))
                ]))

        # Otherwise, initialize new embedding layer
        else:
            # Define embedding layer
            self.embed = nn.Embedding(vocab_size, embedding_dim)

        # Store vocabulary size
        self.vocab_size = vocab_size
        # Store embedding dimension
        self.embedding_dim = embedding_dim

    def forward(self, x, state=None):
        """ Make forward step

        Args
        x (torch.Tensor)        Tensor containing input vectors with size
                                (batch size, window size, number of features)

        state (?)               TODO

        Return
        (torch.Tensor)          Predicted new values (shifted input vector)
        (?)                     TODO
        """
        # Go through embedding layer first
        x = self.embed(x)
        # Recurrent (hidden) layer output
        x, state = self.rnn(x, state)
        # Decision (linear) layer output
        x = self.out(x)
        # Return both new x and state
        return x, state

    def train_batch(self, batch, loss_fn, optimizer):
        """ Train batch of input data

        Args
        batch (torch.Tensor)        Float tensor representing input data
        loss_fn (nn.Module)         Loss function, used to compute train loss
        optimizer (nn.optimizer)    Optimizer used to find out best weights

        Return
        (float)                     Mean loss
        """
        # Take out target variable (last character) from characters window
        target = batch[:, -1]
        # Remove the target variable from the input tensor
        input = batch[:, :-1]

        # Eventually clear previous recorded gradients
        optimizer.zero_grad()
        # Make forward pass
        output, _ = self(input)

        # Evaluate loss only for last output
        loss = loss_fn(output, target)
        # Backward pass
        loss.backward()
        # Update
        optimizer.step()
        # Return average batch loss
        return float(loss.data)

    def test_batch(self, batch, loss_fn):
        """ Test batch of input data

        Args
        batch (torch.Tensor)        Float tensor representing input data
        loss_fn (nn.Module)         Loss function, used to compute train loss

        Return
        (float)                     Mean loss
        """
        # Take out target variable (last character) from characters window
        target = batch[:, -1]
        # Remove the target variable from the input tensor
        input = batch[:, :-1]
        # Make forward pass
        output, _ = self(input)
        # Evaluate loss only for last output
        loss = loss_fn(output, target)
        # Return average batch loss
        return float(loss.data)


# Utility: make grid search
def grid_search(
    train_dl, test_dl, num_epochs, verbose=True,
    device=torch.device('cpu'), **kwargs
):
    """ Make grid search

    Args
    train_dl (DataLoader)    Train dataloader
    test_dl (DataLoader)     Test dataloader
    num_epochs (int)         Number of training epochs
    verbose (bool)           Whether to print verbose output or not
    device (torch.device)    Device to use in training and test

    Return
    (list)                   List of parameters combinations
    (list)                   List of train losses per parameter combination
    (list)                   List of train time per parameter combination
    (list)                   List of test losses per parameter combination
    (list)                   List of test time per parameter combination
    """
    # Make cartesian product of all the keys: list of arguments combinations
    kwargs = list(dict(zip(kwargs.keys(), x)) for x in itertools.product(*kwargs.values()))
    # Initialize train losses and time list
    train_losses, train_times = list(), list()
    # Initialize test losses and time list
    test_losses, test_times = list(), list()

    # Loop through each argument combination
    for i in range(len(kwargs)):
        # Get network keyword arguments
        net_kwargs = {
            kw.split('__')[1]: arg
            for kw, arg in kwargs[i].items()
            if re.search('^net__', kw)
        }
        # Retrieve network from kwargs
        network = kwargs[i].get('net')
        # Make network and put it on selected device
        network = network(**net_kwargs).to(device)

        # Get optimizer keyword arguments
        optimizer = optimizer_kwargs = {
            kw.split('__')[1]: arg
            for kw, arg in kwargs[i].items()
            if re.search('^optim__', kw)
        }
        # Get optimizer (class only)
        optimizer = kwargs[i].get('optim')
        # Make new optimizer instance
        optimizer = optimizer(network.parameters(), **optimizer_kwargs)

        # Get loss_function keyword arguments
        loss_fn_kwargs = {
            kw.split('__')[1]: arg
            for kw, arg in kwargs[i].items()
            if re.search('^loss_fn__', kw)
        }
        # Get loss function (class only)
        loss_fn = kwargs[i].get('loss_fn')
        # Make new loss function instance
        loss_fn = loss_fn(**loss_fn_kwargs)

        # Set newtork in training mode
        network.train()
        # Initialize train loss
        train_loss, train_time = 0.0, 0.0
        # Start training
        for e in range(num_epochs):
            # Train the network, retrieve mean loss and total time for current epoch
            epoch_loss, epoch_time = network.train_epoch(
                train_dl=train_dl,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device
            )
            # Store epoch loss: scale it according to number of epochs
            train_loss = epoch_loss / num_epochs
            # Store epoch time: sum it with other epochs
            train_time = train_time + epoch_time
        # Store current parameters train loss and time
        train_losses.append(train_loss)
        train_times.append(train_time)
        # # Store mean train loss
        # train_losses.append(sum(batch_losses) / len(batch_losses))
        # # Store train time
        # train_times.append(time() - train_start)
        # # Show last training loss
        # print('done in {:.0f} seconds'.format(train_times[-1]), end=' ')
        # print('with mean loss {:.03f}'.format(train_losses[-1]))

        # Print training step summary
        if verbose:
            print('Current network parameters ({:d}):'.format(i + 1))
            print('(' + ', '.join(['{:}: {:}'.format(kw, arg) for kw, arg in kwargs[i].items()]) + ')')
            print('Training done with loss {:.03f}'.format(train_loss), end=' ')
            print('in {:.0f} seconds'.format(train_time))

        # # Define test start time
        # test_start = time()
        # # Initialize list of batch losses
        # batch_losses = list()
        # Set network in evaluation mode
        network.eval()
        # Disabe gradient evaluation for test purposes
        with torch.no_grad():
            # Test the network, retrieve mean loss and total time for current epoch
            test_loss, test_time = network.test_epoch(
                test_dl=test_dl,
                loss_fn=loss_fn,
                device=device
            )
            # Store test loss
            test_losses.append(test_loss)
            # Store test time
            test_times.append(test_time)
            # # Print start of test iterations
            # print('Testing...', end=' ', flush=False)
            # # Iterate batches
            # for batch in test:
            #     # Make forward step
            #     batch_loss = network.test_batch(
            #         net=network,
            #         batch=batch.to(device),
            #         loss_fn=loss_fn
            #     )
            #     # Store batch loss
            #     batch_losses.append(batch_loss)
            # # Save losses
            # test_losses.append(sum(batch_losses) / len(batch_losses))
            # # Store test time
            # test_times.append(time() - test_start)

            # Print test step summary
            if verbose:
                print('Test done with loss {:.03f}'.format(test_loss), end=' ')
                print('in {:.0f} seconds'.format(test_time))
                print()

    # Return eiter parameters, losses and times lists
    return kwargs, train_losses, train_times, test_losses, test_times


# Utility: save epochs to disk
def save_epochs(path, params=list(), train_losses=list(), train_times=list(), test_losses=list(), test_times=list()):
    """ Save epochs as .json

    Args
    params (list)           List of parameters combinations
    train_losses (list)     List of mean loss per train epoch
    train_times (list)      List of time taken per train epoch
    test_losses (list)      List of mean loss per test epoch
    test_times (list)       List of time taken per test epoch
    params_path (list)      Path to output file

    Raise
    (OSError)               In case saving file was not possible
    """
    # Open output file
    with open(path, 'w') as file:
        # Write json file
        json.dump({
            # Save parameters name and value as strings
            'params': [
                {str(kw): str(arg) for kw, arg in params[i]}
                for i in range(len(params))
            ],
            # Save lists
            'train_losses': train_losses,
            'train_times': train_times,
            'test_losses': test_losses,
            'test_times': test_times
        }, file)


# Utility: load epochs from disk
def load_params(path):
    """ Load epochs from .json file

    Args
    path (str)              Path to file holding epochs values

    Return
    (list)                  List of parameters combinations
    (list)                  List of mean loss per train epoch
    (list)                  List of time taken per train epoch
    (list)                  List of mean loss per test epoch
    (list)                  List of time taken per test epoch

    Raise
    (FileNotFoundError)         In case file does not exists
    (OSError)                   In case it was not possible opening file
    """
    # Initialize parameters list
    params = list()
    # Initialize lists of training losses and times
    train_losses, train_times = list(), list()
    # Initialize lists of test losses and times
    test_losses, test_times = list(), list()
    # Open input file
    with open(path, 'r') as file:
        # Load json file
        epochs_dict = json.load(file)
        # Retrieve lists of values
        params = epochs_dict.get('params', list())
        train_losses = epochs_dict.get('train_losses', list())
        train_times = epochs_dict.get('train_times', list())
        test_losses = epochs_dict.get('test_losses', list())
        test_times = epochs_dict.get('test_times', list())
    # Return retrieved values
    return params, train_losses, train_times, test_losses, test_times


# # Utility: save epochs train and test values
# def save_epochs(train_losses, train_times, test_losses, test_times, epochs_path):
#     """ Save epochs values as .tsv
#
#     Args
#     train_losses (list)     List of mean loss per train epoch
#     train_times (list)      List of time taken per train epoch
#     test_losses (list)      List of mean loss per test epoch
#     test_times (list)       List of time taken per test epoch
#     epochs_path (str)       Path to saved file
#
#     Raise
#     (ValueError)            In case input lists sizes does not match
#     (OSError)               In case it was not possible saving file
#     """
#     # Check lists sizes
#     if not (len(train_losses) == len(train_times) == len(test_losses) == len(test_times)):
#         # Raise error
#         raise ValueError('Error: given data sizes are not valid')
#
#     # Otherwise, save epochs as a .tsv file
#     with open(epochs_path, 'w') as file:
#         # Loop through each epoch
#         for e in range(len(train_losses)):
#             # Define line
#             line = '\t'.join([
#                 train_losses[e], train_times[e],
#                 test_losses[e], test_times[e]
#             ])
#             # Write line
#             file.write(line + '\n')
#
#
# # Utility: load epoch train and test values
# def load_epochs(epochs_path):
#     """ Load epochs train and test values
#
#     Args
#     epochs_path (str)       Path to file holding epochs data
#
#     Return
#     (list)                  List of mean loss per train epoch
#     (list)                  List of time taken per train epoch
#     (list)                  List of mean loss per test epoch
#     (list)                  List of time taken per test epoch
#
#     Raise
#     (FileNotFoundError)     In case epochs path does not exist
#     """
#     # Initialize train and test loss (per epoch)
#     train_losses, test_losses = list(), list()
#     # Initialize train and test times (per epoch)
#     train_times, test_times = list(), list()
#     # Open epochs file
#     with open(epochs_path, 'r') as file:
#         # Loop through each row (epoch) in file
#         for line in file:
#             # Split line into tuple
#             line = tuple(line.strip().split('\t'))
#             # Assign tuple values
#             train_loss, train_time, test_loss, test_time = line
#             # Store values
#             train_losses.append(train_loss)
#             train_times.append(train_time)
#             test_losses.append(test_loss)
#             test_times.append(test_time)
#     # Return loaded lists
#     return train_losses, train_times, test_losses, test_times


# Utility: make training and testing
def train_test_epochs(
    net, train_dl, test_dl, loss_fn, optimizer, num_epochs, save_after,
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
        net = net.from_file(path=net_path, optimizer=optimizer)
    # Move network to device
    net.to(device)

    # Initialize train and test loss (per epoch)
    train_losses, test_losses = list(), list()
    # Initialize train and test times (per epoch)
    train_times, test_times = list(), list()

    # Eventually load epochs from file
    if os.path.isfile(epochs_path):
        # Load epochs values
        _, train_losses, train_times, test_losses, test_times = load_params(
            epochs_path=epochs_path
        )

    # Loop through each epoch
    for e in num_epochs:
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

        # Check if current epoch is a checkpoint
        if (e + 1) % save_after == 0:
            # Verbose output
            if verbose:
                print('Epoch nr. {:d}'.format(e + 1))
                print('Train loss (mean) {:.3f}'.format(train_loss), end='')
                print('in {:.0f} seconds'.format(train_time))
                print('Test loss (mean) {:.3f}'.format(test_loss), end='')
                print('in {:.0f} seconds'.format(test_time))

            # Save model weights
            net.to_file(path=net_path, optimizer=optimizer)
            # Save epochs train and test values
            save_params(
                params=list(),
                train_losses=train_losses,
                train_times=train_times,
                test_losses=test_losses,
                test_times=test_times,
                epochs_path=epochs_path
            )
