# Dependencies
from time import time
from torch import nn
import torch


# Define softmax function
softmax_fn = nn.Softmax(dim=1)


# Define multinomial function
multinomial_fn = lambda x: torch.multinomial(softmax_fn(x), 1)


class Charlie(nn.Module):

    # Constructor
    def __init__(self, input_size, hidden_units, layers_num, hidden_type='LSTM', dropout_prob=0.3):
        """ Constructor

        Args
        input_size (int)        Number of features in a single input vector
        hidden_units (int)      Number of hidden units in a single recurrent
                                hidden layer
        layers_num (int)        Number of stacked hidden layers
        hidden_type (str)       Type of hidden layer: LSTM or GRU
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

    # Retrieve device
    def get_device(self):
        return next(self.parameters()).device

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

    # Save to file
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
                'layers_num': self.layers_num,
                'hidden_type': self.hidden_type,
                'dropout_prob': self.dropout_prob,
                # Store model weights
                'weights': self.state_dict()
            },
        }, path)

    @classmethod
    def from_file(cls, path, optimizer=None):
        # Load state dictionary
        state_dict = torch.load(path, map_location='cpu')

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

    def train_batch(self, batch, loss_fn, optimizer=None, eval=False):
        """ Train batch of input data

        Args
        batch (torch.Tensor)        Float tensor representing input data
        loss_fn (nn.Module)         Loss function, used to compute train loss
        optimizer (nn.optimizer)    Optimizer used to find out best weights
        eval (bool)                 Whether to perform training or evaluation

        Return
        (float)                     Mean loss for given batch
        """
        # Case we are in training mode and optimizer is not set
        if optimizer is None and not eval:
            # Raise exception
            raise ValueError('training requires optimizer, none set')

        # Take out target variable (last character) from characters window
        labels_ohe = batch[:, -1, :]
        # Retrieve label as integer (the only position whose value is 1)
        labels_int = labels_ohe.argmax(dim=1)

        # Remove the labels from the input tensor
        input_ohe = batch[:, :-1, :]

        # Training setting
        if not eval:
            # Clear previous recorded gradients
            optimizer.zero_grad()
        # Make forward pass
        output_ohe, _ = self(input_ohe)

        # Evaluate loss only for last output
        loss = loss_fn(output_ohe[:, -1, :], labels_int)
        # Training setting
        if not eval:
            # Backward pass
            loss.backward()
            # Update
            optimizer.step()
        # Return average batch loss
        return float(loss.data)

    def test_batch(self, batch, loss_fn):
        # # Take out target variable (last character) from characters window
        # labels_ohe = batch[:, -1, :]
        # # Retrieve label as integer (the only position whose value is 1)
        # labels_int = labels_ohe.argmax(dim=1)
        #
        # # Remove the labels from the input tensor
        # input_ohe = batch[:, :-1, :]
        # # Make forward pass
        # output_ohe, _ = self(input_ohe)
        #
        # # Evaluate loss only for last output
        # loss = loss_fn(output_ohe[:, -1, :], labels_int)
        # # Return average batch loss
        # return float(loss.data)
        return self.train_batch(batch, loss_fn, eval=True)

    def train_epoch(self, train_dl, loss_fn, optimizer=None, eval=False, device=torch.device('cpu')):
        """ Train a single epoch

        Args
        train_dl (DataLoader)   Test batches iterator
        loss_fn (?)             Function used to compute loss
        optimizer (?)           Optimizer updating network weights
        eval (bool)             Whether to perform training or evaluation
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
                optimizer=optimizer,
                eval=eval
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
        device (torch.device)   Device holding network weights

        Return
        (float)                 Mean loss
        (float)                 Time taken (seconds)
        """
        # # Initialize timers
        # time_beg, time_end = time(), 0.0
        # # Initialize epoch loss
        # epoch_losses = list()
        # # Loop through each batch in current epoch
        # for batch in test_dl:
        #     # Update network
        #     batch_loss = self.test_batch(
        #         batch=batch.to(device),
        #         loss_fn=loss_fn
        #     )
        #     # Store batch loss
        #     epoch_losses.append(batch_loss)
        # # Update timers
        # time_end = time()
        # # Return mean loss and total time
        # return sum(epoch_losses) / len(epoch_losses), time_end - time_beg
        return self.train_epoch(test_dl, loss_fn, eval=True, device=device)

    def generate_text(self, seed, num_chars, encode_fn, decode_fn, decision_how='argmax', crop_len=100):
        """ Given a seed, generate some text

        Args
        seed (str)                  Seed string, fed to encoding function
        num_chars (int)             Number of characters to be predicted
        encode_fn (function)        Function for encoding input string to
                                    torch Tensor
        decode_fn (function)        Function for decoding one hot encoded
                                    vector to string
        decision_how (str)          Whether to use argmax or multinomial to
                                    select one hot encoded vector
        crop_len (int)              Minimum of elements in a sentence

        Return
        (str)                       Automatically generated text
        """
        # Get network device
        device = self.get_device()

        # Define an empty input tensor
        zeros_in = torch.zeros(1, crop_len, self.input_size)

        # Encode seed to feed it into network
        # Must encode to [1, characters, features]
        net_in = encode_fn(seed)

        # Define number of encoded characters
        n = min(crop_len, net_in.shape[1])
        # Pad network input using zeors tenosr
        zeros_in[[0], -n:, :] = net_in[[0], -n:, :]
        # Update network input
        net_in = zeros_in.to(device)

        # # DEBUG
        # print('Seed shape:', net_in.shape)

        # Initialize decision function
        decision_fn = None
        # Case argmax has been selected
        if decision_how == 'argmax':
            # Set argmax decision function
            decision_fn = lambda x: x[:, -1, :].argmax().item()
        # Case softmax has been selected
        if decision_how == 'softmax':
            # Set softmax decision function
            decision_fn = lambda x: multinomial_fn(x[:, -1, :]).item()
        # Case no valid function has been selected
        if decision_fn is None:
            # Raise exception
            raise ValueError('given decision function is not valid')

        # Disable gradient computation
        with torch.no_grad():
            # Initialize network output and state
            net_out, net_state = self(net_in)
            # Apply decision function on output
            next_index = decision_fn(net_out)
            # Decode to character
            next_char = decode_fn(next_index)
            # Update network input
            net_in = encode_fn(next_char).to(device)
            # Update seed, add decoded character
            seed += next_char

            # Loop through each character to predict
            for i in range(num_chars - 1):
                # Predicting next character
                net_out, net_state = self(net_in, net_state)
                # Get the most probable last output index
                next_index = decision_fn(net_out)
                # Decode one hot encoded character to string
                next_char = decode_fn(next_index)
                # Update network input
                net_in = encode_fn(next_char).to(device)
                # Update seed
                seed += next_char

        # Return generated seed
        return seed
