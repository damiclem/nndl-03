# Dependencies
from src.network.charlie import multinomial_fn
from src.network.charlie import Charlie
from torch import nn
import torch
import re


class Worden(Charlie):

    # Constructor
    def __init__(self, vocab_size, embedding_dim, hidden_units, layers_num, hidden_type='LSTM', trained_embeddings=None, freeze_embeddings=False, dropout_prob=0.3):
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
        if trained_embeddings is not None:
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

        # Define new output layer
        self.out = nn.Linear(hidden_units, vocab_size)

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

    def to_file(self, path, optimizer=None):
        # Save state dictionary
        torch.save({
            # Save optimizer state
            'optim': optimizer.state_dict() if optimizer else None,
            # Model state
            'model': {
                # Store model architecture
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'hidden_units': self.hidden_units,
                'layers_num': self.layers_num,
                'hidden_type': self.hidden_type,
                'dropout_prob': self.dropout_prob,
                # Store model weights
                'weights': self.state_dict()
            },
        }, path)

    def train_batch(self, batch, loss_fn, optimizer=None, eval=False):
        """ Train batch of input data

        Args
        batch (torch.Tensor)        Float tensor representing input data
        loss_fn (nn.Module)         Loss function, used to compute train loss
        optimizer (nn.optimizer)    Optimizer used to find out best weights
        eval (bool)                 Whether to perform training or evaluation

        Return
        (float)                     Mean loss
        """
        # Case we are in training mode and optimizer is not set
        if optimizer is None and not eval:
            # Raise exception
            raise ValueError('training requires optimizer, none set')

        # Take out target variable (last character) from characters window
        target = batch[:, -1]
        # Remove the target variable from the input tensor
        input = batch[:, :-1]

        # Training setting
        if not eval:
            # Clear previous recorded gradients
            optimizer.zero_grad()
        # Make forward pass
        output, _ = self(input)

        # Evaluate loss only for last output
        loss = loss_fn(output[:, -1, :], target)
        # Training setting
        if not eval:
            # Backward pass
            loss.backward()
            # Update
            optimizer.step()
        # Return average batch loss
        return float(loss.data)

    def test_batch(self, batch, loss_fn):
        return self.train_batch(batch, loss_fn, eval=True)

    def generate_text(self, seed, num_words, encode_fn, decode_fn, decision_how='argmax', crop_len=100):
        """ Given a seed, generate some text

        Args
        seed (str)                  Seed string, fed to encoding function
        num_words (int)             Number of words to be predicted
        encode_fn (function)        Function for encoding input string to
                                    torch Tensor
        decode_fn (function)        Function for decoding one hot encoded
                                    vector to string
        decision_how (str)          Whether to use argmax or multinomial to
                                    select one hot encoded vector
        crop_len (int)              Minimum of elements in a sentence

        Return
        (list)                      Automatically generated list of words
        """
        # Get network device
        device = self.get_device()

        # Define an empty input tensor
        zeros_in = torch.zeros(1, crop_len).long()

        # Encode seed to feed it into network
        # Must encode to [1, words]
        net_in = encode_fn(seed).long()

        # Define number of encoded characters
        n = min(crop_len, net_in.shape[1])
        # Pad network input using zeors tenosr
        zeros_in[[0], -n:] = net_in[[0], -n:]
        # Update network input
        net_in = zeros_in.to(device)

        # Initialize decision function
        decision_fn = None
        # Case argmax has been selected
        if decision_how == 'argmax':
            # Set argmax decision function
            decision_fn = lambda x: x[:, -1].argmax().item()
        # Case softmax has been selected
        if decision_how == 'softmax':
            # Set softmax decision function
            decision_fn = lambda x: multinomial_fn(x[:, -1]).item()
        # Case no valid function has been selected
        if decision_fn is None:
            # Raise exception
            raise ValueError('given decision function is not valid')

        # Define new seed as list of words
        seed_ = []
        # Disable gradient computation
        with torch.no_grad():
            # Initialize network output and state
            net_out, net_state = self(net_in)
            # Apply decision function on output
            next_index = decision_fn(net_out)
            # Decode to words
            next_word = decode_fn(next_index)
            # Update network input
            net_in = torch.tensor([[next_index]],
                                  dtype=torch.long,
                                  device=device)
            # Store word
            seed_ += [next_word]

            # Loop through each character to predict
            for i in range(num_words - 1):
                # Predicting next character
                net_out, net_state = self(net_in, net_state)
                # Get the most probable last output index
                next_index = decision_fn(net_out)
                # Decode one hot encoded word to string
                next_word = decode_fn(next_index)
                # Update network input
                net_in = torch.tensor([[next_index]],
                                      dtype=torch.long,
                                      device=device)
                # Store word
                seed_ += [next_word]

        # Loop through each retrieved word
        for i, word in enumerate(seed_):
            # Case previous word is newline
            if re.search(r'\n$', word, re.DOTALL):
                # Do not add space
                seed += word
            # Case new word is punctuation
            if re.search(r'^[\"\,\.\!\?\n\']', word, re.DOTALL):
                # Do not add space
                seed += word
            # Othrewise
            else:
                # Add both the word and preceding space
                seed += ' ' + word
        # Return generated seed
        return seed
