# Dependencies
from torch import nn, optim
import torch
import math
import tqdm
import time
import os


class Tracy(nn.Module):

    def __init__(self, num_embeddings, embeddings_dim, num_heads, hidden_size, num_hidden, dropout_prob=0.3, device=torch.device('cpu')):
        """ Constructor

        Args
        num_embeddings (int)        Length of the embeddings vocabulary
        embeddings_dim (int)        Dimension of a single embedding vector
        num_heads (int)             Number of heads for multihead attention
        hidden_size (int)           Size of a single hidden layer
        num_hidden (int)            Number of transformer hidden layers
        dropout_prob (float)        Dropout probability for hidden layers
        device (torch.device)       Device where to move initialized network
        """
        # Call parent constructor (required)
        super().__init__()

        # Initialize positional encoder
        self.positional_encoder = PositionalEncoding(embeddings_dim, dropout_prob=dropout_prob)

        # Define single hidden layer
        hidden_layer = nn.TransformerEncoderLayer(embeddings_dim, num_heads, hidden_size, dropout_prob)
        # Define transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(hidden_layer, num_hidden)

        # Define encoder (embedding) layer
        self.encoder = nn.Embedding(num_embeddings, embeddings_dim)
        # Define decoder (linear) layer
        self.decoder = nn.Linear(embeddings_dim, num_embeddings)

        # Initialize squared attention mask
        self.mask = None

        # Store attributes
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden
        self.dropout_prob = dropout_prob

        # Initialize weights
        self.init_weights()

        # # Move network to given device
        # self.to(device)

    @property
    def device(self):
        """ Return network weights device """
        return next(self.parameters()).device

    def init_weights(self):
        """ Initialize weights for hidden layers """
        # Define uniform initialization rannge
        init_range = 0.1
        # Initialize encoder (embedding) layer weights
        self.encoder.weight.data.uniform_(-init_range, init_range)
        # Initialize decoder (linear) layer weights and bias
        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()

    def _generate_square_subsequent_mask(self, size):
        """ Make squared mask

        Args
        size (int)      Size of the squared mask
        """
        # Initialize mask: get boolean lowest triangolar matrix
        mask = torch.tril(torch.ones(size, size)).float()
        # Turn zeroes to minus infinite
        mask = mask.masked_fill(mask == 0, float('-inf'))
        # Turn ones to zeroes
        mask = mask.masked_fill(mask == 1, float(0.0))
        # Return generated mask
        return mask

    def forward(self, x):
        """ Execute when object gets called

        Args
        x (torch.Tensor)        Input tensor
        """
        # Case attention mask is not defined or has wrong shape
        if self.mask is None or self.mask.shape[0] != x.shape[0]:
            # Make new mask
            mask = self._generate_square_subsequent_mask(x.shape[0])
            # Move mask to same device as current network
            self.mask = mask.to(self.device)

        # Encode input tensor
        x_in = self.encoder(x) * math.sqrt(self.embeddings_dim)
        # Make positional encoding
        x_in = self.positional_encoder(x_in)
        # Generate output through transformer encoder
        x_out = self.transformer_encoder(x_in, self.mask)
        # Turn prediction into one hot encoded label
        x_out = self.decoder(x_out)
        # Return one hot encoded label
        return x_out


class PositionalEncoding(nn.Module):

    def __init__(self, embeddings_dim, max_length=1000, dropout_prob=0.3):
        """ Constructor

        Positional encoding handles either relative and absolute position of
        the given tokens in sequence.

        Args
        embeddings_dim (int)        Dimension of a single embedding vector
        max_length (int)            Maximum number of positions to remember
        dropout_prob (float)        Dropout probability for hidden layers
        """
        # Call parent constructor (required)
        super().__init__()
        # Define dropout
        self.dropout = nn.Dropout(p=dropout_prob)

        # Initialize positional encoding (max length x embeddings dim)
        pos_encoding = torch.zeros(max_length, embeddings_dim)
        # Define position indices (max length x 1)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        # Define weights per position (embeddings_dim / 2)
        weight = torch.arange(0, embeddings_dim, 2).float()
        weight *= -(math.log(10000.0) / embeddings_dim)
        weight = torch.exp(weight)

        # Update even positions in positional encoding
        pos_encoding[:, 0::2] = torch.sin(position * weight)
        # Update odd positions in positional encoding
        pos_encoding[:, 1::2] = torch.cos(position * weight)
        # Add one dimension (1 x max length x embeddings dim)
        pos_encoding = pos_encoding.unsqueeze(0)
        # Transpose matrix (max_length x 1 x embeddings_dim)
        pos_encoding = pos_encoding.transpose(0, 1)

        # Store positional encoding tensor
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        """ Execute when object gets called

        Args
        x (torch.Tensor)        Input tensor holding embeddings

        Return
        (torch.Tensor)          Input tensor, some cells will be put to zero
                                according to doropout probability
        """
        # Add positional encoding weights to input tensor
        x = x + self.pos_encoding[:x.shape[0], :]
        # Apply dropout and return updated tensor
        return self.dropout(x)


# Test
if __name__ == '__main__':

    # Dependencies
    from src.dataset.dickens import DickensGE

    # Define root directory
    ROOT_PATH = os.path.dirname(__file__) + '/../..'
    # Define data path
    DATA_PATH = ROOT_PATH + '/data'

    # Get best device (GPU if possible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # show selected device
    print('Selected device is %s' % str(device))

    # Load dataset, spliting it into words
    with open(DATA_PATH + '/great-expectations-dickens.txt') as file:
        # Load text
        text = file.read()
        # Clean text
        text = DickensGE.clean_text(text)
        # Split text into word tokens
        words = DickensGE.split_words(text)

    # Make word dictionary mapping word to indices and vice-versa
    w2i = {word: i for i, word in enumerate(sorted(set(words)))}
    i2w = {i: word for i, word in enumerate(sorted(set(words)))}

    # Parse words to indices
    words = list(map(lambda word: w2i.get(word), words))
    # Cast words to |words| x 1 tensor
    words = torch.tensor(words, dtype=torch.int)
    # Define number of words
    num_words = words.shape[0]
    # Show words shape
    print('Words tensor has shape %s' % str(words.shape))

    # Define batch size
    batch_size = 20
    # Compute number of batches
    num_batches = num_words // batch_size
    # Trim off remainders: batches with size lower than <batch size>
    words = words.narrow(dim=0, start=0, length=(num_batches*batch_size))
    # Reshape tensor according to batch size (allocate in a contiguous area)
    words = words.view(-1, batch_size).contiguous().long()
    # Show new words shape
    print('Words tensor has now shape %s' % str(words.shape))

    # Define length of words
    words_len = words.shape[0]
    # Define splitting index
    i = int(round(words_len * 0.8, 0))
    # Split words tensor into train and test
    train, test = words[:i, :], words[i:, :]

    # Debug
    print('Train data has shape %s' % str(train.shape))
    print('Test data has shape %s' % str(test.shape))

    # Initialize new network
    net = Tracy(
        num_embeddings=len(w2i),
        embeddings_dim=50,
        hidden_size=256,
        num_hidden=2,
        num_heads=2,
        dropout_prob=0.2
    )
    # Move network to selected device
    net = net.to(device)
    # Initialize loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # Initialize optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.5)
    # Set linear rate update according to epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    # Initialize best model
    best_net = net
    # Initialize best evaluation loss
    best_loss = float('inf')

    # Define number of epochs
    num_epochs = 100
    # Define chunk size
    chunk_size = 35
    # Loop through each epoch
    for e in tqdm.tqdm(range(num_epochs), desc='Training'):
        # Set model in training mode
        net.train()
        # Initialize epoch timer
        train_time = time.time()
        # Initialize epoch loss
        train_loss = 0.0
        # Loop through each training batch
        for b, i in enumerate(range(0, train.shape[0] - 1, chunk_size)):
            # Define chunk end index
            j = min(train.shape[0] - 1, i + chunk_size)
            # Get current input and target
            input, target = train[i:j], train[i+1: j+1]
            # Update target shape and move it to selected device
            target = target.view(-1).to(device)

            # # Show input and target tensor shape
            # print('Input has shape %s' % str(input.shape))
            # print('Target tensor has shape %s' % str(target.shape))

            # Reset optimizer gradient
            optimizer.zero_grad()
            # Compute output
            output = net(input.to(device))
            # Compute loss
            loss = loss_fn(output.view(-1, net.num_embeddings), target)
            # Make backward pass
            loss.backward()
            # Clip gradient (inplace)
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            # Make optimizer update
            optimizer.step()

            # Update epoch loss
            train_loss += float(loss.data)

        # Get mean epoch loss, dividing by the number of batches
        train_loss /= (b + 1)
        # Get total epoch time
        train_time = time.time() - train_time

        # Set network in evaluation mode
        net.eval()
        # Initialize test loss
        test_loss = 0.0
        # Initialize test time
        test_time = time.time()
        # Disable gradient computation
        with torch.no_grad():
            # Loop through each test batch
            for b, i in enumerate(range(0, test.shape[0] - 1, chunk_size)):
                # Define chunk end index
                j = min(test.shape[0] - 1, i + chunk_size)
                # Get current input and target
                input, target = test[i:j], test[i+1: j+1]
                # Update target shape and move it to selected device
                target = target.view(-1).to(device)

                # Compute output
                output = net(input.to(device))
                # Compute loss
                loss = loss_fn(output.view(-1, net.num_embeddings), target)
                # Update epoch loss
                test_loss += float(loss.data)

        # Compute mean epoch loss, dividing by the number of batches
        test_loss /= (b + 1)
        # Compute total epoch time
        test_time = time.time() - test_time

        # Make scheduler step
        scheduler.step()

        # In case test loss is lower than current best
        if test_loss < best_loss:
            # Store current model as best model
            best_net = net
            # Update best loss
            best_loss = test_loss

        # Initialize verbose message
        msg = 'Epoch nr %d:\n' % (e + 1)
        msg += 'train loss (mean): %.05f, ' % train_loss
        msg += 'in %.0f seconds (total)\n' % train_loss
        msg += 'test loss (mean): %.05f, ' % test_loss
        msg += 'in %.0f seconds (total)\n' % test_loss
        # Show verbose message
        print(msg)

    # Initialize seed
    seed = 'It is the brave new world'
    # Clean seed
    _seed = DickensGE.clean_text(seed)
    # Split seed into words
    _seed = DickensGE.split_words(_seed)
    # Tokenize words (map to integers)
    _seed = list(map(lambda w: w2i.get(w), _seed))
    # Cast list to long tensor
    input = torch.tensor([_seed[-1]]).long()
    # Add batch size
    input = input.unsqueeze(0)
    # # Move seed to selected device
    # input = input.to(device)

    # Define number of words to predict
    gen_length = 200
    # Initialize generated text
    gen_string = ''
    # Set network in evaluation mode
    net = best_net.eval()
    # Disable gradient computation
    with torch.no_grad():
        # Loop through each word which must be generated
        for i in range(gen_length):
            # Set seed as input
            output = net(input.to(device)).to('cpu')
            # Reshape output
            output = output.view(-1, net.num_embeddings)
            # Get encoded word index
            index = int(torch.argmax(output[-1, :], dim=0))
            # # Debug
            # print('Output')
            # print('Index is %s with shape %s' % (str(index), str(index.shape)))
            # print('Input is %s with shape %s' % (str(input), str(input.shape)))
            # Decode index to string
            word = i2w.get(index)
            # Add word to seed
            gen_string += ' ' + word

            # Update input
            input = input.squeeze(0).tolist() + [index]
            # Cast input to tensor again
            input = torch.tensor(input).long().unsqueeze(0)

    # Print out generated text
    print('Generated text (seed="%s"):' % seed)
    print(seed + gen_string)
