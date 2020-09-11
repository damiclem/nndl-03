# Dependencies
from src.dataset.transform import OneHotEncode, WordToVector, ToTensor
from src.dataset.dataset import MyDataset
from src.network.charlie import Charlie
from src.network.worden import Worden
import argparse
import torch
import json
import os


# Main
if __name__ == '__main__':

    # Define data path
    DATA_PATH = './data'

    # Initialize argument parser
    parser = argparse.ArgumentParser('Test RNN model')
    # Add argument: type of network (whetherr to use words or chars)
    parser.add_argument(
        '--type', type=str, default='words',
        help='Whether to use words (`words`) or characters (`chars`) as entities'
    )
    # Add argument: seed
    parser.add_argument(
        '--seed', type=str, default='This is an example seed',
        help='Seed used for text generation'
    )
    # Add argument: generated text length
    parser.add_argument(
        '--length', type=int, default=100,
        help='Length of generated text (in words or characters according to selected model)'
    )
    # Add argument: whether to apply softmax or not
    parser.add_argument(
        '--softmax', type=int, default=1,
        help='Whether to apply softmax or not'
    )
    # Parse arguments
    args = parser.parse_args()

    # Case selected network type is not valid
    if args.type not in ['chars', 'words']:
        raise ValueError('only `chars` of `words` values are recognized')

    # Initialize network
    net = None
    # Case selected network is characters
    if args.type == 'chars':

        # Define one hot encoding transformer
        one_hot_encode = OneHotEncode([
            '\n', ' ', '!', '"', "'", ',', '.', '?', 'a', 'b', 'c', 'd', 'e',
            'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
            's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ])

        # Define cast to tensor transformer
        to_tensor = ToTensor()

        # Define encode function
        def encode_fn(string):
            # Clean input text
            string = MyDataset.clean_text(string)
            # Split input string in characters
            chars = MyDataset.split_chars(string)
            # One hot encode characters
            encoded = one_hot_encode(chars)
            # Cast encoded sentence to tensor
            return to_tensor(encoded).unsqueeze(0)

        # Define decode function
        def decode_fn(index):
            return one_hot_encode.decoder.get(index)

        # Load network
        net, _ = Charlie.from_file(path=DATA_PATH + '/charlie/best/net.pth')

    # Case selected network is words
    if args.type == 'words':
        # Open embeddings file
        with open(DATA_PATH + '/worden/embeddings.json', 'r') as file:
            # Load embeddings dictionary
            embeddings = json.load(file)

        # Define word to vector transformer
        word_to_vector = WordToVector([*embeddings.keys()])

        # Define cast to tensor transformer
        to_tensor = ToTensor()

        # Define encode function
        def encode_fn(string):
            # Clean input text
            string = MyDataset.clean_text(string)
            # Split input string in words
            words = MyDataset.split_words(string)
            # Turn unknown words into unknown string
            words = [(w if w in word_to_vector.words else '') for w in words]
            # One hot encode characters
            encoded = word_to_vector(words)
            # Cast encoded sentence to tensor
            return to_tensor(encoded).unsqueeze(0)

        # Define decode function
        def decode_fn(index):
            return word_to_vector.decoder.get(index)

        # Load network
        net, _ = Worden.from_file(path=DATA_PATH + '/worden/best/net.pth')

    # Move network to cpu
    net.cpu()
    # Set network in evaluation mode
    net.eval()
    # Disable gradient computation
    with torch.no_grad():
        # Generate text
        generated = net.generate_text(
            # Set seed sentence
            args.seed,
            # Number of characters to predict
            args.length,
            # Set encoding function
            encode_fn=encode_fn,
            # Set decoding function
            decode_fn=decode_fn,
            # Decision function
            decision_how=('softmax' if args.softmax else 'argmax')
        )

    # Show generated text
    print('Generated text (seed: "%s")' % args.seed)
    print(generated)
