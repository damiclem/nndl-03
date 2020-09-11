# Dependencies
import numpy as np
import random
import torch


class OneHotEncode(object):

    # Constructor
    def __init__(self, alphabet):
        """ Constructor

        Save the given alphabet and mappings for turning word/character to
        integers and vice versa.

        Args
        alphabet (iterable)     Set of words/characters to encode
        """
        # Store the alphabet itself
        self.alphabet = alphabet
        # Define mapping from words to integers
        self.encoder = {e: i for i, e in enumerate(alphabet)}
        # Define mapping from integers to words
        self.decoder = {i: e for i, e in enumerate(alphabet)}

    def __call__(self, sentence):
        """ One hot encoder

        Args
        sentence (list)     Sentence as list of words/chars

        Return
        (list)              List where every word/char has been mapped to an
                            integer
        """
        # Define number of words in sentence (rows)
        n = len(sentence)
        # Define number of encoding dimensions (cols)
        m = len(self.alphabet)

        # Map words/chars to integers
        enc = [self.encoder[e] for e in sentence if e in self.alphabet]

        # Initialize one hot encoded sentence as 2d numpy array
        ohe = np.zeros((n, m), dtype=np.int)
        # Make one hot encoding (map integers to binary arrays)
        ohe[np.arange(n), enc] = 1
        # Return one hot encoded sentence as list of lists
        return ohe.tolist()


class WordToVector(object):

    # Constructor
    def __init__(self, words):
        """ Constructor

        Args
        words (iterable)        Set of words/characters to encode
        """
        # Store list of words
        self.words = words
        # Define mapping from words to integers
        self.encoder = {e: i for i, e in enumerate(words)}
        # Define mapping from integers to words
        self.decoder = {i: e for i, e in enumerate(words)}

    # Return word as its index
    def __call__(self, sentence):
        # Make list of labels from words
        labels = [self.encoder[w] for w in sentence if w in self.words]
        # Return list of labels
        return labels


class RandomCrop(object):

    # Constructor
    def __init__(self, crop_len):
        # Store crop length
        self.crop_len = crop_len

    def __call__(self, sentence):
        """ Crop sentence to given length

        Given an input sentence (a list of words/chars), define its total
        length and take a smaller window at random, according to previously set
        crop length.

        Args
        sentence (list)     List of entities in sentence, such as words/chars

        Return
        (list)              A subset of input sentence consisiting in a window
                            whose length is smaller or equal than input
                            sentence length

        Raise
        (IndexError)        In case crop length is higher than sentence length
        """
        # Check for compatibility of crop length with sentence length
        if len(sentence) < self.crop_len:
            # Raise new index error
            raise IndexError(' '.join([
                'Error: given crop length is {:d}'.format(self.crop_len),
                'while current sentence length is {:d}:'.format(len(sentence)),
                'crop length must be smaller or equal than sentence length'
            ]))

        # Define start annd end index of crop window, at random
        i = random.randint(0, len(sentence) - self.crop_len)
        j = i + self.crop_len
        # Take a subset of coriginal sentence
        sentence = sentence[i:j]
        # Return subset
        return sentence


class ToTensor(object):

    # Constructor
    def __init__(self, dtype=torch.float):
        # Store torch type
        self.dtype = dtype

    def __call__(self, sentence):
        """ Turn list to tensor

        Args
        sentence (list)         List of lists (e.g. one-hot encoded sentence is
                                a list of binary vectors representing
                                words/chars).

        Return
        tensor (torch.Tensor)   Float tensor retrieved by parsing input list
        """
        return torch.tensor(sentence, dtype=self.dtype)
