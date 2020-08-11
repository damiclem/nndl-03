from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import unidecode as ud
import random
import os
import re


class WarAndPeace(Dataset):

    # Constructor
    def __init__(self, file_path, split_how=None, min_len=1, transform=None):

        # Check if file exists
        if not os.path.isfile(file_path):
            # Throw exception: file must be there
            raise FileNotFoundError('Input file does not exist')

        # Open input file buffer
        file = open(file_path, 'r')
        # Retrieve text from file
        text = file.read()
        # Close input file buffer
        file.close()

        # Remove noise from text
        text = self.clean_text(text)
        # Split text into sentences
        sentences = self.split_sentences(text)

        # Check how to split sentences
        if split_how is None:
            # No splitting method set
            pass
        elif split_how == 'words':
            # Split sentences according to words
            sentences = [self.split_words(s) for s in sentences]
        elif split_how == 'chars':
            # Split sentences according to chars
            sentences = [self.split_chars(s) for s in sentences]
        # Case no valid splitting method has been set
        else:
            # Raise new error
            raise ValueError(' '.join([
                'Given splitting method is not valid:',
                'must be one of ("words", "chars" or None)',
                '{} set instead'.format(split_how)
            ]))

        # Apply minimum sentence length threshold
        if min_len:
            # Keep only sentences whose length is greater than threshold
            sentences = [s for s in sentences if len(s) >= min_len]

        # Store (eventually splitted) sentences
        self.sentences = sentences
        # Store sentences transformation pipeline
        self.transform = transform

    def __len__(self):
        """ Length of the dataset

        Return
        (int)       Number of sentences in text
        """
        return len(self.sentences)

    def __getitem__(self, i):
        """ Random access sentence

        Args
        i (int)     Index of chosen sentence

        Return
        (str)       Sentence at index i

        Raise
        (IndexError)  In case index i-th sentence is not available
        """
        # Case index i is greater or equal than number of sequences
        if i >= len(self.sentences):
            # Raise key error
            raise IndexError('Chosen index exceeds sentences indices')

        # Case transform is not set
        if self.transform is None:
            # Just return i-th sentence
            return self.sentences[i]

        # Otherwise, transform it and return transformation result
        return self.transform(self.sentences[i])

    @staticmethod
    def clean_text(text):
        """ Clean text

        Takes as input the book text and removes strange accents through
        unidecode library. Them removes book titles, subtitles and chapters.
        Moreover, it removes strange characters in between two words, inside
        the text.

        Args
        text (str)      Input text to clean

        Return
        (str)           Cleaned input text
        """
        # Remove non-unicode characters
        text = ud.unidecode(text)
        # Lowarcase all
        text = text.lower()
        # Remove books titles
        text = re.sub(r'^book [\w]+: [0-9\- ]+\n$', '', text)
        # Remove chapters
        text = re.sub(r'^chapter [ivx]+\n$', '', text)
        # Remove single newlines
        text = re.sub(r'(?<!\n)\n', ' ', text)
        # Remove udensired punctuation and numbers
        text = re.sub(r'[\*\=\/\d]', '', text)
        # Remove undesired symbols between words
        text = re.sub(r'(?<=\D)[-]+(?=(\D))', ' ', text)
        # Remove double spaces
        text = re.sub(r'[\t ]+', ' ', text)
        # Return cleaned text
        return text

    @staticmethod
    def split_sentences(text):
        """ Split text into sentences

        Split text (string) into sentences (list of strings) according to two
        splitting criterion: (1) Split according to newline character, which is
        sometimes (every time after cleaning) used as sentence stopping
        character. (2) Split according to punctuation (eventually include ending
        doube quotes);

        Args
        text (str)      Input text to be splitted into sentences

        Return
        (list)          List of string sentences
        """
        # Split text into sentences
        return list(re.findall(r'([^\.\!\?\n]+[\.\!\?]+["]{,1}[ ]{,1})', text))

    @staticmethod
    def split_words(text):
        """ Split text into words

        Args
        text (str)          Input text string

        Return
        (list)              Text splitted in a list of words
        """
        # Split sentence into words according to punctuation
        words = list(re.split(r'([ \n\:\;\"\,\(\)\.\!\?])', text))
        # Remove useless characters
        words = [w for w in words if re.search('[^| ]', w)]
        # Return splitted words
        return words

    @staticmethod
    def split_chars(text):
        """ Split text into characters

        Args
        text (list)         Input text string

        Return
        (list)              Text splitted in a list of characters
        """
        return list(text)


def split_train_test(dataset, train_prc=0.8):
    """ Split input dataset

    Split input in two: one for train and one for test, with sentences being
    put in one or in the other according to defined proportion,

    Args
    dataset (Dataset)       The whole dataset which must be split
    train_prc (float)       Percentage of sentences to be assigned to
                            training dataset

    Return
    (Dataset)       Train dataset
    (Dataset)       Test dataset
    """
    # Define dataset length
    n = len(dataset)
    # Define number of training dataset indices
    m = round(train_prc * n)
    # Split datasets in two
    return torch.utils.data.random_split(dataset, [m, n - m])


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
        self.alphabet = set(alphabet)
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
        # Map words/chars to integers
        encoded = [self.encoder[e] for e in sentence if e in self.alphabet]
        # For each word/char in sentence, map it to vector
        encoded = [
            # Make a vector: set 1 only wher index is equal to word/char number
            [int(encoded[i] == j) for j in range(len(self.alphabet))]
            # Loop through each word/char in sentence
            for i in range(len(encoded))
        ]
        # Return one hot encoded sentence
        return encoded


class WordToVector(object):

    # Constructor
    def __init__(self, words):
        """ Constructor

        Args
        words (iterable)        Set of words/characters to encode
        """
        # Store list of words
        self.words = set(words)
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

    def __call__(self, sentence):
        """ Turn list to tensor

        Args
        sentence (list)         List of lists (e.g. one-hot encoded sentence is
                                a list of binary vectors representing
                                words/chars).

        Return
        tensor (torch.Tensor)   Float tensor retrieved by parsing input list
        """
        return torch.tensor(sentence).float()


# Unit testing
if __name__ == '__main__':

    # Define path to project root
    ROOT_PATH = os.path.join(os.path.dirname(__file__), '..')
    # Define path to data folder
    DATA_PATH = os.path.join(ROOT_PATH, 'data')

    # Define semple input text, taken from War and Peace novel by Tolstoj
    sample = """“The past always seems good,” said he, “but did not Suvórov
himself fall into a trap Moreau set him, and from which he did not know
how to escape?”
“Who told you that? Who?” cried the prince. “Suvórov!” And he
jerked away his plate, which Tíkhon briskly caught. “Suvórov!...
Consider, Prince Andrew. Two... Frederick and Suvórov; Moreau!...
Moreau would have been a prisoner if Suvórov had had a free hand; but
he had the Hofs-kriegs-wurst-schnapps-Rath on his hands. It would have
puzzled the devil himself! When you get there you’ll find out what
those Hofs-kriegs-wurst-Raths are! Suvórov couldn’t manage them so
what chance has Michael Kutúzov? No, my dear boy,” he continued,
“you and your generals won’t get on against Buonaparte; you’ll
have to call in the French, so that birds of a feather may fight
together. The German, Pahlen, has been sent to New York in America, to
fetch the Frenchman, Moreau,” he said, alluding to the invitation made
that year to Moreau to enter the Russian service.... “Wonderful!...
Were the Potëmkins, Suvórovs, and Orlóvs Germans? No, lad, either you
fellows have all lost your wits, or I have outlived mine. May God help
you, but we’ll see what will happen. Buonaparte has become a great
commander among them! Hm!...”
“I don’t at all say that all the plans are good,” said Prince
Andrew, “I am only surprised at your opinion of Bonaparte. You
may laugh as much as you like, but all the same Bonaparte is a great
general!”
“Michael Ivánovich!” cried the old prince to the architect who,
busy with his roast meat, hoped he had been forgotten: “Didn’t
I tell you Buonaparte was a great tactician? Here, he says the same
thing.”
“To be sure, your excellency,” replied the architect.
The prince again laughed his frigid laugh."""

    # Try cleaning feature
    cleaned = WarAndPeace.clean_text(sample)
    # Print cleaned text
    print('Cleaned text:')
    print(cleaned)
    print()

    # Split text into sentences
    sentences = WarAndPeace.split_sentences(cleaned)
    # Loop through each splitted sentence
    for i in range(len(sentences)):

        print('{:d}-th sentence:'.format(i+1))
        print(sentences[i])
        print()

        # Split sentence into words
        print('{:d}-th sentence words:'.format(i+1))
        print(WarAndPeace.split_words(sentences[i]))
        print()

        # Slit sentence into characters
        print('{:d}-th sentence chars:'.format(i+1))
        print(WarAndPeace.split_chars(sentences[i]))
        print()

    # Load new dataset
    war_and_peace = WarAndPeace(
        file_path = os.path.join(DATA_PATH, 'war-and-peace-tolstoj.txt'),
        split_how='chars',
        transform=transforms.Compose([
            OneHotEncode(alphabet={'a', 'b', 'c'}),
            ToTensor()
        ])
    )

    # Split dataset in two
    train, test = split_train_test(war_and_peace, 0.8)
    # Check lengths
    print('Original dataset has size {:d}'.format(len(war_and_peace)), end=' ')
    print('while train dataset has length {:d}'.format(len(train)), end=' ')
    print('and test dataset has length {:d}'.format(len(test)))
    print()

    # Loop train dataset first 5 sentences
    print('Train dataset head:')
    for i in range(5):
        print('{:d}-th sentence'.format(i+1))
        print(train[i])
        print()

    # Loop test dataset first 5 sentences
    print('Test dataset head:')
    for i in range(5):
        print('{:d}-th sentence'.format(i+1))
        print(test[i])
        print()
