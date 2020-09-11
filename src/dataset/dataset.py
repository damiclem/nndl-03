# Dependencies
import torch.utils.data as tu
import unidecode as ud
import torch
import os
import re


class MyDataset(tu.Dataset):

    # Allowed punctuation
    punctuation = r'\.\,\!\?"'
    # Allowed charachters
    characters = r'a-zA-Z\''

    # Constructor
    def __init__(self, text='', split_fn=None, min_length=0, transform=None):
        """ Constructor

        Args
        text (str)                      Text used as dataset
        split_fn (function)             Function used to split paragraphs
        min_length (int)                Minimum paragraph length
        transform (torch.Transform)     Paragraph transformation to apply
        """

        # Remove noise from text
        text = self.clean_text(text)
        # Define sentences
        paragraphs = self.split_paragraphs(text)

        # Check how to split sentences
        if callable(split_fn):
            # Split each paragraph
            paragraphs = list(map(split_fn, paragraphs))

        # Keep only sentences whose length is greater than threshold
        paragraphs = [p for p in paragraphs if len(p) >= min_length]

        # Store paragraphs
        self.paragraphs = paragraphs
        # Store paragraph transformation pipeline
        self.transform = transform

    def __len__(self):
        """ Length of the dataset

        Return
        (int)       Number of paragraphs in text
        """
        return len(self.paragraphs)

    def __getitem__(self, i):
        """ Random access to paragraphs

        Args
        i (int)         Index of chosen paragraphs

        Return
        (str)           Paragraph at index i

        Raise
        (IndexError)    In case index i-th paragraph is not available
        """
        # Case index i is greater or equal than number of paragraphs
        if i >= len(self.paragraphs):
            # Raise key error
            raise IndexError('Chosen index exceeds paragraph indices')

        # Case transform is not set
        if self.transform is None:
            # Just return i-th paragraph
            return self.paragraphs[i]

        # Otherwise, transform it and return transformation result
        return self.transform(self.paragraphs[i])

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        # Check if given file exists
        if not os.path.isfile(path):
            # Throw exception: file must be there
            raise FileNotFoundError('input file does not exist')

        # Initialize input text
        text = ''
        # Open file
        with open(path, 'r') as file:
            # read entire file
            text = file.read()

        # Make new dataset instance
        return cls(text=text, *args, **kwargs)

    @classmethod
    def clean_text(cls, text):
        """ Clean text

        Args
        text (str)      Input text to clean

        Return
        (str)           Cleaned input text
        """
        # Remove non-unicode characters
        text = ud.unidecode(text)
        # Lowarcase all
        text = text.lower()
        # Substitute single newlines with blank spaces
        text = re.sub(r'(?<![\r\n])(\r?\n|\n?\r)(?![\r\n])', ' ', text)
        # Substitute double newlines with single ones
        text = re.sub(r'[\n]+', r'\n', text)
        # Remove starting spaces
        text = re.sub(r'\n[\t ]+', '', text)
        # Keep only allowed punctuation and words
        text = re.sub(r'[^{:s}{:s} \n]'.format(cls.characters, cls.punctuation), ' ', text)
        # Remove multiple spaces
        text = re.sub(r'[\t ]+', ' ', text)
        # Return cleaned text
        return text

    @classmethod
    def split_paragraphs(cls, text):
        """ Split text into paragraphs

        Args
        text (str)      Text string to split into paragraphs
        """
        # Split according to newlines
        return list(re.findall(r'([^\n]+\n{,1})', text))

    @classmethod
    def split_words(cls, text):
        # Get characters and punctuation
        chars, punct = cls.characters, cls.punctuation
        # Separate prefix and word
        text = re.sub(r'(?<=[{:s} \n])(\')'.format(punct), r' \1 ', text)
        # Separate suffix and word
        text = re.sub(r'(\'s|\'ll|\'t)(?!>[a-zA-Z])', r' \1 ', text)
        # Separate single points
        text = re.sub(r'(?<![\.])([\.])(?![\.])', r' \1 ', text)
        # Separate multiple points from othe punctioation
        text = re.sub(r'\.{2,}', r' ... ', text)
        # Separate words and punctuation
        text = re.sub(r'([{:s}\n])'.format(punct[2:]), r' \1 ', text)

        # find all space separated words
        words = re.findall(r'[^ ]+', text)
        # Return words
        return list(words)

    @classmethod
    def split_chars(cls, text):
        """ Split text into characters

        Args
        text (list)         Input text string

        Return
        (list)              Text splitted in a list of characters
        """
        return list(text)


def train_test_split(dataset, train_prc=0.8):
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


# Unit testing
if __name__ == '__main__':

    # Dependencies
    from src.dataset.transform import RandomCrop, OneHotEncode, ToTensor

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
    dataset = MyDataset(sample, split_fn=MyDataset.split_words)
    # Show ten paragraphs
    print('Paragraphs (10):')
    print(dataset.paragraphs[:10])

    # # Print cleaned text
    # print('Cleaned text:')
    # print(cleaned)
    # print()
    #
    # # Split text into sentences
    # sentences = MyDataset.split_sentences(cleaned)
    # # Show sentences list start
    # print('Sentences:')
    # # Loop through each splitted sentence
    # for i in range(len(sentences)):
    #
    #     print('  {:d}-th sentence:'.format(i+1))
    #     print('  ' + sentences[i])
    #
    #     # # Split sentence into words
    #     # print('  {:d}-th sentence words:'.format(i+1))
    #     # print('  ' + WarAndPeace.split_words(sentences[i]))
    #
    #     # Slit sentence into characters
    #     print('  {:d}-th sentence chars:'.format(i+1))
    #     print('  ' + str(MyDataset.split_chars(sentences[i])))
    #
    # # Define random crop trasformation
    # random_crop = RandomCrop(30)
    #
    # # Load new dataset
    # dataset = MyDataset(sample, split_how='chars', min_len=50, transform=random_crop)
    # # Initialize output message
    # msg = ['Third sentence sampled three times (full dataset):']
    # # Sample the same sentence three times
    # msg += [repr(''.join(dataset[0]))]
    # msg += [repr(''.join(dataset[0]))]
    # msg += [repr(''.join(dataset[0]))]
    # # Show output message
    # print('\n'.join(msg), end='\n\n')
    #
    # # Split dataset in train and test
    # train_dataset, test_dataset = split_train_test(dataset, 0.7)
    # # Initialize output message
    # msg = ['Total dataset length: {:d}'.format(len(dataset))]
    # msg += ['Train dataset length: {:d}'.format(len(train_dataset))]
    # msg += ['Test dataset length: {:d}'.format(len(test_dataset))]
    # # Show output message
    # print('\n'.join(msg), end='\n\n')
    #
    # # Initialize output message
    # msg = ['Third sentence sampled three times (train dataset):']
    # # Sample the same sentence three times
    # msg += [repr(''.join(train_dataset[0]))]
    # msg += [repr(''.join(train_dataset[0]))]
    # msg += [repr(''.join(train_dataset[0]))]
    # # Show output message
    # print('\n'.join(msg), end='\n\n')
    #
    # # # Load new dataset
    # # war_and_peace = WarAndPeace(
    # #     file_path = os.path.join(DATA_PATH, 'war-and-peace-tolstoj.txt'),
    # #     split_how='chars',
    # #     min_len=10
    # # )
    # #
    # # # Retrieve alphabet
    # # alphabet = set([
    # #     war_and_peace[i][j]
    # #     for i in range(len(war_and_peace))
    # #     for j in range(len(war_and_peace[i]))
    # # ])
    # #
    # # # Define transformer
    # # war_and_peace.transform = transforms.Compose([
    # #     RandomCrop(7),
    # #     OneHotEncode(alphabet),
    # #     ToTensor()
    # # ])
    # #
    # # # Initialize timers
    # # time_beg, time_end = time(), 0.0
    # # # Define batch size
    # # batch_size = 1000
    # # # Compute number of batches
    # # num_batches = math.ceil(len(war_and_peace) / batch_size)
    # # # Define a dataloader
    # # dataloader = DataLoader(war_and_peace, batch_size=batch_size, shuffle=True)
    # # # Loop through each batch
    # # for batch in dataloader:
    # #     # Skip
    # #     pass
    # # # Update timer
    # # time_end = time()
    # # # Verbose
    # # print('Took {:.0f} seconds'.format(time_end - time_beg), end=' ')
    # # print('to go through {:d} batches'.format(num_batches))
    # # print()
    # #
    # # # Split dataset in two
    # # train, test = split_train_test(war_and_peace, 0.8)
    # # # Check lengths
    # # print('Original dataset has size {:d}'.format(len(war_and_peace)), end=' ')
    # # print('while train dataset has length {:d}'.format(len(train)), end=' ')
    # # print('and test dataset has length {:d}'.format(len(test)))
    # # print()
    # #
    # # # Loop train dataset first 5 sentences
    # # print('Train dataset head:')
    # # for i in range(5):
    # #     print('{:d}-th sentence'.format(i+1))
    # #     print(train[i])
    # #     print()
    # #
    # # # Loop test dataset first 5 sentences
    # # print('Test dataset head:')
    # # for i in range(5):
    # #     print('{:d}-th sentence'.format(i+1))
    # #     print(test[i])
    # #     print()
