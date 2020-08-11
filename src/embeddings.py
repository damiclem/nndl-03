import matplotlib.pyplot as plt
import numpy as np
import re


class Embeddings:

    # Abstract class
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def from_normal(mean, std, dim, words):
        """ Initialize embeddings from normal distribution

        Args
        mean (float)        Mean for normal distribution
        std (float)         Std. dev. for normal distribution
        dim (int)           Embedding dimensions
        words (iterable)    Words (keys) to hold in embeddings

        Return
        (dict)              Dictionary associating words to randomly
                            initialized vectors
        """
        # Initialize embeddings
        embeddings = dict()
        # Loop through each word in text
        for word in words:
            # Initialize random vector
            vector = np.random.normal(loc=mean, scale=std, size=(dim, )).tolist()
            # Store vector word vector
            embeddings.setdefault(word, vector)

    @staticmethod
    def plot_dist(embeddings):
        # Retrieve vectors matrix
        vectors = np.array([*embeddings.values()], dtype=np.float)
        # Compute mean per column
        dist_mean = np.mean(vectors, axis=0)
        # Compute std per column
        dist_std = np.std(vectors, axis=0)

        # Compute average mean value
        avg_mean = np.mean(dist_mean)
        # Compute average std
        avg_std = np.mean(dist_std)

        # Intialize plot
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

        # Set title
        axs[0].set_title('Mean distribution')
        # Plot mean distribution
        axs[0].hist(x=dist_mean, bins=10, density=False)
        # Plot average line
        axs[0].axvline(x=avg_mean, color='r')
        # Add text
        axs[0].text(x=(avg_mean + 0.05), y=20, color='r', s=str(round(avg_mean, 2)))

        # Set title
        axs[1].set_title('Std.dev. distribution')
        # Plot standard deviation distribution
        axs[1].hist(x=dist_std, bins=10, density=False)
        # Plot average line
        axs[1].axvline(x=avg_std, color='r')
        # Add text
        axs[1].text(x=(avg_std + 0.05), y=20, color='r', s=str(round(avg_std, 2)))

        # Show plot
        plt.show()


class Glove(object):

    # Abstract class
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def from_file(path, words=None):
        """ Load embeddings from file

        Args
        path (str)      Path to input glove file
        words (list)    List of words to keep

        Return
        (dict)          Dictionary mapping words to float vectors
        """
        # Intitialize words embeddings (word: vector)
        embeddings = dict()
        # Open file
        with open(path, 'r') as file:
            # Loop through each line
            for line in file:
                # Clean line from newline characters
                line = re.sub(r'[\n\r]+', '', line)
                # Split line according to spaces
                line = re.split(r'\s+', line)
                # Get word (first item in line) and vetcor (other items)
                word, vector = line[0], [float(v) for v in line[1:]]
                # If words is not empty and current word is not required
                if words and (word not in words):
                    # Skip iteration
                    continue
                # Otherwise, save embedding
                embeddings.setdefault(word, vector)
        # Return either list of words and vectors
        return embeddings
