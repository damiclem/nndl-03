import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re


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


def plot_embeddings(embeddings):
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
    sns.distplot(dist_mean, bins=10, norm_hist=False, kde=False, ax=axs[0])
    # Plot average line
    axs[0].axvline(avg_mean, color='darkorange', ls='--')
    # Add text
    axs[0].text(x=(avg_mean + 0.05), y=20, s=str(round(avg_mean, 2)),
                color='darkorange', weight='bold')

    # Set title
    axs[1].set_title('Std.dev. distribution')
    # Plot standard deviation distribution
    sns.distplot(dist_std, bins=10, norm_hist=False, kde=False, ax=axs[1])
    # Plot average line
    axs[1].axvline(x=avg_std, color='darkorange', ls='--')
    # Add text
    axs[1].text(x=(avg_std + 0.05), y=20, s=str(round(avg_std, 2)),
                color='darkorange', weight='bold')

    # Show plot
    plt.show()
