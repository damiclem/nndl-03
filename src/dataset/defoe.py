# Dependencies
from src.dataset.dataset import MyDataset
import re


class DefoeJP(MyDataset):

    # Constructor
    def __init__(self, text='', split_fn=None, min_length=0, transform=None):
        # Split text in lines
        text = re.split(r'\n', text)
        # Remove legal notes and index
        text = text[50:9517]
        # Rejoin text
        text = '\n'.join(text)
        # Call parent constructor
        super(DefoeJP, self).__init__(
            text=text,
            split_fn=split_fn,
            min_length=min_length,
            transform=transform
        )
