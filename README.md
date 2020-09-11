# nndl-03

Third exercise for Neural Networks and Deep Learning course @ UniPD

## Abstract

This project aims to the developement of a recurrent deep neural network, such
as Long Short Term Memory (LSTM) or Gated Recurrent Units (GRU),  which
can imitate an author's writing of a novel or an essay.

First step in the development of such deep neural network is to define a
dataset, which will be the text of a chosen book, novel or essay freely
available from [Project Gutenberg](https://www.gutenberg.org/wiki/Main_Page),
then to clean it from useless characters and split it in either characters or
words, according to the chosen method.

Then, three neural networks are being developed: one using characters as inputs,
one using words as inputs and the last one using words either, but using
transfer learning, i.e. an already trained neural network will be used as a base
upon which a new decision layer will be trained to generate some text accoring
to chosen input dataset.

## Dataset

Dataset chosen is the english version of War and Peace by Tolstoj due to its
size: working with Deep Neural Netowrk in general requires a huge amount of
example data and this novel, whit a size of 3.2 Mb, is one of the bigeest found
among texts available on Project Gutenberg.

However, being it transposed from Russian language, there could be characters
involving strange accents, which will then be removed and substituted by common
characters.

Moreover, in every text coming from Project Gutenmebeg first and last rows
are dedicated to Project Gutenmberg itself, e.g. for distribution policies or
updates tracking. Moreover, these texts contain also the title of the text
itself, the title and the index of the chapters. These rows are useless and
noisy for the text generator and must therefore be deleted before training.

## Neural Networks

For each neural network involved, being it based on characters or words, or
being its architecture built from scratch or through transfer learning, it is
important to find out which are the best parameters which we can use. The ideal
way of doing so would be to check at generated text. However, being the training
process very long before we can actually obtain some proper human readable text,
we will instead compare different parameters according to some quantitative
value such as loss and then chose the best one after a short evaluation process.

After selecting the best parameters for a given recurrent neural network, it
must undergo proper training pipeline in order to find its weigths. This will
be a long process however, then we will involve a checkpoint mechanism in order
to save found weigths after a few iterations, ensuring that any interruption to
the training process will not cause a big loss of data. Checkpoint mechaninsm
not only saves neural network weights associated to its architecture, but also
loss and sample of generated text.

### Charlie: character based text generator

### Worden: word based text generator
