import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


def quantile_classify(metric, sequences, high_cut=0.75, low_cut=0.25):
    """This function creates a new dataframe containing the specified
    metric and sequence and computes a new column, 'class', based on
    the high and low cuts"""

    dataframe = pd.concat([metric, sequences], axis=1)
    
    # convert high and low cut quantiles into values based on the values
    # of the metric
    low_cut = metric.quantile(low_cut)
    high_cut = metric.quantile(high_cut)
    
    # make a histogram of the data to show the locations of the cut points
    hist = metric.hist(bins=100)
    plt.axvline(low_cut, color='k', linestyle='dashed', linewidth=3)
    plt.axvline(high_cut, color='r', linestyle='dashed', linewidth=3)
    
    # function to assign class based on high and low cut
    def assign_class(metric):
        if metric <= low_cut:
            return 0
        elif metric >= high_cut:
            return 1
        return
    # apply to the dataframe then remove values not assigned a class
    dataframe['class'] = dataframe.iloc[:, 0].apply(assign_class)
    dataframe = dataframe[pd.notnull(dataframe['class'])]
    
    counts = pd.value_counts(dataframe['class'])
    print(len(metric), "samples input.")
    print(counts[1], "samples above high cut,", counts[0],
          "samples below low cut,", len(metric) - len(dataframe),
          "samples removed.")

    return dataframe, hist


def value_classify(metric, sequences, high_value, low_value):
    """This function creates a new dataframe containing the specified
    metric and sequence and computes a new column, 'class', based on
    the high and low values specified for metric"""

    dataframe = pd.concat([metric, sequences], axis=1)

    # make a histogram of the data to show the locations of the cut points
    hist = metric.hist(bins=100)
    plt.axvline(low_value, color='k', linestyle='dashed', linewidth=3)
    plt.axvline(high_value, color='r', linestyle='dashed', linewidth=3)
    
    # function to assign class based on high and low cut
    def assign_class(metric):
        if metric <= low_value:
            return 0
        elif metric >= high_value:
            return 1
        return
    # apply to the dataframe then remove values not assigned a class
    dataframe['class'] = dataframe.iloc[:, 0].apply(assign_class)
    dataframe = dataframe[pd.notnull(dataframe['class'])]
    
    counts = pd.value_counts(dataframe['class'])
    print(len(metric), "samples input.")
    print(counts[1], "samples above high value,", counts[0],
          "samples below low value,", len(metric) - len(dataframe),
          "samples removed.")

    return dataframe, hist


def documentize_sequence(seqs, tag, nt_seq):
    """This function converts raw nucleotide or amino acid sequences
    into documents that can be encoded with the Keras Tokenizer().
    If a purification tag is supplied, it will be removed from
    the sequence document. """
    # setup 'docs' for use with Tokenizer
    def seq_to_doc(seq):
        # drop initial tag if it is supplied
        if tag:
            if tag not in seq:
                return None
            seq = seq.split(tag)[1]
        # split the sequence at every letter if not a nt_seq
        if not nt_seq:
            return ' '.join([aa for aa in seq])
        # split the sequence every 3 letters if it is a nt_seq
        return ' '.join([seq[i:i + 3]
                         for i in range(0, len(seq), 3)])

    return seqs.apply(seq_to_doc)


def encode_sequence(sequences, classes, max_length=0,
                    tag=None, nt_seq=True):
    dataframe = pd.concat([classes, sequences], axis=1)
    # if there are more than 4 letters, we should treat as amino acid seq
    if len(list(set(sequences[sequences.index[0]]))) > 4:
        nt_seq = False
    # turn sequence into documents for keras.Tokenizer()
    dataframe['seq_doc'] = documentize_sequence(sequences, tag, nt_seq)
    # drop sequences that weren't divisible by 3
    dataframe = dataframe[pd.notnull(dataframe['seq_doc'])]
    # define sequence documents
    docs = list(dataframe['seq_doc'])
    # create the tokenizer
    t = Tokenizer()
    # fit the tokenizer on the documents
    t.fit_on_texts(docs)
    # integer encode documents
    x = t.texts_to_sequences(docs)
    # pad or truncate sequences if max_length is nonzero
    if max_length:
        x = sequence.pad_sequences(x, maxlen=max_length)
    y = dataframe.loc[:, dataframe.columns[0]].values

    return x, y
