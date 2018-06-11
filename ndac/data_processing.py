import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def quantile_classify(metric, sequence, high_cut=0.75, low_cut=0.25):
    """This function creates a new dataframe containing the specified metric and sequence and computes
    a new column, 'class', based on the high and low cuts"""

    dataframe = pd.concat([metric, sequence], axis=1)
    
    #convert high and low cut quantiles into values based on the values of the metric
    low_cut = metric.quantile(low_cut)
    high_cut = metric.quantile(high_cut)
    
    #make a histogram of the data to show the locations of the cut points
    hist = metric.hist(bins=100)
    plt.axvline(low_cut, color='k', linestyle='dashed', linewidth=3)
    plt.axvline(high_cut, color='r', linestyle='dashed', linewidth=3)
    
    #function to assign class based on high and low cut
    def assign_class(metric):
        if metric <= low_cut:
            return 0
        elif metric >= high_cut:
            return 1
        return
    #apply to the dataframe then remove values not assigned a calss
    dataframe['class'] = dataframe.iloc[:,0].apply(assign_class)
    dataframe = dataframe[pd.notnull(dataframe['class'])]
    
    counts = pd.value_counts(dataframe['class'])
    print(len(metric),"samples input.")
    print(counts[1],"samples above high cut,", counts[0], "samples below low cut,",
          len(metric) - len(dataframe), "samples removed.")
    
    
    return dataframe, hist