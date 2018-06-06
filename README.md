# NOVO-DNA/AA-Classfication (NDAC): Multiclassification of Protein Characteristics from Nucleotide or Amino Acid Sequences 

This repository contains the source code for our DIRECT capstone project with Novo Nordisk.

**Download Software**

In command line, type: pip install Novo-dna-aa

**Software Description**

NDAC is a software package that provides multiple classification identifiers for a specific protein characteristics(e.g. protein expression, protein solubility) predicted from nucleotide or amino acid sequence data via a trained Long short-term memory/convolution neural network (LSTM-CNN) architecture. Here, the front-end user can either: (1) Input training data, consisting of both proteing property and sequence data, followed by retraining of the saved LSTM-CNN model with optimized hyperparameters using data from Sastry et. al ( https://github.com/SBRG/Protein_ML) and predicting of various protein property classes (e.g. high, medium or low expression). 
(2) Predict protein property classes of input test data, consisting primarily of nucleotide sequence, from the LSTM-CNN model trained from the same Sastry et. al sequence dataset. (3) Simply encode nucleotide data or amino acid data via one-hot or color encoding. in addition to padding sequences for nueral network training. 

By predicting class identifiers for protein properties of interest, more rapid screening of ineffective nucleotide sequences can occur, possibly resulting in reduced usage of resources(fewer number of experiments) during optimization of biologic products made from transfected eukaryotic or prokaryotic expression platforms and reduce time from drug discovery to clinical studies

**Software Dependencies:**

Python version 3

Python Packages: Keras, Tensorflow, Scikit-Learn, Matplotlib, Numpy, Pandas, HDF5

Mac OS X and Windows are both able to download python 3 without any dependencies. a `conda install "your library package"` code will need to be run on the terminal or git-Bash in order to install Keras, Tensorflow, Scikit-Learn, Matplotlib, Numpy, Pandas, HDF5. 

**Data Collection and Sequence Encoding**

*Data Collection*
    - Protein expression levels and sequence data are collected from a specified data source.  The primary data source will be the data published in Sastry et. al., but any other source of data containing a metric (e.g. protein expression level, solubility) and a sequence (e.g. nucleotide sequence, amino acid) may be used.
      *Components*
            - Pandas Module
  
*Sequence Encoding*
     - Starting with a nucleotide or amino acid sequence generate an encoded sequence that can be fed into a machine learning model.  Encoding styles may include color, one-hot, or another strategy.  Encoded sequences need to be padded so all have the same length.
      
      *Components*
            - One Hot Encoder
            - Color Encoder
      
      *Test Cases*
            - Feed in various length sequences and return data that is all the same length
            - Feed in a short sequence and verify the encoding results match what is expected 
      
      *Padding*
          - Following encoding, sequences will be padded to ensure the batch size is preserved during prediction of classes or training of data
 
**LSTM-CNN Sequence Embedding and Architecture Training**

*Sequence Embedding*
          - Starting with the give nucleotide sequence, encoded dictionaries can be created from either the multiples of 3 nucleotides as read from the nucleotide sqeuence or using 64 codons; here, the dictionary will used to create a more dense representation of the provided sequences. Here, embedding length is determined from hyperparameter tuning, optimized from the LSTM-CNN architecture.
    
   *Padding*
          - Following embedding, sequences will be padded to ensure the batch size is preserved during prediction of classes or training of data
   
*Data Classification*
          - Sequences are classified into 2 or more classes based on the value of the metric. For example as in Sastry et. al.  protein expression levels are used to classify each sequence into high and low expression groups based on relative cut off levels (e.g. 1st and 4th quartiles).  In the case of Sastry et. al sequences are labeled with a 0 for low/unacceptable expression, 1 for high/acceptable expression, and the middle 2 quartiles are discarded. 
        *Components*
              - Two Class Classifier (eg. high/low)
              - Multi Class Classifier (eg. high, medium, low)
       *Test Cases*
              - Pass the functions for small dataframes that should have a known number of each class and verify the results are as expected.

*Protein Property Prediction Model Training*
          - Starting with encoded sequence data (nucleotide or amino acid), generate a model that predicts the class of each sequence. 
   *Components*
          - Multiple filter width model
          - Multiple sets of filters model
          - Embedding/LSTM model
          - Model saver
   *Test Cases*
          - This will be using mostly Keras functions.  Write a simple test to make sure a model is saved.


**LSTM-CNN Model Prediction and Interpretation**

*Protein Property Prediction*
          - Using the previously trained and saved model feed in new data and make class predictions. 
      *Components*
           - Model predict
           - ROC Curves 
           - Test-train accuracy plots
      *Test Cases*
           - This will use mostly Keras functions.  Write a simple test to make sure a model can be loaded to make a prediction. 


#### All the functions used in the software use cases are available for your modification

*For running tests:*
In a shell script, open the directory "UnitTests", type in and run: nosetests -verbose Test.(the name of 'test.py' file desired to run) based on the use cases you want to access.
