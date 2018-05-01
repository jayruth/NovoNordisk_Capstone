import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers.merge import Concatenate
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# combine low and high expression examples
high_exp = pd.read_csv("high_exp_one_hot.csv", index_col=0)
low_exp = pd.read_csv("low_exp_one_hot.csv", index_col=0)
data_df = pd.concat([high_exp, low_exp], axis=0)

del high_exp 
del low_exp

# convert string data from csv to numpy arrays
def string_to_matrix(string):
    string = str(string)
    list_of_strings = string.split('], [')
    list_of_lists = [channels.strip().replace('[', '').replace(']', '').replace(',', '').split() 
                     for channels in list_of_strings
                     if 'nan' not in list_of_strings
                    ]
    
    remaining_pad = 181 - len(list_of_lists)
    while remaining_pad > 0:
        list_of_lists.append(list([0 for x in range(0, 64)]))
        remaining_pad = remaining_pad - 1
        
    return np.array(list_of_lists).astype(np.float)

data_df['one_hot_matrix'] = data_df['one_hot_matrix'].apply(string_to_matrix)

# get X and y data from data_df
max_len = 181
width = 64

X = np.zeros((22615, max_len, width))
for idx, one_hot_matrix in enumerate(data_df['one_hot_matrix'].values):
    X[idx, :, :] = one_hot_matrix

y = data_df['class'].values

del data_df

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

del X
del y

# tune hyperparameters for simple model

# define simple model per Yoon Kim (2014)
def create_model(filter_sizes=3, num_filters=10):
    # prepare input shape
    input_shape = (181, 64)
    model_input = Input(shape=input_shape)
    z = model_input

    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    z = Dropout(0.5)(z)
    model_output = Dense(1, activation="sigmoid")(z)

    model = Model(model_input, model_output)
    model.compile(loss="binary_crossentropy", optimizer="adam", 
                  metrics=["accuracy"])

    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn=create_model, batch_size=50,
                        epochs=25, verbose=2)
# define the grid search parameters
# model hyperparameters
filter_sizes = [(3, 3, 3), (3, 4, 5), (5, 5, 5),
               (3, 5, 7), (7, 7, 7), (5, 7, 10),
               (10, 10, 10), (3, 4, 5, 6)]
num_filters = [10, 20, 50, 100, 200]


param_grid = dict(filter_sizes=filter_sizes, num_filters=num_filters)

grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    cv=10, n_jobs=4, pre_dispatch='n_jobs')
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

grid_df = pd.DataFrame(grid_result.cv_results_['params'])
grid_df['means'] = grid_result.cv_results_['mean_test_score']
grid_df['stddev'] = grid_result.cv_results_['std_test_score']

# print results to csv file
grid_df.to_csv('grid_search_results.csv')
