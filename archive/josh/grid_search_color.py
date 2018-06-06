import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# combine low and high expression examples
high_exp = pd.read_csv("high_exp.csv", index_col=0)
low_exp = pd.read_csv("low_exp.csv", index_col=0)
data_df = pd.concat([high_exp, low_exp], axis=0)


# convert string data from csv to numpy arrays
def string_to_matrix(color_string):
    color_string = str(color_string)
    list_of_strings = color_string.replace('[', '').replace(']', '').split('\n')
    list_of_lists = [channels.strip().split()
                     for channels in list_of_strings
                     if 'nan' not in list_of_strings
                     ]
    remaining_pad = 181 - len(list_of_lists)
    while remaining_pad > 0:
        list_of_lists.append(list([0, 0, 0, 0]))
        remaining_pad = remaining_pad - 1
    return np.array(list_of_lists).astype(np.float)


data_df['color_matrix'] = data_df['color_matrix'].apply(string_to_matrix)

# get X and y data from data_df
X = np.zeros((22604, 181, 4))
for idx, colors in enumerate(data_df['color_matrix'].values):
    X[idx, :, :] = colors
y = data_df['class'].values

# create train test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


# tune hyperparameters for simple model

# define simple model per Yoon Kim (2014)
def create_model(filter_size=3, num_filters=10,
                 pool_size=2, hidden_dims=50,
                 ):
    model = Sequential()
    model.add(Conv1D(num_filters, filter_size, activation='selu', input_shape=(181, 4)))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn=create_model, batch_size=50,
                        epochs=10, verbose=0)
# define the grid search parameters
# model hyperparameters
filter_size = [3, 5, 7, 10]
num_filters = [10, 20, 50, 100, 200]
pool_size = [2, 5, 10, 20, 50]
# pool_stride = [2, 3, 5]
# dropout_prob = 0.5
hidden_dims = [10, 50, 100]

# Training parameters
# batch_size = 50
# epochs = 20

param_grid = dict(filter_size=filter_size, num_filters=num_filters,
                  pool_size=pool_size, hidden_dims=hidden_dims)

grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    cv=10, n_jobs=-1)
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


grid_df = pd.DataFrame(grid_result.cv_results_['params'])
grid_df['means'] = grid_result.cv_results_['mean_test_score']
grid_df['stddev'] = grid_result.cv_results_['std_test_score']

# print results to csv file
grid_df.to_csv('grid_search_results.csv')
