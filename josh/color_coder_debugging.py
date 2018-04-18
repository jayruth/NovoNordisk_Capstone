import pandas as pd
import numpy as np
import matplotlib

import json

# prep data
data = pd.read_csv('DF_all_w_codon_counts.csv', index_col=0)
def nt_seq_to_list(nt_sequence):
    return np.array(list(map(''.join, zip(*[iter(str(nt_sequence))]*3))))

data['codon_array'] = data['true_nt_seq'].apply(nt_seq_to_list)

# import json color_dict
with open('codon_colors.json', 'r') as fp:
    codon_colors = json.load(fp)


# convert codon_array to color_matrix

def list_to_color_matrix(codon_array):
    # mark for garbage if codon_array is nan
    if len(codon_array) < 10:
        return np.nan

    # otherwise return color matrix
    color_matrix = np.ndarray(shape=(len(codon_array), 4))
    for idx, codon in enumerate(codon_array):
        color_matrix[idx] = np.array(codon_colors[codon])
    return color_matrix

data['color_matrix'] = data['codon_array'].apply(list_to_color_matrix)

print('lol')
