import numpy as np
import re

train_file = 'train.tsv'
embed_file = 'wiki.en.vec'

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels( train_file ):
    x_text = []
    labels = []
    dict_label = {'pants-fire': 0, 'false':1, 'barely-true':2, 'half-true':3, 'mostly-true':4, 'true':5} 
    
    s = [0,1,2,3,4,5]
    num_labels = 6
    m = np.zeros(( len(s), num_labels ))
    m[ np.arange(num_labels), s ] = 1
    m = m.tolist()

    with open( train_file, "r" ) as f:
            content = f.readlines()
    
    for line in content:
        split_line = line.split('\t')
        label, stmt, sub, speaker, title, state_info, party, context = split_line
        x_text.append( clean_str( stmt.strip() ) )
        labels.append( m[dict_label[label]] )
        
    return x_text, labels 

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    res1, res2 = load_data_and_labels( train_file )
