"""

Creates data dictionary which can be exported to csv, text files. Required for memory efficient lazy loading of custom datasets in pytorch

"""

import os
import pandas as pd

base_path = ""

def flatten_list(l):
    
    l = [value for item in l for value in item]
    return l

def create_data_dictionary(base_path):
    
    data = pd.DataFrame()
    files,directory,class_index = [],[],[]
    dirs =  next(os.walk(base_path))[1]
    index = 0            
    for d in dirs:
        f = next(os.walk(base_path+"/"+d+"/"))[2]
        count = len(f)
        files.append(f)
        directory.append([d] * count)
        class_index.append([index] * count)
        index += 1
    
    data['Directory'] = flatten_list(directory)
    data['Files'] = flatten_list(files)
    data['Labels'] = flatten_list(class_index)
    
    return data

def create_train_test_dataset(data,split = 0.7):

    labels = data['Labels'].tolist()
    class_index = dict()
    classes = list(set(labels))
    for item in classes:
        class_index[item] = [i for i,x in enumerate(labels) if x == item]

    train = []
    diff = list(range(len(labels)))

    for item in classes:
        c = len(class_index[item])
        index = random.sample(class_index[item],int(c*split))
        train.extend(index)

    test = list(set(diff) - set(train))

    trainset = data.iloc[train]
    testset = data.iloc[test]
        
    return trainset,testset
