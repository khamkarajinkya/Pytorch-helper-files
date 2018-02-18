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
    
