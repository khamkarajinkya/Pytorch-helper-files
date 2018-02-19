"""
Creates lazy load pytorch iterator for custom dataset.

input:
DataFrame/CSV : containing data dictionary
base path: To Load Images from

Output:
Pytorch Tensor datasets for images and labels

"""

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image

class Create_Dataset(Dataset):
    
    def __init__(self,data,base_path,width,height):
        
        self.base_path = base_path
        self.data = data
        self.width = width
        self.height = height
        
    def __getitem__(self,index):
        image = Image.open(self.base_path+"/"+self.data.iloc[index,0]+"/"+self.data.iloc[index,1])
        image = image.resize((self.width,self.height),Image.LANCZOS).convert('RGB')
        image = np.array(image,dtype = np.float32)
        x = torch.from_numpy(image).float()
        y = np.asarray(self.data.iloc[:,2])[index]
        
        return x,y
    
    def __len__(self):
        
        return len(self.data.index)
        
