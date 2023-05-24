from torch.utils.data import Dataset
import pandas as pd
import torch
import matplotlib.pyplot as plt
import parameters as P


def preprocessing(df:pd.DataFrame):
    labels = df['label'].values.astype('float')
    feature_vecs  = df.drop(columns=['label']).values
    
    return feature_vecs, labels  


class dataset_mnist(Dataset):
    def __init__(self, data:pd.DataFrame):
        self.labels = data['label'].values.astype('int')
        self.features = (data.drop(columns=['label']).values)/255.

    def __len__(self):
        return len(self.labels)
        

    def __getitem__(self, idx):
        features = torch.from_numpy(self.features[idx]).view(P.L, P.INPUT_SIZE)
        label = self.labels[idx]
        
        item = {'features':features, 'label':label}
    
        return item

    def show(self,idx):
        item = self.__getitem__(idx)
        timg, label = item['features'], item['label'].item()
        img = timg.cpu().detach().numpy()
        
        plt.title(str(int(label)))
        plt.imshow(img, cmap='gray')
                   

