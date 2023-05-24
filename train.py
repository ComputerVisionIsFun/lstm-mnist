import pandas as pd
import numpy as np
import mnist_preprocessing as mp
import matplotlib.pyplot as plt
import dataset_mnist as dm
import random
from torch.utils.data import DataLoader
import model_mnist as mm
import parameters as P
import train_process as T
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch


# dataset and dataset
train_df = pd.read_csv('./ mnist/mnist_train.csv')
test_df = pd.read_csv('./ mnist/mnist_test.csv')
train_dataset, test_dataset = dm.dataset_mnist(train_df), dm.dataset_mnist(test_df)
idx = random.randint(0, len(train_dataset) - 1)
train_dataloader = DataLoader(dataset = train_dataset, batch_size=P.BATHCH_SIZE)
test_dataloader = DataLoader(dataset = test_dataset, batch_size=P.BATHCH_SIZE)

# define model, optimizer and loss function
model = mm.model_mnist()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
acc_epochs = T.train(model, P.N_EPOCHS, optimizer, loss_fn, train_dataloader, test_dataloader)

# save weights