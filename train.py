import pandas as pd
import matplotlib.pyplot as plt
import dataset_mnist as dm
import random
from torch.utils.data import DataLoader
import model_mnist as mm
import parameters as P
import training_process as T
import torch.optim as optim
import torch.nn as nn
from joblib import dump


# dataset and dataset
train_df = pd.read_csv('./mnist/mnist_train.csv')
val_df = pd.read_csv('./mnist/mnist_test.csv')
train_dataset, val_dataset = dm.dataset_mnist(train_df), dm.dataset_mnist(val_df)
idx = random.randint(0, len(train_dataset) - 1)
train_dataloader = DataLoader(dataset = train_dataset, batch_size=P.BATHCH_SIZE)
val_dataloader = DataLoader(dataset = val_dataset, batch_size=P.BATHCH_SIZE)

# define model, optimizer and loss function
model = mm.model_mnist()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
acc_epochs = T.train(model, P.N_EPOCHS, optimizer, loss_fn, train_dataloader, val_dataloader, P.EARLY_STOP, P.MODEL_SAVE_PATH)

# Plotting acc and save acc_epochs
plt.plot(range(1, len(acc_epochs['train']) + 1), acc_epochs['train'], label='Train')
plt.plot(range(1, len(acc_epochs['val']) + 1), acc_epochs['val'], label='Test')
plt.ylabel('accuracy', fontsize = 13)
plt.xlabel('epoch(s)', fontsize = 13)
plt.title('Accuray During Training', fontsize = 13)
plt.legend(fontsize = 13)
plt.savefig(P.ACCURACY_DURING_TRAINING_PLOT_SAVE_PATH)
dump(acc_epochs, P.ACCURACY_DURING_TRAINING_SAVE_PATH)