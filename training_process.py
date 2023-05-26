import torch.optim as optim
import parameters as P
import torch


def train(model, n_epochs, optimizer, loss_fn, train_dataloader, val_dataloader, early_stop, model_save_path)->dict:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 
    model.to(device)
    size_train = train_dataloader.dataset.__len__()
    size_val = val_dataloader.dataset.__len__()
    acc_epochs = {'train':[], 'val':[]}
    previous_acc_test = 0
    for epoch in range(n_epochs):
        n_train_correct, n_val_correct = 0, 0
        
        # train
        model.train()
        for xy_batch in train_dataloader:

            x_batch = xy_batch['features'].float().to(device)
            y_batch = xy_batch['label'].long().to(device)#y_batch is of the shape (N,10,1)
            
            y_pred = model(x_batch)#y_pred is of shape (N, C), where N:batch_size, C:# of classes
                        
            loss = loss_fn(y_pred, y_batch)

            y_pred = torch.argmax(y_pred, dim=1)
            
            n_train_correct = n_train_correct + (y_pred==y_batch).sum().item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            for xy_batch in val_dataloader:
                x_batch = xy_batch['features'].float().to(device)
                y_batch = xy_batch['label'].long().to(device)
                y_pred = model(x_batch)
                y_pred = torch.argmax(y_pred, dim=1)
                n_val_correct = n_val_correct + (y_pred==y_batch).sum().item()
        

        train_acc = n_train_correct/size_train
        val_acc = n_val_correct/size_val
        acc_epochs['train'].append(train_acc)
        acc_epochs['val'].append(val_acc)

        print('Epoch({}/{}):train acc:{}, val acc:{}'.format(epoch + 1,n_epochs, train_acc, val_acc))
    
        # update previous_acc_test and save weights
        if val_acc>previous_acc_test:
            previous_acc_test = val_acc
            torch.save(model, model_save_path)

        # early stop
        if val_acc>early_stop:
            return acc_epochs

    return acc_epochs