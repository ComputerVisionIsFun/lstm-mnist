import torch.optim as optim
import parameters as P
import torch


def train(model, n_epochs, optimizer, loss_fn, train_dataloader, test_dataloader)->dict:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 
    model.to(device)
    size_train = train_dataloader.dataset.__len__()
    size_test = test_dataloader.dataset.__len__()
    acc_epochs = {'train':[], 'validation':[]}
    for epoch in range(n_epochs):
        n_train_correct, n_test_correct = 0, 0
        
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
            for xy_batch in test_dataloader:
                x_batch = xy_batch['features'].float().to(device)
                y_batch = xy_batch['label'].long().to(device)
                y_pred = model(x_batch)
                y_pred = torch.argmax(y_pred, dim=1)
                n_test_correct = n_test_correct + (y_pred==y_batch).sum().item()
        
        acc_epochs['train'].append(n_train_correct/size_train)
        acc_epochs['validation'].append(n_test_correct/size_test)

        print('Epoch({}/{}):train acc:{}, test acc:{}'.format(epoch + 1,n_epochs,n_train_correct/size_train, n_test_correct/size_test))
    
        # save weights


        # early stop


    return acc_epochs