
import torch
import torch.nn.functional as F


from utilstools.utils2 import accuracy

def trains_multiview(model,optimizer,labels,idx_train,lamda,eita):
    model.train()
    optimizer.zero_grad()

    z, loss1,loss2 = model()

    loss_xy = 0.25*F.nll_loss(z[idx_train], labels[idx_train]) - lamda*(torch.stack(loss1).mean())- eita*loss2

    loss_xy.backward()

    optimizer.step()
    return loss_xy.item()

def tests_multiview(model,y, idx_test,final=False,final_predictions = None):
    model.eval()
    z,_,_ = model()
    acc_test = accuracy(z[idx_test], y[idx_test])
    return acc_test
