from data_gen import *

from train_utils import train, test
from network import Net 
from torch.optim.lr_scheduler import StepLR
import torch

def trainLCN(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels, w_list_old=None, b_list_old=None):

    model = Net(input_dim=input_dim, output_dim=2, hidden_dim=1, num_layer=12, num_back_layer=0, dense=True, drop_type='none', net_type='locally_constant', approx='interpolation').to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    train_loader = get_train_loader(train_data, train_data_labels, batch_size=32)
    valid_loader = get_train_loader(vali_data, vali_data_labels, batch_size=32)
    test_loader = get_train_loader(test_data, test_data_labels, batch_size=32)

    best_score = 100000
    for epoch in range(0, 10):
            scheduler.step(epoch)

            train_approximate_loss = train(model, device, train_loader, optimizer, epoch, 'none',1)
            # used for plotting learning curves
            train_loss, train_score = test(model, device, train_loader, 'train')
            valid_loss, valid_score = test(model, device, valid_loader, 'valid')
            test_loss, test_score = test(model, device, test_loader, 'test')
            
            # early stopping version
            if valid_score > best_score:
                state = {'model': model.state_dict()}
                torch.save(state, "best_lcn.pt")
                best_score = valid_score

            # "convergent" version
            state = {'model': model.state_dict()}
            torch.save(state, "last_lcn.pt")
            print(train_loss, train_score, valid_loss, valid_score, test_loss, test_score)


