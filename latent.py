from data_gen import *

import sys
sys.path.append('./LatentTrees') # replace with the actual path to the LatentTrees package
from LatentTrees.src.LT_models import LTBinaryClassifier
from LatentTrees.src.monitors import MonitorTree
from LatentTrees.src.optimization import train_batch


def trainLatentTree(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels, w_list_old=None, b_list_old=None):
    train_loader = get_train_loader(train_data, train_data_labels, batch_size=32)
    valid_loader = get_train_loader(vali_data, vali_data_labels, batch_size=32)
    test_loader = get_train_loader(test_data, test_data_labels, batch_size=32)

    model = LTBinaryClassifier(5, 2, reg=0.001, linear=True, split_func='linear', comp_func='concatenate')
    criterion = torch.nn.BCELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    monitor = MonitorTree(True, 'monitor/')

    train_batch(train_data, train_data_labels, model, optimizer, criterion, monitor=monitor)    

    def test_LatentTree(model, data_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                output = model(data)
                output = output > 0.5
                target = target.numpy()
                output = output.numpy().flatten()
                correct += np.sum(output == target)
                total += len(target)
        return correct / total

    valid_acc = test_LatentTree(model, valid_loader)
    test_acc = test_LatentTree(model, test_loader)
    print(valid_acc, test_acc)
    torch.save(model, "best_lt.pt")