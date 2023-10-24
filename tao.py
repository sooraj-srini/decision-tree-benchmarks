from data_gen import *

from taodecisiontree import TaoTreeClassifier

def trainTAO(train_data, train_data_labels, vali_data, vali_data_labels, test_data, test_data_labels, w_list_old=None, b_list_old=None):

    m = TaoTreeClassifier(randomize_tree=False, weight_errors=False,
                            node_model='linear', model_args={'max_depth': 5},
                            verbose=1, n_iters=40)

    m.fit(train_data, train_data_labels)
    print('Train acc', np.mean(m.predict(train_data) == train_data_labels))
    print('Test acc', np.mean(m.predict(test_data) == test_data_labels))