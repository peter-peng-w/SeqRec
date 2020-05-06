import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data
from torch.utils.data import Dataset
# import config

def load_data():
    # This one only use the interacion without using the rating
    train_data = pd.read_csv('../data/train/train_data.csv', header=None, 
                    names=['user','item'],usecols=[0,1], dtype={0: np.int32, 1: np.int32})
    # This load the rating of the interaction. Apply simple threshold to generate labels later
    train_rating = pd.read_csv('../data/train/train_data.csv', header=None,
                    names=['rating'],usecols=[2], dtype={2: np.float32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()
    train_rating = train_rating.values.tolist()
    train_label = []

    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for idx, x in enumerate(train_data):
        if train_rating[idx][0] > 3.0:
            train_mat[x[0], x[1]] = 1.0
            train_label.append(1)
        elif train_rating[idx][0] > 1.1:
            train_mat[x[0], x[1]] = 0.5
            train_label.append(0)
        else:
            train_mat[x[0], x[1]] = 0.0
            train_label.append(0)
        
    print('Finished generate the train_mat')
    ## NOTE: Currently we don't use this kind of way to add the test data
    # This data is being added to measure some metric like RMSE.

    # test_data = pd.read_csv('./test/test_data.csv', header=None,
    #                 names=['user','item'], usecols=[0,1], dtype={0: np.int32, 1: np.int32})
    # test_rating = pd.read_csv('./test/test_data.csv', header=None,
    #                 names=['rating'],usecols=[2], dtype=np.float32)

    # test_data = test_data.values.tolist()
    # test_rating = test_rating.values.tolist()
    # test_label = []
    test_data = []
    test_rating = []
    test_label = []

    # for rate in test_rating:
    #     if rate > 2.1:
    #         test_label.append(1)
    #     else:
    #         test_label.append(0)

    # NOTE: [UPDATE] Add negative testing. Use this as test set.
    test_with_neg = []
    with open('../data/test/test_data_with_neg.csv', 'r') as f_1:
        line = f_1.readline()
        while line != None and line != '':
            datas = line.strip('\n').split(',')
            cur_user = int(datas[0])
            for item in datas[1:]:
                test_with_neg.append([cur_user, int(item)])
            line = f_1.readline()

    print('Finished read the test_with_neg data.')
    return train_data, test_data, train_rating, test_rating, train_label, test_label, test_with_neg, user_num, item_num, train_mat

class NCFDataSet(Dataset):
    def __init__(self, tuple_data, neg_train_num_per_tuple, 
            train_mat, num_item, train_label, test_label, is_training=True):
        super(NCFDataSet, self).__init__()
        self.tuple_feature_data = tuple_data
        # Used to insert negative tuples into the train set.
        self.neg_train_num_per_tuple = neg_train_num_per_tuple
        self.train_mat = train_mat
        self.num_item = num_item
        self.is_training = is_training
        # The label of the test set.
        # NOTE: [UPDATE] Add negative items into the test set.
        self.test_label = test_label
        self.test_label_with_neg = [0 for _ in range(len(self.tuple_feature_data))]
        self.test_label_with_neg[0] = 1
        # The initial label of the train set when we didn't add the negative data.
        self.train_label_pre = train_label 

    def insert_neg_for_train(self):
        # This is only needed to be done before training
        self.neg_train_tuple = []
        for x in self.tuple_feature_data:
            user = x[0]
            for neg_num_iter in range(self.neg_train_num_per_tuple):
                select_neg_item = np.random.randint(self.num_item)
                while (user, select_neg_item) in self.train_mat:
                    select_neg_item = np.random.randint(self.num_item)
                self.neg_train_tuple.append([user, select_neg_item])

        labels_init = self.train_label_pre
        labels_neg = [0 for _ in range(len(self.neg_train_tuple))]

        self.total_feature = self.tuple_feature_data + self.neg_train_tuple
        self.total_label = labels_init + labels_neg

    def __getitem__(self, index):
        if self.is_training:
            features = self.total_feature
            labels = self.total_label
        else:
            features = self.tuple_feature_data
            labels = self.test_label_with_neg
        user = features[index][0]
        item = features[index][1]
        label = labels[index]
        return user, item, label

    def __len__(self):
        if self.is_training:
            return (self.neg_train_num_per_tuple + 1) * len(self.train_label_pre)
        else:
            return len(self.test_label_with_neg)
