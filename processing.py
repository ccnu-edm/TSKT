import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import build_dense_graph
import csv
import pdb

class KTDataset(Dataset):
    def __init__(self, features, questions, answers, t1_list, t2_list, t3_list):
        super(KTDataset, self).__init__()
        self.features = features
        self.questions = questions
        self.answers = answers
        self.t1_list = t1_list
        self.t2_list = t2_list
        self.t3_list = t3_list

    def __getitem__(self, index):
        return self.features[index], self.questions[index], self.answers[index], self.t1_list[index], self.t2_list[index], self.t3_list[index]

    def __len__(self):
        return len(self.features)


def pad_collate(batch):
    (features, questions, answers,t1, t2, t3) = zip(*batch)
    features = [torch.LongTensor(feat) for feat in features]
    questions = [torch.LongTensor(qt) for qt in questions]
    answers = [torch.LongTensor(ans) for ans in answers]
    t1 = [torch.LongTensor(tt1) for tt1 in t1]
    t2 = [torch.LongTensor(tt2) for tt2 in t2]
    t3 = [torch.LongTensor(tt3) for tt3 in t3]
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)
    t1_pad = pad_sequence(t1, batch_first=True, padding_value=0)
    t2_pad = pad_sequence(t2, batch_first=True, padding_value=0)
    t3_pad = pad_sequence(t3, batch_first=True, padding_value=0)
    return feature_pad, question_pad, answer_pad, t1_pad, t2_pad, t3_pad


def load_dataset(file_path, batch_size, graph_type, dkt_graph_path=None, train_ratio=0.7, val_ratio=0.2, shuffle=True, model_type='GKT', use_binary=True, use_cuda=True,max_qnum=275):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        batch_size: the size of a student batch
        graph_type: the type of the concept graph
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
    Return:
        concept_num: the number of all concepts(or questions)
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT'], otherwise graph is None
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """
    '''
    df = pd.read_csv(file_path)
    if "skill_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {file_path}")
    if "correct" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {file_path}")
    if "user_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {file_path}")

    # if not (df['correct'].isin([0, 1])).all():
    #     raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

    # Step 1.1 - Remove questions without skill
    df.dropna(subset=['skill_id'], inplace=True)

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)  # we can also use problem_id to represent exercises

    # Step 3 - Cross skill id with answer to form a synthetic feature
    if use_binary:
        df['skill_with_answer'] = df['skill'] * 2 + df['correct']
    else:
        df['skill_with_answer'] = df['skill'] * 12 + df['correct'] - 1


    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []

    def get_data(series):
        feature_list.append(series['skill_with_answer'].tolist())
        question_list.append(series['skill'].tolist())
        answer_list.append(series['correct'].eq(1).astype('int').tolist())
        seq_len_list.append(series['correct'].shape[0])

    df.groupby('user_id').apply(get_data)
    max_seq_len = np.max(seq_len_list)
    print('max seq_len: ', max_seq_len)
    student_num = len(seq_len_list)
    print('student num: ', student_num)
    feature_dim = int(df['skill_with_answer'].max() + 1)
    print('feature_dim: ', feature_dim)
    question_dim = int(df['skill'].max() + 1)
    print('question_dim: ', question_dim)
    concept_num = question_dim
    # 记一下结果有几个，是2个（0或1）还是12个（1,2,3,4,5,6,7,8,9,10,11,12）
    reslen = 2 if use_binary else 12
    assert feature_dim == reslen * question_dim
    '''
    # pdb.set_trace()
    length = 200
    q_num = max_qnum
    concept_num = max_qnum
    rows = csv.reader(open(file_path, 'r'), delimiter=',')
    rows = [[int(e) for e in row if e != ''] for row in rows]
    q_rows, r_rows = [], []
    feature_list = []
    question_list = []
    answer_list = []
    t1_list = []
    t2_list = []
    t3_list = []
    seq_len_list = []
    for q_row, r_row, t1_row, t2_row, t3_row in zip(rows[1::6], rows[2::6], rows[3::6], rows[4::6], rows[5::6]):
        num = len(q_row)
        n = num // length
        for i in range(n + 1):
            question_list.append(q_row[i * length: (i + 1) * length])
            answer_list.append(r_row[i * length: (i + 1) * length])
            t1_list.append(t1_row[i * length: (i + 1) * length])
            t2_list.append(t2_row[i * length: (i + 1) * length])
            t3_list.append(t3_row[i * length: (i + 1) * length])
            seq_len_list.append(len(r_row[i * length: (i + 1) * length]))
            feature_list.append([(q + q_num * (1 - r)) for q,r in zip(q_row[i * length: (i + 1) * length],r_row[i * length: (i + 1) * length])])
    question_list = [row for row in question_list if len(row) > 2]
    feature_list = [row for row in feature_list if len(row) > 2]
    answer_list = [row for row in answer_list if len(row) > 2]
    t1_list = [row for row in t1_list if len(row) > 2]
    t2_list = [row for row in t2_list if len(row) > 2]
    t3_list = [row for row in t3_list if len(row) > 2]
    seq_len_list = [row for row in seq_len_list if row > 2]
    student_num = len(seq_len_list)


    kt_dataset = KTDataset(feature_list, question_list, answer_list, t1_list, t2_list, t3_list)
    train_size = int(train_ratio * student_num)
    val_size = int(val_ratio * student_num)
    test_size = student_num - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(kt_dataset, [train_size, val_size, test_size])
    print('train_size: ', train_size, 'val_size: ', val_size, 'test_size: ', test_size)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    valid_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate)

    graph = None
    if model_type == 'GKT':
        if graph_type == 'Dense':
            graph = build_dense_graph(concept_num)
        elif graph_type == 'Transition':
            graph = build_transition_graph(question_list, seq_len_list, train_dataset.indices, student_num, concept_num)
        elif graph_type == 'DKT':
            graph = build_dkt_graph(dkt_graph_path, concept_num)
        if use_cuda and graph != None:
            graph = graph.cuda()
    return concept_num, graph, train_data_loader, valid_data_loader, test_data_loader


def build_transition_graph(question_list, seq_len_list, indices, student_num, concept_num):
    graph = np.zeros((concept_num, concept_num))
    student_dict = dict(zip(indices, np.arange(student_num)))
    for i in range(student_num):
        if i not in student_dict:
            continue
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            next = questions[j + 1]
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))
    def inv(x):
        if x == 0:
            return x
        return 1. / x
    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    # covert to tensor
    graph = torch.from_numpy(graph).float()
    return graph


def build_dkt_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    graph = torch.from_numpy(graph).float()
    return graph