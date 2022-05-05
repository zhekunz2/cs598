"""
Patient2Vec: a self-attentive representation learning framework
"""
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
import random
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class Patient2Vec(nn.Module):
    """
    Self-attentive representation learning framework,
    including convolutional embedding layer,
    recurrent autoencoder with an encoder, recurrent module, and a decoder.
    In addition, a linear layer is on top of each decode step and the weights are shared at these step.
    """

    def __init__(self, input_size, hidden_size, n_layers, att_dim, initrange,
                 output_size, rnn_type, seq_len, pad_size, n_filters, bi, dropout_p=0.5):
        """
        Initilize a recurrent model
        :param input_size: int
        :param hidden_size: int
        :param n_layers: number of layers; int
        :param att_dim: dimension of the attention; int
        :param initrange: upper bound of the initial weights; symmetric
        :param output_size: int
        :param rnn_type: str, such as 'GRU'
        :param seq_len: length of the sequence; int
        :param pad_size: padding size; int
        :param n_filters: number of hops; int
        :param bi: bidirectional; bool
        :param dropout_p: dropout rate; float
        """
        super(Patient2Vec, self).__init__()

        self.initrange = initrange
        # convolution
        self.b = 1
        if bi:
            self.b = 2

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=input_size, stride=2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=hidden_size * self.b, stride=2)
        # Bidirectional RNN
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, n_layers, dropout=dropout_p,
                                         batch_first=True, bias=True, bidirectional=bi)
        # initialize 2-layer attention weight matrics
        self.att_w1 = nn.Linear(hidden_size * self.b, att_dim, bias=False)
        # final linear layer
        self.linear = nn.Linear(hidden_size * self.b * n_filters, output_size, bias=True)

        self.func_softmax = nn.Softmax()
        self.func_sigmoid = nn.Sigmoid()
        self.func_tanh = nn.Hardtanh(0, 1)
        # Add dropout
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.init_weights()

        self.pad_size = pad_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.n_filters = n_filters

    def init_weights(self):
        """
        weight initialization
        """
        for param in self.parameters():
            param.data.uniform_(-self.initrange, self.initrange)

    def convolutional_layer(self, inputs):
        convolution_all = []
        conv_wts = []
        for i in range(self.seq_len):
            convolution_one_month = []
            for j in range(self.pad_size):
                # batch, 1, num_features
                convolution = self.conv(torch.unsqueeze(inputs[:, i, j], dim=1))
                # batch, 1, 1
                convolution_one_month.append(convolution)
            convolution_one_month = torch.stack(convolution_one_month, dim=2)
            # batch, 1, pad
            convolution_one_month = torch.squeeze(convolution_one_month, dim=3)
            # convolution_one_month = torch.transpose(convolution_one_month, 0, 1)
            # convolution_one_month = torch.transpose(convolution_one_month, 1, 2)
            convolution_one_month = self.func_tanh(convolution_one_month)
            # convolution_one_month = torch.unsqueeze(convolution_one_month, dim=1)
            # batch, 1, pad * batch, pad, num_features
            vec = torch.bmm(convolution_one_month, inputs[:, i])
            convolution_all.append(vec)
            conv_wts.append(convolution_one_month)
        convolution_all = torch.stack(convolution_all, dim=1)
        # batch, 4, num_features
        convolution_all = torch.squeeze(convolution_all, dim=2)
        conv_wts = torch.squeeze(torch.stack(conv_wts, dim=1), dim=2)
        return convolution_all, conv_wts

    def encode_rnn(self, embedding, batch_size):
        self.weight = next(self.parameters()).data
        init_state = (Variable(self.weight.new(self.n_layers * self.b, batch_size, self.hidden_size).zero_()))
        embedding = self.dropout(embedding)
        outputs_rnn, states_rnn = self.rnn(embedding, init_state)
        return outputs_rnn

    def add_beta_attention(self, states, batch_size):
        # beta attention
        att_wts = []
        for i in range(self.seq_len):
            m1 = self.conv2(torch.unsqueeze(states[:, i], dim=1))
            att_wts.append(torch.squeeze(m1, dim=2))
        att_wts = torch.stack(att_wts, dim=2)
        att_beta = []
        for i in range(self.n_filters):
            a0 = F.softmax(att_wts[:, i], dim=-1)
            att_beta.append(a0)
        att_beta = torch.stack(att_beta, dim=1)
        context = torch.bmm(att_beta, states)
        context = context.view(batch_size, -1)
        return att_beta, context

    def forward(self, inputs, batch_size):
        """
        the recurrent module
        """
        # Convolutional
        convolutions, alpha = self.convolutional_layer(inputs)
        # RNN
        states_rnn = self.encode_rnn(convolutions, batch_size)
        # Add attentions and get context vector
        beta, context = self.add_beta_attention(states_rnn, batch_size)
        # Final linear layer with demographic info added as extra variables
        linear_y = self.linear(context)
        out = F.softmax(linear_y, dim=-1)
        return out, alpha, beta


def get_loss(pred, y, criterion, mtr, a=0.5):
    """
    To calculate loss
    :param pred: predicted value
    :param y: actual value
    :param criterion: nn.CrossEntropyLoss
    :param mtr: beta matrix
    """
    mtr_t = torch.transpose(mtr, 1, 2)
    aa = torch.bmm(mtr, mtr_t)
    loss_fn = 0
    for i in range(aa.size()[0]):
        aai = torch.add(aa[i, ], Variable(torch.neg(torch.eye(mtr.size()[1]))))
        loss_fn += torch.trace(torch.mul(aai, aai).data)
    loss_fn /= aa.size()[0]
    loss = torch.add(criterion(pred, y), Variable(torch.FloatTensor([loss_fn * a])))
    return loss

def compare_dates(x, y):
    if x[0] != y[0]:
        return x[0] > y[0]
    elif x[1] != y[1]:
        return x[1] > y[1]
    elif x[2] != y[2]:
        return x[2] > y[2]
    return True

def prepare_data():
        #  ============== Prepare Data ===========================
    DATA_PATH = './mimic-iii-clinical-database-1.4/'

    admissions = pd.read_csv(os.path.join(DATA_PATH,'ADMISSIONS.csv'), header=0, usecols = ["SUBJECT_ID","HADM_ID","ADMITTIME"], squeeze=True)
    drug_codes = pd.read_csv(os.path.join(DATA_PATH,'DRGCODES.csv'), header=0, usecols = ["SUBJECT_ID","HADM_ID","DRG_CODE"], squeeze=True)
    drug_codes.DRG_CODE = drug_codes.DRG_CODE.astype(str)
    drug_codes = drug_codes.groupby(["SUBJECT_ID","HADM_ID"], as_index=False).agg({'DRG_CODE': ' '.join})

    diagnosis_codes = pd.read_csv(os.path.join(DATA_PATH,'DIAGNOSES_ICD.csv'), header=0, usecols = ["SUBJECT_ID","HADM_ID","ICD9_CODE"], squeeze=True)
    diagnosis_codes.ICD9_CODE = diagnosis_codes.ICD9_CODE.astype(str)
    diagnosis_codes = diagnosis_codes.groupby(["SUBJECT_ID","HADM_ID"], as_index=False).agg({'ICD9_CODE': ' '.join})
    diagnosis_codes = diagnosis_codes.rename(columns={"ICD9_CODE": "DIAG_CODES"})

    procedure_codes = pd.read_csv(os.path.join(DATA_PATH,'PROCEDURES_ICD.csv'), header=0, usecols = ["SUBJECT_ID","HADM_ID","ICD9_CODE"], squeeze=True)
    procedure_codes.ICD9_CODE = procedure_codes.ICD9_CODE.astype(str)
    procedure_codes = procedure_codes.groupby(["SUBJECT_ID","HADM_ID"], as_index=False).agg({'ICD9_CODE': ' '.join})
    procedure_codes = procedure_codes.rename(columns={"ICD9_CODE": "PROC_CODES"})

    final_table = admissions.merge(drug_codes[["SUBJECT_ID","HADM_ID","DRG_CODE"]])
    final_table = final_table.merge(diagnosis_codes[["SUBJECT_ID","HADM_ID","DIAG_CODES"]])
    final_table = final_table.merge(procedure_codes[["SUBJECT_ID","HADM_ID","PROC_CODES"]])
    print(final_table)

    final_table = final_table.values.tolist()
    final_table = final_table[:10000]

    # convert DIAG_CODES/PROC_CODES/DRG_CODES to list of int
    diag_map = {}
    proc_map = {}
    drg_map = {}
    unique_drg_id = 0
    unique_diag_id = 0
    unique_proc_id = 0

    current_visit = 0
    current_patient = -1
    max_visit = 0
    for admission in final_table:
        # SUBJECT_ID  HADM_ID  ADMITTIME  DRG_CODE  DIAG_CODES   PROC_CODES
        if admission[0] == current_patient:
            current_visit += 1
        else:
            max_visit = max(max_visit, current_visit)
            current_patient = admission[0]
            current_visit = 1
        DRG_CODES = admission[3].split(' ')
        DIAG_CODES = admission[4].split(' ')
        PROC_CODES = admission[5].split(' ')
        
        new_drg_codes = []
        new_diag_codes = []
        new_proc_codes = []

        for i in DRG_CODES:
            new_drg_codes.append(int(i))
            if int(i) not in drg_map.keys():
                drg_map[int(i)] = unique_drg_id
                unique_drg_id += 1

        for i in DIAG_CODES:
            if i not in diag_map.keys():
                diag_map[i] = unique_diag_id
                unique_diag_id += 1
            new_diag_codes.append(int(diag_map[i]))

        for i in PROC_CODES:
            if i not in proc_map.keys():
                proc_map[i] = unique_proc_id
                unique_proc_id += 1
            new_proc_codes.append(int(proc_map[i]))
        
        admission[3] = new_drg_codes
        admission[4] = new_diag_codes
        admission[5] = new_proc_codes

    # mark patient labels
    patient_map = {} # ID: [dates]
    for admission in final_table:
        date = admission[2].split(' ')[0]
        date = date.split('-')
        int_date = []
        for i in date:
            int_date.append(int(i))
        if admission[0] in patient_map.keys():
            patient_map[admission[0]].append(int_date)
        else:
            patient_map[admission[0]] = [int_date]


    patient_labels = {}
    patient_earliest_date_map = {}
    id_map = {}
    unique_id = 0
    for i in patient_map.keys():
        if len(patient_map[i]) >= 2:
            oldest_date = patient_map[i][0]
            newest_date = patient_map[i][0]
            for date in patient_map[i]:
                if compare_dates(date, oldest_date):
                    oldest_date = date
                if not compare_dates(date, newest_date):
                    newest_date = date
            label = oldest_date[0] > newest_date[0] + 1 or (oldest_date[0] > newest_date[0] and oldest_date[1] >= newest_date[1])
            patient_earliest_date_map[i] = newest_date
            patient_labels[i] = label
        else:
            patient_labels[i] = False
            patient_earliest_date_map[i] = patient_map[i][0]
        id_map[i] = unique_id
        unique_id+=1

    # constructing input tensor
    n_drg = len(drg_map.keys())
    n_diag = len(diag_map.keys())
    n_proc = len(proc_map.keys())
    print(n_drg, n_diag, n_proc)
    # 1026 3925 1260
    input_tensor = torch.zeros(unique_id+1, 4, 10, n_drg + n_diag + n_proc)
    visit_idx = [0, 0, 0, 0]
    current_patient = -1
    for admission in final_table:
        # SUBJECT_ID  HADM_ID  ADMITTIME  DRG_CODE  DIAG_CODES   PROC_CODES

        first_date = patient_earliest_date_map[admission[0]]
        date = admission[2].split(' ')[0]
        date = date.split('-')
        ad_date = []
        for i in date:
            ad_date.append(int(i))
        
        seq = 0
        if ad_date[0] == first_date[0]:
            # same year
            seq = int((ad_date[1] - first_date[1]) / 3)
        elif ad_date[0] == first_date[0] + 1 and ad_date[1] < first_date[1]:
            seq = int((ad_date[1] - first_date[1] + 12) / 3)
        else:
            continue
        
        if not admission[0] == current_patient:
            visit_idx = [0, 0, 0, 0]
            current_patient = admission[0]
    
        for drg in admission[3]:
            input_tensor[id_map[admission[0]]][seq][visit_idx[seq]][drg_map[drg]] = 1
        for diag in admission[4]:
            input_tensor[id_map[admission[0]]][seq][visit_idx[seq]][n_drg + diag] = 1
        for proc in admission[5]:
            input_tensor[id_map[admission[0]]][seq][visit_idx[seq]][n_drg + n_diag + proc] = 1
        visit_idx[seq] += 1

    y_tensor = torch.zeros(unique_id+1, dtype=torch.long)
    for i in patient_labels.keys():
        y_tensor[id_map[i]] = patient_labels[i]
    
    # input_tensor.shape [42210, 4, 23, xxx] y_tensor.shape = [42210]
    train_size = int(y_tensor.shape[0] / 5) * 4
    train_x = input_tensor[0:train_size, :, :]
    train_y = y_tensor[0:train_size]

    test_x = input_tensor[train_size:, :, :]
    test_y = y_tensor[train_size:]

    torch.save(train_x, 'train_x.pt')
    torch.save(train_y, 'train_y.pt')
    torch.save(test_x, 'test_x.pt')
    torch.save(test_y, 'test_y.pt')

def eval_model(model, test_x, test_y):
    y_pred = torch.LongTensor()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    model.eval()
    n_test_samples = test_y.shape[0]
    for i in range(n_test_samples):
        y_hat, _, _ = model(torch.unsqueeze(test_x[i], dim=0), 1)
        y_hat = torch.unsqueeze(y_hat[0][1], dim=0)
        y_score = torch.cat((y_score,  y_hat.detach().to('cpu')), dim=0)
        y_hat = (y_hat > 0.5).long()
        y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, torch.unsqueeze(test_y[i], dim=0).detach().to('cpu')), dim=0)

    print('gold positive', torch.sum(test_y))
    print('predict positive', torch.sum(y_pred))

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    roc_auc = roc_auc_score(y_true, y_score)
    return p, r, f, roc_auc

if __name__ == '__main__':
    # prepare_data()
    train_x = torch.load('train_x.pt')
    # train_x = torch.flatten(train_x, start_dim=1)
    train_y = torch.load('train_y.pt')

    test_x = torch.load('test_x.pt')
    # test_x = torch.flatten(test_x, start_dim=1)
    test_y = torch.load('test_y.pt')

    # input_size, hidden_size, n_layers, att_dim, initrange, output_size, rnn_type, seq_len, pad_size, n_filters, bi
    model = Patient2Vec(6211, 256, 1, 0, 1, 2, 'GRU', 4, 10, 3, False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  

    # Train
    n_epochs = 25
    batch_size = 6336
    n_train_samples = test_y.shape[0]
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        loss = None
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs, _, _ = model(train_x, batch_size)         
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        # print statistics
        train_loss += loss.item()
        train_loss = train_loss
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
        if epoch % 25 == 0:
            p, r, f, roc_auc = eval_model(model, test_x, test_y)
            print('Epoch: {} \t Validation p: {:.2f}, r:{:.2f}, f: {:.2f}, roc_auc: {:.2f}'.format(epoch+1, p, r, f, roc_auc))
