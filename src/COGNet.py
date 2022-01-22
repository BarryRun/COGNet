import collections
import torch
import torch.nn as nn
import argparse
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils import data
from loss import cross_entropy_loss
import os
import torch.nn.functional as F
import random
from collections import defaultdict

from torch.utils.data.dataloader import DataLoader
from data_loader import mimic_data, pad_batch_v2_train, pad_batch_v2_eval, pad_num_replace

import sys
sys.path.append("..")
from COGNet_ablation import COGNet_wo_copy, COGNet_wo_visit_score, COGNet_wo_graph, COGNet_wo_diag, COGNet_wo_proc
from COGNet_model import COGNet, policy_network
from util import llprint, sequence_metric, sequence_output_process, ddi_rate_score, get_n_params, output_flatten, print_result
from recommend import eval, test

torch.manual_seed(1203)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = 'COGNet'
resume_path = ''

if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
# parser.add_argument('--Test', action='store_true', default=True, help="test mode")
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--emb_dim', type=int, default=64, help='embedding dimension size')
parser.add_argument('--max_len', type=int, default=45, help='maximum prediction medication sequence')
parser.add_argument('--beam_size', type=int, default=4, help='max num of sentences in beam searching')

args = parser.parse_args()

def main(args):
    # load data
    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'

    # ehr_adj_path = '../data/weighted_ehr_adj_final.pkl'
    ehr_adj_path = '../data/ehr_adj_final.pkl'
    ddi_adj_path = '../data/ddi_A_final.pkl'
    ddi_mask_path = '../data/ddi_mask_H.pkl'
    device = torch.device('cuda')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    print(f"Diag num:{len(diag_voc.idx2word)}")
    print(f"Proc num:{len(pro_voc.idx2word)}")
    print(f"Med num:{len(med_voc.idx2word)}")

    # frequency statistic
    med_count = defaultdict(int)
    for patient in data:
        for adm in patient:
            for med in adm[2]:
                med_count[med] += 1
    
    ## rare first
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur_medications = sorted(data[i][j][2], key=lambda x:med_count[x])
            data[i][j][2] = cur_medications


    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    train_dataset = mimic_data(data_train)
    eval_dataset = mimic_data(data_eval)
    test_dataset = mimic_data(data_test)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_batch_v2_train, shuffle=True, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=pad_batch_v2_eval, shuffle=True, pin_memory=True)
    
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]

    model = COGNet(voc_size, ehr_adj, ddi_adj, ddi_mask_H, emb_dim=args.emb_dim, device=device)

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        smm_record, ja, prauc, precision, recall, f1, med_num = test(model, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, 0, device, TOKENS, ddi_adj, args)
        result = []
        for _ in range(10):
            data_num = len(ja)
            final_length = int(0.8 * data_num)
            idx_list = list(range(data_num))
            random.shuffle(idx_list)
            idx_list = idx_list[:final_length]
            avg_ja = np.mean([ja[i] for i in idx_list])
            avg_prauc = np.mean([prauc[i] for i in idx_list])
            avg_precision = np.mean([precision[i] for i in idx_list])
            avg_recall = np.mean([recall[i] for i in idx_list])
            avg_f1 = np.mean([f1[i] for i in idx_list])
            avg_med = np.mean([med_num[i] for i in idx_list])
            cur_smm_record = [smm_record[i] for i in idx_list]
            ddi_rate = ddi_rate_score(cur_smm_record, path='../data/ddi_A_final.pkl')
            result.append([ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med])
            llprint('\nDDI Rate: {:.4}, Jaccard: {:.4}, PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
                    ddi_rate, avg_ja, avg_prauc, avg_precision, avg_recall, avg_f1, avg_med))
        result = np.array(result)
        mean = result.mean(axis=0)
        std = result.std(axis=0)

        outstring = ""
        for m, s in zip(mean, std):
            outstring += "{:.4f} $\pm$ {:.4f} & ".format(m, s)

        print (outstring)
        print ('test time: {}'.format(time.time() - tic))
        return 

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 200
    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch))

        model.train()
        for idx, data in enumerate(train_dataloader):
            diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data

            diseases = pad_num_replace(diseases, -1, DIAG_PAD_TOKEN).to(device)
            procedures = pad_num_replace(procedures, -1, PROC_PAD_TOKEN).to(device)
            dec_disease = pad_num_replace(dec_disease, -1, DIAG_PAD_TOKEN).to(device)
            stay_disease = pad_num_replace(stay_disease, -1, DIAG_PAD_TOKEN).to(device)
            dec_proc = pad_num_replace(dec_proc, -1, PROC_PAD_TOKEN).to(device)
            stay_proc = pad_num_replace(stay_proc, -1, PROC_PAD_TOKEN).to(device)
            medications = medications.to(device)
            m_mask_matrix = m_mask_matrix.to(device)
            d_mask_matrix = d_mask_matrix.to(device)
            p_mask_matrix = p_mask_matrix.to(device)
            dec_disease_mask = dec_disease_mask.to(device)
            stay_disease_mask = stay_disease_mask.to(device)
            dec_proc_mask = dec_proc_mask.to(device)
            stay_proc_mask = stay_proc_mask.to(device)
            output_logits = model(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask,
                dec_proc, stay_proc, dec_proc_mask, stay_proc_mask)
            labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2] + 2, END_TOKEN, device, max_len=args.max_len)
            loss = F.nll_loss(predictions, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            llprint('\rtraining step: {} / {}'.format(idx, len(train_dataloader)))

        print ()
        tic2 = time.time()
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, eval_dataloader, voc_size, epoch, device, TOKENS, args)
        print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print ('ddi: {}, Med: {}, Ja: {}, F1: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
                ))

        torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, \
            'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, ja, ddi_rate)), 'wb'))

        if best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print ('best_epoch: {}'.format(best_epoch))

        dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))


if __name__ == '__main__':
    main(args)
