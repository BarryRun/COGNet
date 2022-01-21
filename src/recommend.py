import pickle
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
from data_loader import mimic_data, pad_num_replace
from beam import Beam

import sys
sys.path.append("..")
from models import Leap, CopyDrug_batch, CopyDrug_tranformer, CopyDrug_generate_prob, CopyDrug_diag_proc_encode
from COGNet_model import CopyModel
from util import llprint, sequence_metric, sequence_metric_v2, sequence_output_process, ddi_rate_score, get_n_params, output_flatten, print_result

torch.manual_seed(1203)

# 读取disease跟proc的英文名
icd_diag_path = '../data/D_ICD_DIAGNOSES.csv'
icd_proc_path = '../data/D_ICD_PROCEDURES.csv'
code2diag = {}
code2proc = {}
with open(icd_diag_path, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        line = line.strip().split(',"')
        if line[-1] == '': line = line[:-1]
        _, icd_code, _, title = line
        code2diag[icd_code[:-1]] = title

with open(icd_proc_path, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        _, icd_code, _, title = line.strip().split(',"')
        code2proc[icd_code[:-1]] = title



def eval_recommend_batch(model, batch_data, device, TOKENS, args):
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS

    diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = batch_data
    # continue
    # 根据vocab对padding数值进行替换
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

    batch_size = medications.size(0)
    max_visit_num = medications.size(1)

    input_disease_embdding, input_proc_embedding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory = model.encode(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, 
        seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20)

    partial_input_medication = torch.full((batch_size, max_visit_num, 1), SOS_TOKEN).to(device)
    parital_logits = None


    for i in range(args.max_len):
        partial_input_med_num = partial_input_medication.size(2)
        partial_m_mask_matrix = torch.zeros((batch_size, max_visit_num, partial_input_med_num), device=device).float()
        # print('val', i, partial_m_mask_matrix.size())

        parital_logits = model.decode(partial_input_medication, input_disease_embdding, input_proc_embedding, encoded_medication, last_seq_medication, cross_visit_scores,
            d_mask_matrix, p_mask_matrix, partial_m_mask_matrix, last_m_mask, drug_memory)
        _, next_medication = torch.topk(parital_logits[:, :, -1, :], 1, dim=-1)
        partial_input_medication = torch.cat([partial_input_medication, next_medication], dim=-1)

    return parital_logits



def test_recommend_batch(model, batch_data, device, TOKENS, ddi_adj, args):
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS

    diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = batch_data
    # continue
    # 根据vocab对padding数值进行替换
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

    batch_size = medications.size(0)
    visit_num = medications.size(1)

    input_disease_embdding, input_proc_embedding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory = model.encode(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, 
        seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20)

    # partial_input_medication = torch.full((batch_size, visit_num, 1), SOS_TOKEN).to(device)
    # parital_logits = None

     # 为每一个样本声明一个beam
    # 这里为了方便实现，写死batch_size必须为1
    assert batch_size == 1
    # visit_num个batch
    beams = [Beam(args.beam_size, MED_PAD_TOKEN, SOS_TOKEN, END_TOKEN, ddi_adj, device) for _ in range(visit_num)]

    # 构建decode输入，每一个visit上需要重复beam_size次数据
    input_disease_embdding = input_disease_embdding.repeat_interleave(args.beam_size, dim=0)
    input_proc_embedding = input_proc_embedding.repeat_interleave(args.beam_size, dim=0)
    encoded_medication = encoded_medication.repeat_interleave(args.beam_size, dim=0)
    last_seq_medication = last_seq_medication.repeat_interleave(args.beam_size, dim=0)
    cross_visit_scores = cross_visit_scores.repeat_interleave(args.beam_size, dim=0)
    # cross_visit_scores = cross_visit_scores.repeat_interleave(args.beam_size, dim=2)
    d_mask_matrix = d_mask_matrix.repeat_interleave(args.beam_size, dim=0)
    p_mask_matrix = p_mask_matrix.repeat_interleave(args.beam_size, dim=0)
    last_m_mask = last_m_mask.repeat_interleave(args.beam_size, dim=0)

    for i in range(args.max_len):
        len_dec_seq = i + 1
        # b.get_current_state(): (beam_size, len_dec_seq) --> (beam_size, 1, len_dec_seq)
        # dec_partial_inputs: (beam_size, visit_num, len_dec_seq)
        dec_partial_inputs = torch.cat([b.get_current_state().unsqueeze(dim=1) for b in beams], dim=1)
        # dec_partial_inputs = dec_partial_inputs.view(args.beam_size, visit_num, len_dec_seq)

        partial_m_mask_matrix = torch.zeros((args.beam_size, visit_num, len_dec_seq), device=device).float()
        # print('val', i, partial_m_mask_matrix.size())

        # parital_logits: (beam_size, visit_sum, len_dec_seq, all_med_num)
        parital_logits = model.decode(dec_partial_inputs, input_disease_embdding, input_proc_embedding, encoded_medication, last_seq_medication, cross_visit_scores,
            d_mask_matrix, p_mask_matrix, partial_m_mask_matrix, last_m_mask, drug_memory)

        # word_lk: (beam_size, visit_sum, all_med_num)
        word_lk = parital_logits[:, :, -1, :]

        active_beam_idx_list = []   # 记录目前仍然active的beam
        for beam_idx in range(visit_num):
            # # 如果当前beam完成了，则跳过，这里beams的size应该是不变的
            # if beams[beam_idx].done: continue
            # inst_idx = beam_inst_idx_map[beam_idx]  # 该beam所对应的adm下标
            # 更新beam，同时返回当前beam是否完成，如果未完成则表示active
            if not beams[beam_idx].advance(word_lk[:, beam_idx, :]):
                active_beam_idx_list.append(beam_idx)

        # 如果没有active的beam，则全部样本预测完毕
        if not active_beam_idx_list: break

    # Return useful information
    all_hyp = []
    all_prob = []
    for beam_idx in range(visit_num):
        scores, tail_idxs = beams[beam_idx].sort_scores()   # 每个beam按照score排序，找出最优的生成
        hyps = beams[beam_idx].get_hypothesis(tail_idxs[0])
        probs = beams[beam_idx].get_prob_list(tail_idxs[0])
        all_hyp += [hyps]    # 注意这里只关注最优解，否则写法上要修改
        all_prob += [probs]

    return all_hyp, all_prob


# evaluate
def eval(model, eval_dataloader, voc_size, epoch, device, TOKENS, args):
    model.eval()
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    smm_record = []
    med_cnt, visit_cnt = 0, 0

    # fw = open("prediction_results.txt", "w")

    for idx, data in enumerate(eval_dataloader):
        diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data
        visit_cnt += seq_length.sum().item()

        output_logits = eval_recommend_batch(model, data, device, TOKENS, args)

        # 每一个med上的预测结果
        labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2], END_TOKEN, device, training=False, testing=False, max_len=args.max_len)

        y_gt = []       # groud truth 表示正确的label   0-1序列
        y_pred = []     # 预测的结果    0-1序列
        y_pred_prob = []    # 预测的每一个药物的平均概率，非0-1序列
        y_pred_label = []   # 预测的结果，非0-1序列
        # 针对每一个admission的预测结果
        for label, prediction in zip(labels, predictions):
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[label] = 1    # 01序列，表示正确的label
            y_gt.append(y_gt_tmp)

            # label: med set
            # prediction: [med_num, probability]
            out_list, sorted_predict = sequence_output_process(prediction, [voc_size[2], voc_size[2]+1])
            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(prediction[:, :-2], axis=0))

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            med_cnt += len(sorted_predict)

            # if idx < 100:
            #     fw.write(print_result(label, sorted_predict))

        smm_record.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(idx, len(eval_dataloader)))

    # fw.close()

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path='../data/ddi_A_final.pkl')

    llprint('\nDDI Rate: {:.4}, Jaccard: {:.4},  PRAUC: {:.4}, AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.4}\n'.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


# test 
def test(model, test_dataloader, diag_voc, pro_voc, med_voc, voc_size, epoch, device, TOKENS, ddi_adj, args):
    model.eval()
    END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN = TOKENS
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt_list = []
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    all_pred_list = []
    all_label_list = []

    ja_by_visit = [[] for _ in range(5)]
    auc_by_visit = [[] for _ in range(5)]
    pre_by_visit = [[] for _ in range(5)]
    recall_by_visit = [[] for _ in range(5)]
    f1_by_visit = [[] for _ in range(5)]
    smm_record_by_visit = [[] for _ in range(5)]

    for idx, data in enumerate(test_dataloader):
        diseases, procedures, medications, seq_length, \
            d_length_matrix, p_length_matrix, m_length_matrix, \
                d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                    dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, \
                        dec_proc, stay_proc, dec_proc_mask, stay_proc_mask = data
        visit_cnt += seq_length.sum().item()

        output_logits, output_probs = test_recommend_batch(model, data, device, TOKENS, ddi_adj, args)

        labels, predictions = output_flatten(medications, output_logits, seq_length, m_length_matrix, voc_size[2], END_TOKEN, device, training=False, testing=True, max_len=args.max_len)
        _, probs = output_flatten(medications, output_probs, seq_length, m_length_matrix, voc_size[2], END_TOKEN, device, training=False, testing=True, max_len=args.max_len)
        y_gt = []       
        y_pred = []    
        y_pred_label = [] 
        y_pred_prob = [] 

        label_hisory = []
        label_hisory_list = []
        pred_list = []
        jaccard_list = []
        def cal_jaccard(set1, set2):
            if not set1 or not set2:
                return 0
            set1 = set(set1)
            set2 = set(set2)
            a, b = len(set1 & set2), len(set1 | set2)
            return a/b
        def cal_overlap_num(set1, set2):
            count = 0
            for d in set1:
                if d in set2:
                    count += 1
            return count

        # 针对每一个admission的预测结果
        for label, prediction, prob_list in zip(labels, predictions, probs):
            label_hisory += label.tolist()  ### case study

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[label] = 1    # 01序列，表示正确的label
            y_gt.append(y_gt_tmp)

            out_list = []
            out_prob_list = []
            for med, prob in zip(prediction, prob_list):
                if med in [voc_size[2], voc_size[2]+1]:
                    break
                out_list.append(med)
                out_prob_list.append(prob[:-2]) # 去掉SOS与EOS符号

            ## case study
            if label_hisory:
                jaccard_list.append(cal_jaccard(prediction, label_hisory))
            pred_list.append(out_list)
            label_hisory_list.append(label.tolist())

            # 对于没预测的药物，取每个位置上平均的概率，否则直接取对应的概率
            # pred_out_prob_list = np.mean(out_prob_list, axis=0)
            pred_out_prob_list = np.max(out_prob_list, axis=0)
            # pred_out_prob_list = np.min(out_prob_list, axis=0)
            for i in range(131):
                if i in out_list:
                    pred_out_prob_list[i] = out_prob_list[out_list.index(i)][i]

            y_pred_prob.append(pred_out_prob_list)
            y_pred_label.append(out_list)

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            med_cnt += len(prediction)
            med_cnt_list.append(len(prediction))


        smm_record.append(y_pred_label)
        for i in range(min(len(labels), 5)):
            # single_ja, single_p, single_r, single_f1 = sequence_metric_v2(np.array(y_gt[i:i+1]), np.array(y_pred[i:i+1]), np.array(y_pred_label[i:i+1]))
            single_ja, single_auc, single_p, single_r, single_f1 = sequence_metric(np.array([y_gt[i]]), np.array([y_pred[i]]), np.array([y_pred_prob[i]]),np.array([y_pred_label[i]]))
            ja_by_visit[i].append(single_ja)
            auc_by_visit[i].append(single_auc)
            pre_by_visit[i].append(single_p)
            recall_by_visit[i].append(single_r)
            f1_by_visit[i].append(single_f1)
            smm_record_by_visit[i].append(y_pred_label[i:i+1])

        # 存储所有预测结果
        all_pred_list.append(pred_list)
        all_label_list.append(labels)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(idx, len(test_dataloader)))

        # 统计不同visit的指标
        if idx%100==0:
            print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
            print('count:', [len(buf) for buf in ja_by_visit])
            print('jaccard:', [np.mean(buf) for buf in ja_by_visit])
            print('auc:', [np.mean(buf) for buf in auc_by_visit])
            print('precision:', [np.mean(buf) for buf in pre_by_visit])
            print('recall:', [np.mean(buf) for buf in recall_by_visit])
            print('f1:', [np.mean(buf) for buf in f1_by_visit])
            print('DDI:', [ddi_rate_score(buf) for buf in smm_record_by_visit])

    print('\tvisit1\tvisit2\tvisit3\tvisit4\tvisit5')
    print('count:', [len(buf) for buf in ja_by_visit])
    print('jaccard:', [np.mean(buf) for buf in ja_by_visit])
    print('auc:', [np.mean(buf) for buf in auc_by_visit])
    print('precision:', [np.mean(buf) for buf in pre_by_visit])
    print('recall:', [np.mean(buf) for buf in recall_by_visit])
    print('f1:', [np.mean(buf) for buf in f1_by_visit])
    print('DDI:', [ddi_rate_score(buf) for buf in smm_record_by_visit])

    pickle.dump(all_pred_list, open('out_list.pkl', 'wb'))
    pickle.dump(all_label_list, open('out_list_gt.pkl', 'wb'))

    return smm_record, ja, prauc, avg_p, avg_r, avg_f1, med_cnt_list
