from sklearn import preprocessing
from torch.nn.functional import pad
from torch.utils import data
import torch
import random


class mimic_data(data.Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def pad_batch(batch):
    seq_length = torch.tensor([len(data) for data in batch])
    batch_size = len(batch)
    max_seq = max(seq_length)

    # 统计每一个seq疾病、手术、药物的数量，以及相应的最值
    # 同时为每一个seq的disease计算与上一个seq的disease之间的交集和差集
    d_length_matrix = []
    p_length_matrix = []
    m_length_matrix = []
    d_max_num = 0
    p_max_num = 0
    m_max_num = 0
    d_dec_list = []
    d_stay_list = []
    for data in batch:
        d_buf, p_buf, m_buf = [], [], []
        d_dec_list_buf, d_stay_list_buf = [], []
        for idx, seq in enumerate(data):
            d_buf.append(len(seq[0]))
            p_buf.append(len(seq[1]))
            m_buf.append(len(seq[2]))
            d_max_num = max(d_max_num, len(seq[0]))
            p_max_num = max(p_max_num, len(seq[1]))
            m_max_num = max(m_max_num, len(seq[2]))
            if idx==0:
                # 第一个seq，则交集与差集为空
                d_dec_list_buf.append([])
                d_stay_list_buf.append([])
            else:
                # 计算差集与交集
                cur_d = set(seq[0])
                last_d = set(data[idx-1][0])
                stay_list = list(cur_d & last_d)
                dec_list = list(last_d - cur_d)
                d_dec_list_buf.append(dec_list)
                d_stay_list_buf.append(stay_list)
        d_length_matrix.append(d_buf)
        p_length_matrix.append(p_buf)
        m_length_matrix.append(m_buf)
        d_dec_list.append(d_dec_list_buf)
        d_stay_list.append(d_stay_list_buf)

    # 生成m_mask_matrix
    m_mask_matrix = torch.full((batch_size, max_seq, m_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(m_length_matrix[i])):
            m_mask_matrix[i, j, :m_length_matrix[i][j]] = 0.

    # 生成d_mask_matrix
    d_mask_matrix = torch.full((batch_size, max_seq, d_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(d_length_matrix[i])):
            d_mask_matrix[i, j, :d_length_matrix[i][j]] = 0.

    # 生成p_mask_matrix
    p_mask_matrix = torch.full((batch_size, max_seq, p_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(p_length_matrix[i])):
            p_mask_matrix[i, j, :p_length_matrix[i][j]] = 0.

    # 分别生成dec_disease_tensor和stay_disease_tensor
    dec_disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1)
    stay_disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1)
    dec_disease_mask = torch.full((batch_size, max_seq, d_max_num), -1e9)
    stay_disease_mask = torch.full((batch_size, max_seq, d_max_num), -1e9)
    for b_id, (dec_seqs, stay_seqs) in enumerate(zip(d_dec_list, d_stay_list)):
        for s_id, (dec_adm, stay_adm) in enumerate(zip(dec_seqs, stay_seqs)):
            dec_disease_tensor[b_id, s_id, :len(dec_adm)] = torch.tensor(dec_adm)
            stay_disease_tensor[b_id, s_id, :len(stay_adm)] = torch.tensor(stay_adm)
            dec_disease_mask[b_id, s_id, :len(dec_adm)] = 0.
            stay_disease_mask[b_id, s_id, :len(dec_adm)] = 0.

    # 分别生成disease、procedure、medication的数据
    disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1)
    procedure_tensor = torch.full((batch_size, max_seq, p_max_num), -1)
    medication_tensor = torch.full((batch_size, max_seq, m_max_num), 0)

    # 分别拼接成一个batch的数据
    for b_id, data in enumerate(batch):
        for s_id, adm in enumerate(data):
            # adm部分的数据按照disease、procedure、medication排序
            disease_tensor[b_id, s_id, :len(adm[0])] = torch.tensor(adm[0])
            procedure_tensor[b_id, s_id, :len(adm[1])] = torch.tensor(adm[1])
            medication_tensor[b_id, s_id, :len(adm[2])] = torch.tensor(adm[2])
    # print(disease_tensor[1])
    return disease_tensor, procedure_tensor, medication_tensor, seq_length, \
        d_length_matrix, p_length_matrix, m_length_matrix, \
            d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                dec_disease_tensor, stay_disease_tensor, dec_disease_mask, stay_disease_mask


def pad_batch_v2_train(batch):
    seq_length = torch.tensor([len(data) for data in batch])
    batch_size = len(batch)
    max_seq = max(seq_length)

    # 统计每一个seq疾病、手术、药物的数量，以及相应的最值
    # 同时为每一个seq的disease计算与上一个seq的disease之间的交集和差集
    d_length_matrix = []
    p_length_matrix = []
    m_length_matrix = []
    d_max_num = 0
    p_max_num = 0
    m_max_num = 0
    d_dec_list = []
    d_stay_list = []
    p_dec_list = []
    p_stay_list = []
    for data in batch:
        d_buf, p_buf, m_buf = [], [], []
        d_dec_list_buf, d_stay_list_buf = [], []
        p_dec_list_buf, p_stay_list_buf = [], []
        for idx, seq in enumerate(data):
            d_buf.append(len(seq[0]))
            p_buf.append(len(seq[1]))
            m_buf.append(len(seq[2]))
            d_max_num = max(d_max_num, len(seq[0]))
            p_max_num = max(p_max_num, len(seq[1]))
            m_max_num = max(m_max_num, len(seq[2]))
            if idx==0:
                # 第一个seq，则交集与差集为空
                d_dec_list_buf.append([])
                d_stay_list_buf.append([])
                p_dec_list_buf.append([])
                p_stay_list_buf.append([])
            else:
                # 计算差集与交集
                cur_d = set(seq[0])
                last_d = set(data[idx-1][0])
                stay_list = list(cur_d & last_d)
                dec_list = list(last_d - cur_d)
                d_dec_list_buf.append(dec_list)
                d_stay_list_buf.append(stay_list)

                cur_p = set(seq[1])
                last_p = set(data[idx-1][1])
                proc_stay_list = list(cur_p & last_p)
                proc_dec_list = list(last_p - cur_p)
                p_dec_list_buf.append(proc_dec_list)
                p_stay_list_buf.append(proc_stay_list)
        d_length_matrix.append(d_buf)
        p_length_matrix.append(p_buf)
        m_length_matrix.append(m_buf)
        d_dec_list.append(d_dec_list_buf)
        d_stay_list.append(d_stay_list_buf)
        p_dec_list.append(p_dec_list_buf)
        p_stay_list.append(p_stay_list_buf)

    # 生成m_mask_matrix
    m_mask_matrix = torch.full((batch_size, max_seq, m_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(m_length_matrix[i])):
            m_mask_matrix[i, j, :m_length_matrix[i][j]] = 0.

    # 生成d_mask_matrix
    d_mask_matrix = torch.full((batch_size, max_seq, d_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(d_length_matrix[i])):
            d_mask_matrix[i, j, :d_length_matrix[i][j]] = 0.

    # 生成p_mask_matrix
    p_mask_matrix = torch.full((batch_size, max_seq, p_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(p_length_matrix[i])):
            p_mask_matrix[i, j, :p_length_matrix[i][j]] = 0.

    # 分别生成dec_disease_tensor和stay_disease_tensor
    dec_disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1)
    stay_disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1)
    dec_disease_mask = torch.full((batch_size, max_seq, d_max_num), -1e9)
    stay_disease_mask = torch.full((batch_size, max_seq, d_max_num), -1e9)
    for b_id, (dec_seqs, stay_seqs) in enumerate(zip(d_dec_list, d_stay_list)):
        for s_id, (dec_adm, stay_adm) in enumerate(zip(dec_seqs, stay_seqs)):
            dec_disease_tensor[b_id, s_id, :len(dec_adm)] = torch.tensor(dec_adm)
            stay_disease_tensor[b_id, s_id, :len(stay_adm)] = torch.tensor(stay_adm)
            dec_disease_mask[b_id, s_id, :len(dec_adm)] = 0.
            stay_disease_mask[b_id, s_id, :len(dec_adm)] = 0.

    # 分别生成dec_disease_tensor和stay_disease_tensor
    dec_proc_tensor = torch.full((batch_size, max_seq, p_max_num), -1)
    stay_proc_tensor = torch.full((batch_size, max_seq, p_max_num), -1)
    dec_proc_mask = torch.full((batch_size, max_seq, p_max_num), -1e9)
    stay_proc_mask = torch.full((batch_size, max_seq, p_max_num), -1e9)
    for b_id, (dec_seqs, stay_seqs) in enumerate(zip(p_dec_list, p_stay_list)):
        for s_id, (dec_adm, stay_adm) in enumerate(zip(dec_seqs, stay_seqs)):
            dec_proc_tensor[b_id, s_id, :len(dec_adm)] = torch.tensor(dec_adm)
            stay_proc_tensor[b_id, s_id, :len(stay_adm)] = torch.tensor(stay_adm)
            dec_proc_mask[b_id, s_id, :len(dec_adm)] = 0.
            stay_proc_mask[b_id, s_id, :len(dec_adm)] = 0.

    # 分别生成disease、procedure、medication的数据
    disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1)
    procedure_tensor = torch.full((batch_size, max_seq, p_max_num), -1)
    medication_tensor = torch.full((batch_size, max_seq, m_max_num), 0)

    # 分别拼接成一个batch的数据
    for b_id, data in enumerate(batch):
        for s_id, adm in enumerate(data):
            # adm部分的数据按照disease、procedure、medication排序
            disease_tensor[b_id, s_id, :len(adm[0])] = torch.tensor(adm[0])
            procedure_tensor[b_id, s_id, :len(adm[1])] = torch.tensor(adm[1])
            # dynamic shuffle
            # cur_medications = adm[2]
            # random.shuffle(cur_medications)
            # medication_tensor[b_id, s_id, :len(adm[2])] = torch.tensor(cur_medications)
            medication_tensor[b_id, s_id, :len(adm[2])] = torch.tensor(adm[2])

    # print(disease_tensor[1])
    return disease_tensor, procedure_tensor, medication_tensor, seq_length, \
        d_length_matrix, p_length_matrix, m_length_matrix, \
            d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                dec_disease_tensor, stay_disease_tensor, dec_disease_mask, stay_disease_mask, \
                    dec_proc_tensor, stay_proc_tensor, dec_proc_mask, stay_proc_mask



def pad_batch_v2_eval(batch):
    seq_length = torch.tensor([len(data) for data in batch])
    batch_size = len(batch)
    max_seq = max(seq_length)

    # 统计每一个seq疾病、手术、药物的数量，以及相应的最值
    # 同时为每一个seq的disease计算与上一个seq的disease之间的交集和差集
    d_length_matrix = []
    p_length_matrix = []
    m_length_matrix = []
    d_max_num = 0
    p_max_num = 0
    m_max_num = 0
    d_dec_list = []
    d_stay_list = []
    p_dec_list = []
    p_stay_list = []
    for data in batch:
        d_buf, p_buf, m_buf = [], [], []
        d_dec_list_buf, d_stay_list_buf = [], []
        p_dec_list_buf, p_stay_list_buf = [], []
        for idx, seq in enumerate(data):
            d_buf.append(len(seq[0]))
            p_buf.append(len(seq[1]))
            m_buf.append(len(seq[2]))
            d_max_num = max(d_max_num, len(seq[0]))
            p_max_num = max(p_max_num, len(seq[1]))
            m_max_num = max(m_max_num, len(seq[2]))
            if idx==0:
                # 第一个seq，则交集与差集为空
                d_dec_list_buf.append([])
                d_stay_list_buf.append([])
                p_dec_list_buf.append([])
                p_stay_list_buf.append([])
            else:
                # 计算差集与交集
                cur_d = set(seq[0])
                last_d = set(data[idx-1][0])
                stay_list = list(cur_d & last_d)
                dec_list = list(last_d - cur_d)
                d_dec_list_buf.append(dec_list)
                d_stay_list_buf.append(stay_list)

                cur_p = set(seq[1])
                last_p = set(data[idx-1][1])
                proc_stay_list = list(cur_p & last_p)
                proc_dec_list = list(last_p - cur_p)
                p_dec_list_buf.append(proc_dec_list)
                p_stay_list_buf.append(proc_stay_list)
        d_length_matrix.append(d_buf)
        p_length_matrix.append(p_buf)
        m_length_matrix.append(m_buf)
        d_dec_list.append(d_dec_list_buf)
        d_stay_list.append(d_stay_list_buf)
        p_dec_list.append(p_dec_list_buf)
        p_stay_list.append(p_stay_list_buf)

    # 生成m_mask_matrix
    m_mask_matrix = torch.full((batch_size, max_seq, m_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(m_length_matrix[i])):
            m_mask_matrix[i, j, :m_length_matrix[i][j]] = 0.

    # 生成d_mask_matrix
    d_mask_matrix = torch.full((batch_size, max_seq, d_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(d_length_matrix[i])):
            d_mask_matrix[i, j, :d_length_matrix[i][j]] = 0.

    # 生成p_mask_matrix
    p_mask_matrix = torch.full((batch_size, max_seq, p_max_num), -1e9)
    for i in range(batch_size):
        for j in range(len(p_length_matrix[i])):
            p_mask_matrix[i, j, :p_length_matrix[i][j]] = 0.

    # 分别生成dec_disease_tensor和stay_disease_tensor
    dec_disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1)
    stay_disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1)
    dec_disease_mask = torch.full((batch_size, max_seq, d_max_num), -1e9)
    stay_disease_mask = torch.full((batch_size, max_seq, d_max_num), -1e9)
    for b_id, (dec_seqs, stay_seqs) in enumerate(zip(d_dec_list, d_stay_list)):
        for s_id, (dec_adm, stay_adm) in enumerate(zip(dec_seqs, stay_seqs)):
            dec_disease_tensor[b_id, s_id, :len(dec_adm)] = torch.tensor(dec_adm)
            stay_disease_tensor[b_id, s_id, :len(stay_adm)] = torch.tensor(stay_adm)
            dec_disease_mask[b_id, s_id, :len(dec_adm)] = 0.
            stay_disease_mask[b_id, s_id, :len(dec_adm)] = 0.

    # 分别生成dec_disease_tensor和stay_disease_tensor
    dec_proc_tensor = torch.full((batch_size, max_seq, p_max_num), -1)
    stay_proc_tensor = torch.full((batch_size, max_seq, p_max_num), -1)
    dec_proc_mask = torch.full((batch_size, max_seq, p_max_num), -1e9)
    stay_proc_mask = torch.full((batch_size, max_seq, p_max_num), -1e9)
    for b_id, (dec_seqs, stay_seqs) in enumerate(zip(p_dec_list, p_stay_list)):
        for s_id, (dec_adm, stay_adm) in enumerate(zip(dec_seqs, stay_seqs)):
            dec_proc_tensor[b_id, s_id, :len(dec_adm)] = torch.tensor(dec_adm)
            stay_proc_tensor[b_id, s_id, :len(stay_adm)] = torch.tensor(stay_adm)
            dec_proc_mask[b_id, s_id, :len(dec_adm)] = 0.
            stay_proc_mask[b_id, s_id, :len(dec_adm)] = 0.

    # 分别生成disease、procedure、medication的数据
    disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1)
    procedure_tensor = torch.full((batch_size, max_seq, p_max_num), -1)
    medication_tensor = torch.full((batch_size, max_seq, m_max_num), 0)

    # 分别拼接成一个batch的数据
    for b_id, data in enumerate(batch):
        for s_id, adm in enumerate(data):
            # adm部分的数据按照disease、procedure、medication排序
            disease_tensor[b_id, s_id, :len(adm[0])] = torch.tensor(adm[0])
            procedure_tensor[b_id, s_id, :len(adm[1])] = torch.tensor(adm[1])
            medication_tensor[b_id, s_id, :len(adm[2])] = torch.tensor(adm[2])

    # print(disease_tensor[1])
    return disease_tensor, procedure_tensor, medication_tensor, seq_length, \
        d_length_matrix, p_length_matrix, m_length_matrix, \
            d_mask_matrix, p_mask_matrix, m_mask_matrix, \
                dec_disease_tensor, stay_disease_tensor, dec_disease_mask, stay_disease_mask, \
                    dec_proc_tensor, stay_proc_tensor, dec_proc_mask, stay_proc_mask


def pad_num_replace(tensor, src_num, target_num):
    # replace_tensor = torch.full_like(tensor, target_num)
    return torch.where(tensor==src_num, target_num, tensor)
    

