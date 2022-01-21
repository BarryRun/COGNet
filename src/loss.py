import torch
import torch.nn as nn
import torch.nn.functional as F


class cross_entropy_loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, labels, logits, seq_length, m_length_matrix, med_num, END_TOKEN):
        # labels: [batch_size, max_seq_length, max_med_num]
        # logits: [batch_size, max_seq_length, max_med_num, med_num]
        # m_length_matrix: [batch_size, seq_length]
        # seq_length: [batch_size]

        batch_size, max_seq_length = labels.size()[:2]
        assert max_seq_length == max(seq_length)
        whole_seqs_num = seq_length.sum().item()
        whole_med_sum = sum([sum(buf) for buf in m_length_matrix]) + whole_seqs_num 

        labels_flatten = torch.empty(whole_med_sum).to(self.device)
        logits_flatten = torch.empty(whole_med_sum, med_num).to(self.device)

        start_idx = 0
        for i in range(batch_size): 
            for j in range(seq_length[i]):  
                for k in range(m_length_matrix[i][j]+1):  
                    if k==m_length_matrix[i][j]:
                        labels_flatten[start_idx] = END_TOKEN
                    else:
                        labels_flatten[start_idx] = labels[i, j, k]
                    logits_flatten[start_idx, :] = logits[i, j, k, :]
                    start_idx += 1


        loss = F.cross_entropy(logits_flatten, labels_flatten.long())
        return loss



