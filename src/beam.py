""" Manage beam search info structure.
    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

from shutil import which
from util import ddi_rate_score
import torch
import numpy as np
import copy
import random
from torch.autograd.grad_mode import F


class Beam(object):
    ''' Store the necessary info for beam search '''
    def __init__(self, size, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, ddi_adj, device):
        self.ddi_adj = ddi_adj
        self.PAD = PAD_TOKEN
        self.BOS = BOS_TOKEN
        self.EOS = EOS_TOKEN
        # print(PAD_TOKEN, EOS_TOKEN, BOS_TOKEN)

        self.device = device
        self.size = size
        self.done = False   # 表示当前是否已经完成了beam search的过程

        self.beam_status = [False] * size   # 用于记录每一个beam是否已经处于EOS状态

        self.tt = torch.cuda if device.type=='cuda' else torch

        # 每一个生成结果的分数，初始是beam_size个0
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size).fill_(self.BOS)]
        self.prob_list = []


    def get_current_state(self, sort=True):
        "Get the outputs for the current timestep."
        if sort:
            return self.get_tentative_hypothesis()
        else:
            return self.get_tentative_hypothesis_wo_sort()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_lk):
        "Update the status and check for finished or not."
        num_words = word_lk.size(1)
        if self.done:
            self.prev_ks.append(torch.tensor(list(range(self.size)), device=self.device)) 
            self.next_ys.append(torch.tensor([self.EOS]*self.size, device=self.device))
            self.prob_list.append([[0]*num_words, [0]*num_words])
            return True

        active_beam_idx = torch.tensor([idx for idx in range(self.size) if self.beam_status[idx]==False]).long().to(self.device)
        end_beam_idx = torch.tensor([idx for idx in range(self.size) if self.beam_status[idx]==True]).long().to(self.device)
        active_word_lk = word_lk[active_beam_idx]   # active_beam_num * num_words

        cur_output = self.get_current_state(sort=False) 

        active_scores = self.scores[active_beam_idx]
        end_scores = self.scores[end_beam_idx]

        if len(self.prev_ks) > 0:
            beam_lk = active_word_lk + active_scores.unsqueeze(dim=1).expand_as(active_word_lk)   # (active_beam_num, num_words)
        else:
            beam_lk = active_word_lk[0]

        flat_beam_lk = beam_lk.view(-1)
        active_max_idx = len(flat_beam_lk)
        flat_beam_lk = torch.cat([flat_beam_lk, end_scores], dim=-1)

        self.all_scores.append(self.scores)


        sorted_scores, sorted_score_ids = torch.sort(flat_beam_lk, descending=True)
        select_num, cur_idx = 0, 0
        selected_scores = []
        selected_words = []
        selected_beams = []
        new_active_status = []

        prob_buf = []
        while select_num < self.size:
            cur_score, cur_id = sorted_scores[cur_idx], sorted_score_ids[cur_idx]
            if cur_id >= active_max_idx:
                which_beam = end_beam_idx[cur_id-active_max_idx]
                which_word = torch.tensor(self.EOS).to(self.device)
                select_num += 1
                new_active_status.append(True)
                selected_scores.append(cur_score)
                selected_beams.append(which_beam)
                selected_words.append(which_word)
                prob_buf.append([0]*num_words)
            else:
                which_beam_idx = cur_id // num_words
                which_beam = active_beam_idx[which_beam_idx]
                which_word = cur_id - which_beam_idx*num_words
                if which_word not in cur_output[which_beam]:
                    if which_word in [self.EOS, self.BOS]:
                        new_active_status.append(True)
                    else:
                        new_active_status.append(False)
                    select_num += 1
                    selected_scores.append(cur_score)
                    selected_beams.append(which_beam)
                    selected_words.append(which_word)
                    prob_buf.append(active_word_lk[which_beam_idx].detach().cpu().numpy().tolist())
            cur_idx += 1
        self.prob_list.append(prob_buf)

        self.beam_status = new_active_status    
        self.scores = torch.stack(selected_scores)
        self.prev_ks.append(torch.stack(selected_beams))   
        self.next_ys.append(torch.stack(selected_words))   
        
        if_done = True
        for i in range(self.size):
            if not self.beam_status[i]:
                if_done = False
                break
        if if_done: 
            self.done=True
        
        return self.done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)


    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()    
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.BOS] + h for h in hyps]
            dec_seq = torch.from_numpy(np.array(hyps)).long().to(self.device)
        return dec_seq

    def get_tentative_hypothesis_wo_sort(self):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            keys = list(range(self.size))
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.BOS] + h for h in hyps]
            dec_seq = torch.from_numpy(np.array(hyps)).long().to(self.device)
        return dec_seq

    def get_hypothesis(self, k):
        """
        Walk back to construct the full hypothesis.
        Parameters.
             * `k` - the position in the beam to construct.
         Returns.
            1. The hypothesis
            2. The attention at each time step.
        """
        hyp = []
        for j in range(len(self.prev_ks)-1, -1, -1):
            hyp.append(self.next_ys[j + 1][k].item())
            k = self.prev_ks[j][k]
        return hyp[::-1]
    
    def get_prob_list(self, k):
        ret_prob_list = []
        for j in range(len(self.prev_ks)-1, -1, -1):
            ret_prob_list.append(self.prob_list[j][k]) 
            k = self.prev_ks[j][k]
        return ret_prob_list[::-1]