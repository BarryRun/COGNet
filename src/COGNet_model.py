import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.modules.linear import Linear
from data.processing import process_visit_lg2

from layers import SelfAttend
from layers import GraphConvolution


class COGNet(nn.Module):
    """在CopyDrug_batch基础上将medication的encode部分修改为transformer encoder"""
    def __init__(self, voc_size, ehr_adj, ddi_adj, ddi_mask_H, emb_dim=64, device=torch.device('cpu:0')):
        super(COGNet, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        self.nhead = 2
        self.SOS_TOKEN = voc_size[2]        # start of sentence
        self.END_TOKEN = voc_size[2]+1      # end   新增的两个编码，两者均是针对于药物的embedding
        self.MED_PAD_TOKEN = voc_size[2]+2      # 用于embedding矩阵中的padding（全为0）
        self.DIAG_PAD_TOKEN = voc_size[0]+2
        self.PROC_PAD_TOKEN = voc_size[1]+2

        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

        # dig_num * emb_dim
        self.diag_embedding = nn.Sequential(
            nn.Embedding(voc_size[0]+3, emb_dim, self.DIAG_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # proc_num * emb_dim
        self.proc_embedding = nn.Sequential(
            nn.Embedding(voc_size[1]+3, emb_dim, self.PROC_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # med_num * emb_dim
        self.med_embedding = nn.Sequential(
            # 添加padding_idx，表示取0向量
            nn.Embedding(voc_size[2]+3, emb_dim, self.MED_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # 用于对上一个visit的medication进行编码
        # self.medication_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, dim_feedforward=emb_dim*8, batch_first=True, dropout=0.2)
        self.medication_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)
        # 用于对当前visit的疾病与症状进行编码
        # self.diagnoses_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, dim_feedforward=emb_dim*8, batch_first=True, dropout=0.2)
        # self.procedure_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, dim_feedforward=emb_dim*8, batch_first=True, dropout=0.2)        
        self.diagnoses_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)
        self.procedure_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)
        # self.enc_gru = nn.GRU(emb_dim, emb_dim, batch_first=True, bidirectional=True)

        # self.ehr_gcn = GCN(
        #     voc_size=voc_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        # self.ddi_gcn = GCN(
        #     voc_size=voc_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.gcn =  GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        # 聚合单个visit内的diag和proc得到visit-level的表达
        self.diag_self_attend = SelfAttend(emb_dim)
        self.proc_self_attend = SelfAttend(emb_dim)

        self.decoder = MedTransformerDecoder(emb_dim, self.nhead, dim_feedforward=emb_dim*2, dropout=0.2, 
                 layer_norm_eps=1e-5)

        # 用于对每一个visit的diagnoses进行编码

        # 用于生成药物序列
        self.dec_gru = nn.GRU(emb_dim*3, emb_dim, batch_first=True)

        self.diag_attn = nn.Linear(emb_dim*2, 1)
        self.proc_attn = nn.Linear(emb_dim*2, 1)
        self.W_diag_attn = nn.Linear(emb_dim, emb_dim)
        self.W_proc_attn = nn.Linear(emb_dim, emb_dim)
        self.W_diff_attn = nn.Linear(emb_dim, emb_dim)
        self.W_diff_proc_attn = nn.Linear(emb_dim, emb_dim)

        # weights
        self.Ws = nn.Linear(emb_dim*2, emb_dim)  # only used at initial stage
        self.Wo = nn.Linear(emb_dim, voc_size[2]+2)  # generate mode
        # self.Wc = nn.Linear(emb_dim*2, emb_dim)  # copy mode
        self.Wc = nn.Linear(emb_dim, emb_dim)  # copy mode

        self.W_dec = nn.Linear(emb_dim, emb_dim)
        self.W_stay = nn.Linear(emb_dim, emb_dim)
        self.W_proc_dec = nn.Linear(emb_dim, emb_dim)
        self.W_proc_stay = nn.Linear(emb_dim, emb_dim)

        # swtich network to calculate generate probablity
        self.W_z = nn.Linear(emb_dim, 1)


        self.weight = nn.Parameter(torch.tensor([0.3]), requires_grad=True)
        # bipartite local embedding
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(
            ddi_mask_H.shape[1], voc_size[2], False)

    def encode(self, diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20):
        device = self.device
        # batch维度以及seq维度上并行计算（现在不考虑时间序列信息），每一个medication序列仍然按顺序预测
        batch_size, max_visit_num, max_med_num = medications.size()
        max_diag_num = diseases.size()[2]
        max_proc_num = procedures.size()[2]
        
        ############################ 数据预处理 #########################
        # 1. 对当前的disease与procedure进行编码
        input_disease_embdding = self.diag_embedding(diseases).view(batch_size * max_visit_num, max_diag_num, self.emb_dim)      # [batch, seq, max_diag_num, emb]
        input_proc_embedding = self.proc_embedding(procedures).view(batch_size * max_visit_num, max_proc_num, self.emb_dim)      # [batch, seq, max_proc_num, emb]
        d_enc_mask_matrix = d_mask_matrix.view(batch_size * max_visit_num, max_diag_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_diag_num,1) # [batch*seq, nhead, input_length, output_length]
        d_enc_mask_matrix = d_enc_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_diag_num, max_diag_num)
        p_enc_mask_matrix = p_mask_matrix.view(batch_size * max_visit_num, max_proc_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_proc_num,1)
        p_enc_mask_matrix = p_enc_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_proc_num, max_proc_num)
        input_disease_embdding = self.diagnoses_encoder(input_disease_embdding, src_mask=d_enc_mask_matrix).view(batch_size, max_visit_num, max_diag_num, self.emb_dim)
        input_proc_embedding = self.procedure_encoder(input_proc_embedding, src_mask=p_enc_mask_matrix).view(batch_size, max_visit_num, max_proc_num, self.emb_dim)

        # 1.1 encode visit-level diag and proc representations
        visit_diag_embedding = self.diag_self_attend(input_disease_embdding.view(batch_size * max_visit_num, max_diag_num, -1), d_mask_matrix.view(batch_size * max_visit_num, -1))
        visit_proc_embedding = self.proc_self_attend(input_proc_embedding.view(batch_size * max_visit_num, max_proc_num, -1), p_mask_matrix.view(batch_size * max_visit_num, -1))
        visit_diag_embedding = visit_diag_embedding.view(batch_size, max_visit_num, -1)
        visit_proc_embedding = visit_proc_embedding.view(batch_size, max_visit_num, -1)

        # 1.3 计算 visit-level的attention score
        # [batch_size, max_visit_num, max_visit_num]
        cross_visit_scores = self.calc_cross_visit_scores(visit_diag_embedding, visit_proc_embedding)
        

        # 3. 构造一个last_seq_medication，表示上一次visit的medication，第一次的由于没有上一次medication，用0填补（用啥填补都行，反正不会用到）
        last_seq_medication = torch.full((batch_size, 1, max_med_num), 0).to(device)
        last_seq_medication = torch.cat([last_seq_medication, medications[:, :-1, :]], dim=1)
        # m_mask_matrix矩阵同样也需要后移
        last_m_mask = torch.full((batch_size, 1, max_med_num), -1e9).to(device) # 这里用较大负值，避免softmax之后分走了概率
        last_m_mask = torch.cat([last_m_mask, m_mask_matrix[:, :-1, :]], dim=1)
        # 对last_seq_medication进行编码
        last_seq_medication_emb = self.med_embedding(last_seq_medication)
        last_m_enc_mask = last_m_mask.view(batch_size * max_visit_num, max_med_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1,self.nhead,max_med_num,1)
        last_m_enc_mask = last_m_enc_mask.view(batch_size * max_visit_num * self.nhead, max_med_num, max_med_num)
        encoded_medication = self.medication_encoder(last_seq_medication_emb.view(batch_size * max_visit_num, max_med_num, self.emb_dim), src_mask=last_m_enc_mask) # (batch*seq, max_med_num, emb_dim)
        encoded_medication = encoded_medication.view(batch_size, max_visit_num, max_med_num, self.emb_dim)

        # vocab_size, emb_size
        ehr_embedding, ddi_embedding = self.gcn()
        drug_memory = ehr_embedding - ddi_embedding * self.inter
        drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float()
        drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)

        return input_disease_embdding, input_proc_embedding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory

    def decode(self, input_medications, input_disease_embedding, input_proc_embedding, last_medication_embedding, last_medications, cross_visit_scores,
        d_mask_matrix, p_mask_matrix, m_mask_matrix, last_m_mask, drug_memory):
        """
        input_medications: [batch_size, max_visit_num, max_med_num + 1], 开头包含了 SOS_TOKEN
        """
        batch_size = input_medications.size(0)
        max_visit_num = input_medications.size(1)
        max_med_num = input_medications.size(2)
        max_diag_num = input_disease_embedding.size(2)
        max_proc_num = input_proc_embedding.size(2)

        input_medication_embs = self.med_embedding(input_medications).view(batch_size * max_visit_num, max_med_num, -1)
        # input_medication_embs = self.dropout_emb(input_medication_embs)
        input_medication_memory = drug_memory[input_medications].view(batch_size * max_visit_num, max_med_num, -1)

        # m_sos_mask = torch.zeros((batch_size, max_visit_num, 1), device=self.device).float() # 这里用较大负值，避免softmax之后分走了概率
        m_self_mask = m_mask_matrix

        last_m_enc_mask = m_self_mask.view(batch_size * max_visit_num, max_med_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_med_num, 1)
        medication_self_mask = last_m_enc_mask.view(batch_size * max_visit_num * self.nhead, max_med_num, max_med_num)
        m2d_mask_matrix = d_mask_matrix.view(batch_size * max_visit_num, max_diag_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_med_num, 1)
        m2d_mask_matrix = m2d_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_med_num, max_diag_num)
        m2p_mask_matrix = p_mask_matrix.view(batch_size * max_visit_num, max_proc_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_med_num,1)
        m2p_mask_matrix = m2p_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_med_num, max_proc_num)

        dec_hidden = self.decoder(input_medication_embedding=input_medication_embs, input_medication_memory=input_medication_memory,
            input_disease_embdding=input_disease_embedding.view(batch_size * max_visit_num, max_diag_num, -1), 
            input_proc_embedding=input_proc_embedding.view(batch_size * max_visit_num, max_proc_num, -1), 
            input_medication_self_mask=medication_self_mask, 
            d_mask=m2d_mask_matrix, 
            p_mask=m2p_mask_matrix)

        score_g = self.Wo(dec_hidden) # (batch * max_visit_num, max_med_num, voc_size[2]+2)
        score_g = score_g.view(batch_size, max_visit_num, max_med_num, -1)
        prob_g = F.softmax(score_g, dim=-1)
        score_c = self.copy_med(dec_hidden.view(batch_size, max_visit_num, max_med_num, -1), last_medication_embedding, last_m_mask, cross_visit_scores)
        # (batch_size, max_visit_num * input_med_num, max_visit_num * max_med_num)
        
        ###### case study
        # 这里前提是batch_size等于1
        # 几个取值的说明：
        #   1.取最新生成的药物对于历史药物的attention值，所以第三维度为-1
        #   2.取第最后一个visit的copy值，所以第二维度为-1
        #   3.取最后一个visit对倒数第二个visit的药物的attention值，所以第四维度取最后max_med_num个
        # score_c_buf = score_c.view(batch_size, max_visit_num, max_med_num, -1)
        # score_c_buf = score_c_buf[0, -1, -1, :] # visit_num * (visit_num * max_med_num)
        # max_med_num_in_last = len(score_c_buf) // max_visit_num
        # print(score_c_buf[-max_med_num_in_last:])
        prob_c_to_g = torch.zeros_like(prob_g).to(self.device).view(batch_size, max_visit_num * max_med_num, -1) # (batch, max_visit_num * input_med_num, voc_size[2]+2)

        # 用scatter操作代替嵌套循环
        # 根据last_seq_medication中的indice，将score_c中的值加到score_c_to_g中去
        copy_source = last_medications.view(batch_size, 1, -1).repeat(1, max_visit_num * max_med_num, 1)
        prob_c_to_g.scatter_add_(2, copy_source, score_c)
        prob_c_to_g = prob_c_to_g.view(batch_size, max_visit_num, max_med_num, -1)

        generate_prob = F.sigmoid(self.W_z(dec_hidden)).view(batch_size, max_visit_num, max_med_num, 1)
        prob =  prob_g * generate_prob + prob_c_to_g * (1. - generate_prob)
        prob[:, 0, :, :] = prob_g[:, 0, :, :] # 第一个seq由于没有last_medication信息，仅取prob_g的概率

        return torch.log(prob)

    # def forward(self, input, last_input=None, max_len=20):
    def forward(self, diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20):
        device = self.device
        # batch维度以及seq维度上并行计算（现在不考虑时间序列信息），每一个medication序列仍然按顺序预测
        batch_size, max_seq_length, max_med_num = medications.size()
        max_diag_num = diseases.size()[2]
        max_proc_num = procedures.size()[2]
        
        input_disease_embdding, input_proc_embedding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory = self.encode(diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, 
            seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=20)

        # 4. 构造给decoder的medications，用于decoding过程中的teacher forcing，注意维度上增加了一维，因为会多生成一个END_TOKEN
        input_medication = torch.full((batch_size, max_seq_length, 1), self.SOS_TOKEN).to(device)    # [batch_size, seq, 1]
        input_medication = torch.cat([input_medication, medications], dim=2)      # [batch_size, seq, max_med_num + 1]

        m_sos_mask = torch.zeros((batch_size, max_seq_length, 1), device=self.device).float() # 这里用较大负值，避免softmax之后分走了概率
        m_mask_matrix = torch.cat([m_sos_mask, m_mask_matrix], dim=-1)

        output_logits = self.decode(input_medication, input_disease_embdding, input_proc_embedding, encoded_medication, last_seq_medication, cross_visit_scores,
            d_mask_matrix, p_mask_matrix, m_mask_matrix, last_m_mask, drug_memory)

        # 5. 加入ddi loss
        # output_logits_part = torch.exp(output_logits[:, :, :, :-2] + m_mask_matrix.unsqueeze(-1))    # 去掉SOS与EOS
        # output_logits_part = torch.mean(output_logits_part, dim=2)
        # neg_pred_prob1 = output_logits_part.unsqueeze(-1)
        # neg_pred_prob2 = output_logits_part.unsqueeze(-2)
        # neg_pred_prob = neg_pred_prob1 * neg_pred_prob2 # bach * seq * max_med_num * all_med_num * all_med_num
        # batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        # return output_logits, batch_neg
        return output_logits

    def calc_cross_visit_scores(self, visit_diag_embedding, visit_proc_embedding):
        """
        visit_diag_embedding: (batch * visit_num * emb)
        visit_proc_embedding: (batch * visit_num * emb)
        """
        max_visit_num = visit_diag_embedding.size(1)
        batch_size = visit_diag_embedding.size(0)

        # mask表示每个visit只能看到自己之前的visit
        mask = (torch.triu(torch.ones((max_visit_num, max_visit_num), device=self.device)) == 1).transpose(0, 1)    # 下三角矩阵
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)   # batch * max_visit_num * max_visit_num

        # 每个visit后移一位
        padding = torch.zeros((batch_size, 1, self.emb_dim), device=self.device).float()
        diag_keys = torch.cat([padding, visit_diag_embedding[:, :-1, :]], dim=1)    # batch * max_visit_num * emb
        proc_keys = torch.cat([padding, visit_proc_embedding[:, :-1, :]], dim=1)

        # 得到每个visit跟自己前面所有visit的score
        diag_scores = torch.matmul(visit_diag_embedding, diag_keys.transpose(-2, -1)) \
                 / math.sqrt(visit_diag_embedding.size(-1))
        proc_scores = torch.matmul(visit_proc_embedding, proc_keys.transpose(-2, -1)) \
                 / math.sqrt(visit_proc_embedding.size(-1))
        # 1st visit's scores is not zero!
        scores = F.softmax(diag_scores + proc_scores + mask, dim=-1)

        ###### case study
        # 将第0个val置0，然后重新归一化
        # scores_buf = scores
        # scores_buf[:, :, 0] = 0.
        # scores_buf = scores_buf / torch.sum(scores_buf, dim=2, keepdim=True)

        # print(scores_buf)
        return scores

    def copy_med(self, decode_input_hiddens, last_medications, last_m_mask, cross_visit_scores):
        """
        decode_input_hiddens: [batch_size, max_visit_num, input_med_num, emb_size]
        last_medications: [batch_size, max_visit_num, max_med_num, emb_size]
        last_m_mask: [batch_size, max_visit_num, max_med_num]
        cross_visit_scores: [batch_size, max_visit_num, max_visit_num]
        """
        max_visit_num = decode_input_hiddens.size(1)
        input_med_num = decode_input_hiddens.size(2)
        max_med_num = last_medications.size(2)
        copy_query = self.Wc(decode_input_hiddens).view(-1, max_visit_num*input_med_num, self.emb_dim)
        attn_scores = torch.matmul(copy_query, last_medications.view(-1, max_visit_num*max_med_num, self.emb_dim).transpose(-2, -1)) / math.sqrt(self.emb_dim)
        med_mask = last_m_mask.view(-1, 1, max_visit_num * max_med_num).repeat(1, max_visit_num * input_med_num, 1)
        # [batch_size, max_vist_num * input_med_num, max_visit_num * max_med_num]
        attn_scores = F.softmax(attn_scores + med_mask, dim=-1)

        # (batch_size, max_visit_num * input_med_num, max_visit_num)
        visit_scores = cross_visit_scores.repeat(1, 1, input_med_num).view(-1, max_visit_num * input_med_num, max_visit_num)

        # (batch_size, max_visit_num * input_med_num, max_visit_num * max_med_num)
        visit_scores = visit_scores.unsqueeze(-1).repeat(1, 1, 1, max_med_num).view(-1, max_visit_num * input_med_num, max_visit_num * max_med_num)

        scores = torch.mul(attn_scores, visit_scores).clamp(min=1e-9)
        row_scores = scores.sum(dim=-1, keepdim=True)
        scores = scores / row_scores    # (batch_size, max_visit_num * input_med_num, max_visit_num * max_med_num)

        return scores


class MedTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 layer_norm_eps=1e-5) -> None:
        super(MedTransformerDecoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2d_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.m2p_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.nhead = nhead

        # self.align = nn.Linear(d_model, d_model)

    def forward(self, input_medication_embedding, input_medication_memory, input_disease_embdding, input_proc_embedding, 
        input_medication_self_mask, d_mask, p_mask):
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            input_medication_embedding: [*, max_med_num+1, embedding_size]
        Shape:
            see the docs in Transformer class.
        """
        input_len = input_medication_embedding.size(0)
        tgt_len = input_medication_embedding.size(1)

        # [batch_size*visit_num, max_med_num+1, max_med_num+1]
        subsequent_mask = self.generate_square_subsequent_mask(tgt_len, input_len * self.nhead, input_disease_embdding.device)
        self_attn_mask = subsequent_mask + input_medication_self_mask

        x = input_medication_embedding + input_medication_memory

        x = self.norm1(x + self._sa_block(x, self_attn_mask))
        # attentioned_disease_embedding = self._m2d_mha_block(x, input_disease_embdding, d_mask)
        # attentioned_proc_embedding = self._m2p_mha_block(x, input_proc_embedding, p_mask)
        # x = self.norm3(x + self._ff_block(torch.cat([attentioned_disease_embedding, self.align(attentioned_proc_embedding)], dim=-1)))
        x = self.norm2(x + self._m2d_mha_block(x, input_disease_embdding, d_mask) + self._m2p_mha_block(x, input_proc_embedding, p_mask))
        x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _m2d_mha_block(self, x, mem, attn_mask):
        x = self.m2d_multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                need_weights=False)[0]
        return self.dropout2(x)
    
    def _m2p_mha_block(self, x, mem, attn_mask):
        x = self.m2p_multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def generate_square_subsequent_mask(self, sz: int, batch_size: int, device):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        return mask


class PositionEmbedding(nn.Module):
    """
    We assume that the sequence length is less than 512.
    """
    def __init__(self, emb_size, max_length=512):
        super(PositionEmbedding, self).__init__()
        self.max_length = max_length
        self.embedding_layer = nn.Embedding(max_length, emb_size)

    def forward(self, batch_size, seq_length, device):
        assert(seq_length <= self.max_length)
        ids = torch.arange(0, seq_length).long().to(torch.device(device))
        ids = ids.unsqueeze(0).repeat(batch_size, 1)
        emb = self.embedding_layer(ids)
        return emb


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.parameter.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class policy_network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(policy_network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)