import torch
# from torch._C import contiguous_format
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import GraphConvolution
import math
from torch.nn.parameter import Parameter

'''
Our model
'''


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
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


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(
            N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):
        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors


class SafeDrugModel(nn.Module):
    def __init__(self, vocab_size, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprints, average_projection, emb_dim=256, device=torch.device('cpu:0')):
        super(SafeDrugModel, self).__init__()

        self.device = device

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)])
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )

        # bipartite local embedding
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(
            ddi_mask_H.shape[1], vocab_size[2], False)

        # MPNN global embedding
        self.MPNN_molecule_Set = list(zip(*MPNNSet))

        self.MPNN_emb = MolecularGraphNeuralNetwork(
            N_fingerprints, emb_dim, layer_hidden=2, device=device).forward(self.MPNN_molecule_Set)
        self.MPNN_emb = torch.mm(average_projection.to(
            device=self.device), self.MPNN_emb.to(device=self.device))
        # self.MPNN_emb.to(device=self.device)
        self.MPNN_emb = torch.tensor(self.MPNN_emb, requires_grad=True)
        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        self.init_weights()

    def forward(self, input):

        # patient health representation
        i1_seq = []
        i2_seq = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm in input:
            i1 = sum_embedding(self.dropout(self.embeddings[0](
                torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))))  # (1,1,dim)
            i2 = sum_embedding(self.dropout(self.embeddings[1](
                torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)

        o1, h1 = self.encoders[0](
            i1_seq
        )
        o2, h2 = self.encoders[1](
            i2_seq
        )
        patient_representations = torch.cat(
            [o1, o2], dim=-1).squeeze(dim=0)  # (seq, dim*2)
        query = self.query(patient_representations)[-1:, :]  # (seq, dim)

        # MPNN embedding
        MPNN_match = F.sigmoid(torch.mm(query, self.MPNN_emb.t()))
        MPNN_att = self.MPNN_layernorm(
            MPNN_match + self.MPNN_output(MPNN_match))

        # local embedding
        bipartite_emb = self.bipartite_output(
            F.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t())

        result = torch.mul(bipartite_emb, MPNN_att)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


class DMNC(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device('cpu:0')):
        super(DMNC, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[2]
        self.token_end = vocab_size[2] + 1

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i] if i != 2 else vocab_size[2] + 2, emb_dim) for i in range(K)])
        self.dropout = nn.Dropout(p=0.5)

        self.encoders = nn.ModuleList([DNC(
            input_size=emb_dim,
            hidden_size=emb_dim,
            rnn_type='gru',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=16,
            cell_size=emb_dim,
            read_heads=1,
            batch_first=True,
            gpu_id=0,
            independent_linears=False
        ) for _ in range(K - 1)])

        self.decoder = nn.GRU(emb_dim + emb_dim * 2, emb_dim * 2,
                              batch_first=True)  # input: (y, r1, r2,) hidden: (hidden1, hidden2)
        self.interface_weighting = nn.Linear(
            emb_dim * 2, 2 * (emb_dim + 1 + 3))  # 2 read head (key, str, mode)
        self.decoder_r2o = nn.Linear(2 * emb_dim, emb_dim * 2)

        self.output = nn.Linear(emb_dim * 2, vocab_size[2] + 2)

    def forward(self, input, i1_state=None, i2_state=None, h_n=None, max_len=20):
        # input (3, code)
        i1_input_tensor = self.embeddings[0](
            torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
        i2_input_tensor = self.embeddings[1](
            torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        o1, (ch1, m1, r1) = \
            self.encoders[0](i1_input_tensor, (None, None, None)
                             if i1_state is None else i1_state)
        o2, (ch2, m2, r2) = \
            self.encoders[1](i2_input_tensor, (None, None, None)
                             if i2_state is None else i2_state)

        # save memory state
        i1_state = (ch1, m1, r1)
        i2_state = (ch2, m2, r2)

        predict_sequence = [self.token_start] + input[2]
        if h_n is None:
            h_n = torch.cat([ch1[0], ch2[0]], dim=-1)

        output_logits = []
        r1 = r1.unsqueeze(dim=0)
        r2 = r2.unsqueeze(dim=0)

        if self.training:
            for item in predict_sequence:
                # teacher force predict drug
                item_tensor = self.embeddings[2](
                    torch.LongTensor([item]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

                o3, h_n = self.decoder(
                    torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(
                    h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys[:, 0, :].unsqueeze(
                                                  dim=1),
                                              read_strengths[:, 0].unsqueeze(
                                                  dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(
                                                  dim=1),
                                              read_strengths[:, 1].unsqueeze(
                                                  dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output_logits.append(output)
        else:
            item_tensor = self.embeddings[2](
                torch.LongTensor([self.token_start]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)
            for idx in range(max_len):
                # predict
                # teacher force predict drug
                o3, h_n = self.decoder(
                    torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(
                    h_n.squeeze(0))

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(self.encoders[0],
                                              read_keys[:, 0, :].unsqueeze(
                                                  dim=1),
                                              read_strengths[:, 0].unsqueeze(
                                                  dim=1),
                                              read_modes[:, 0, :].unsqueeze(dim=1), i1_state[1])

                r2, _ = self.read_from_memory(self.encoders[1],
                                              read_keys[:, 1, :].unsqueeze(
                                                  dim=1),
                                              read_strengths[:, 1].unsqueeze(
                                                  dim=1),
                                              read_modes[:, 1, :].unsqueeze(dim=1), i2_state[1])

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output = F.softmax(output, dim=-1)
                output_logits.append(output)

                input_token = torch.argmax(output, dim=-1)
                input_token = input_token.item()
                item_tensor = self.embeddings[2](
                    torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device))  # (1, seq, codes)

        return torch.cat(output_logits, dim=0), i1_state, i2_state, h_n

    def read_from_memory(self, dnc, read_key, read_str, read_mode, m_hidden):
        read_vectors, hidden = dnc.memories[0].read(
            read_key, read_str, read_mode, m_hidden)
        return read_vectors, hidden

    def decode_read_variable(self, input):
        w = 64
        r = 2
        b = input.size(0)

        input = self.interface_weighting(input)
        # r read keys (b * w * r)
        read_keys = F.tanh(input[:, :r * w].contiguous().view(b, r, w))
        # r read strengths (b * r)
        read_strengths = F.softplus(
            input[:, r * w:r * w + r].contiguous().view(b, r))
        # read modes (b * 3*r)
        read_modes = F.softmax(
            input[:, (r * w + r):].contiguous().view(b, r, 3), -1)
        return read_keys, read_strengths, read_modes


class GAMENet(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cpu:0'), ddi_in_memory=True):
        super(GAMENet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.5)

        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(K-1)])

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.ehr_gcn = GCN(
            voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(
            voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

        self.init_weights()

    def forward(self, input):
        # input (adm, 3, codes)

        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for adm in input:
            i1 = mean_embedding(self.dropout(self.embeddings[0](
                torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))))  # (1,1,dim)
            i2 = mean_embedding(self.dropout(self.embeddings[1](
                torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)

        o1, h1 = self.encoders[0](
            i1_seq
        )  # o1:(1, seq, dim*2) hi:(1,1,dim*2)
        o2, h2 = self.encoders[1](
            i2_seq
        )
        patient_representations = torch.cat(
            [o1, o2], dim=-1).squeeze(dim=0)  # (seq, dim*4)
        queries = self.query(patient_representations)  # (seq, dim)

        # graph memory module
        '''I:generate current input'''
        query = queries[-1:]  # (1,dim)

        '''G:generate graph memory bank and insert history information'''
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        else:
            drug_memory = self.ehr_gcn()

        if len(input) > 1:
            history_keys = queries[:(queries.size(0)-1)]  # (seq-1, dim)

            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(
                history_values).to(self.device)  # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(
            torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(
                torch.mm(query, history_keys.t()))  # (1, seq-1)
            weighted_values = visit_weight.mm(history_values)  # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory)  # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        output = self.output(
            torch.cat([query, fact1, fact2], dim=-1))  # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)


class Leap(nn.Module):
    def __init__(self, voc_size, emb_dim=64, device=torch.device('cpu:0')):
        super(Leap, self).__init__()
        self.voc_size = voc_size
        self.device = device
        self.SOS_TOKEN = voc_size[2]        # start of sentence
        self.END_TOKEN = voc_size[2]+1      # end   新增的两个编码，两者均是针对于药物的embedding

        # dig_num * emb_dim
        self.enc_embedding = nn.Sequential(
            nn.Embedding(voc_size[0], emb_dim, ),
            nn.Dropout(0.3)
        )

        # med_num * emb_dim
        self.dec_embedding = nn.Sequential(
            nn.Embedding(voc_size[2] + 2, emb_dim, ),
            nn.Dropout(0.3)
        )

        self.dec_gru = nn.GRU(emb_dim*2, emb_dim, batch_first=True)

        self.attn = nn.Linear(emb_dim*2, 1)

        self.output = nn.Linear(emb_dim, voc_size[2]+2)

    def forward(self, input, max_len=20):
        device = self.device
        # input (3, codes)
        input_tensor = torch.LongTensor(input[0]).to(device)
        # (len, dim)
        # 对疾病进行编码
        input_embedding = self.enc_embedding(
            input_tensor.unsqueeze(dim=0)).squeeze(dim=0)

        output_logits = []
        hidden_state = None
        if self.training:
            # training 过程
            # 对于每一个当前已知的药物分别进行计算，有点像teacher-forcing的思路
            for med_code in [self.SOS_TOKEN] + input[2]:
                dec_input = torch.LongTensor(
                    [med_code]).unsqueeze(dim=0).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(
                    dim=0)  # (1,dim)  取对应药物的embedding

                if hidden_state is None:    # 使用上一个adm的hidden_state
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(
                    input_embedding.size(0), 1)  # (len, dim)

                combined_input = torch.cat(
                    [hidden_state_repeat, input_embedding], dim=-1)  # (len, dim*2)
                # (1, len)   计算该药物针对所有疾病的attention weight
                attn_weight = F.softmax(self.attn(combined_input).t(), dim=-1)
                input_embedding = attn_weight.mm(
                    input_embedding)  # (1, dim)    # 加权求和

                _, hidden_state = self.dec_gru(torch.cat(
                    [input_embedding, dec_input], dim=-1).unsqueeze(dim=0), hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0)  # (1,dim)

                # (1, med_num) 表示当前位置上每个med的logits
                output_logits.append(self.output(F.relu(hidden_state)))
            return torch.cat(output_logits, dim=0)

        else:
            # testing 过程，这里不能用input[2]也就是medication的信息
            # 控制最大的长度（可以根据数据的范围调整）
            for di in range(max_len):
                if di == 0:
                    dec_input = torch.LongTensor([[self.SOS_TOKEN]]).to(
                        device)  # 第一个位置用SOS，后面的则用上一个位置的预测结果
                dec_input = self.dec_embedding(
                    dec_input).squeeze(dim=0)  # (1,dim)
                if hidden_state is None:
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(
                    input_embedding.size(0), 1)  # (len, dim)
                combined_input = torch.cat(
                    [hidden_state_repeat, input_embedding], dim=-1)  # (len, dim*2)
                attn_weight = F.softmax(
                    self.attn(combined_input).t(), dim=-1)  # (1, len)
                input_embedding = attn_weight.mm(input_embedding)  # (1, dim)
                _, hidden_state = self.dec_gru(torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0),
                                               hidden_state.unsqueeze(dim=0))
                hidden_state = hidden_state.squeeze(dim=0)  # (1,dim)
                output = self.output(F.relu(hidden_state))
                # data是直接取数据，这里直接获取当前位置上最有可能的logits
                topv, topi = output.data.topk(1)
                output_logits.append(F.softmax(output, dim=-1))
                dec_input = topi.detach()
            return torch.cat(output_logits, dim=0)


class Retain(nn.Module):
    def __init__(self, voc_size, emb_size=64, device=torch.device('cpu:0')):
        super(Retain, self).__init__()
        self.device = device
        self.voc_size = voc_size
        self.emb_size = emb_size
        self.input_len = voc_size[0] + voc_size[1] + voc_size[2]
        self.output_len = voc_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_size,
                         padding_idx=self.input_len),
            nn.Dropout(0.5)
        )

        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)

        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)

        self.output = nn.Linear(emb_size, self.output_len)

    def forward(self, input):
        device = self.device
        # input: (visit, 3, codes )
        max_len = max([(len(v[0]) + len(v[1]) + len(v[2])) for v in input])
        input_np = []
        for visit in input:
            input_tmp = []
            input_tmp.extend(visit[0])
            input_tmp.extend(list(np.array(visit[1]) + self.voc_size[0]))
            input_tmp.extend(
                list(np.array(visit[2]) + self.voc_size[0] + self.voc_size[1]))
            if len(input_tmp) < max_len:
                input_tmp.extend([self.input_len]*(max_len - len(input_tmp)))

            input_np.append(input_tmp)

        visit_emb = self.embedding(torch.LongTensor(
            input_np).to(device))  # (visit, max_len, emb)
        visit_emb = torch.sum(visit_emb, dim=1)  # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0))  # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0))  # h: (1, visit, emb)

        g = g.squeeze(dim=0)  # (visit, emb)
        h = h.squeeze(dim=0)  # (visit, emb)
        attn_g = F.softmax(self.alpha_li(g), dim=-1)  # (visit, 1)
        attn_h = F.tanh(self.beta_li(h))  # (visit, emb)

        c = attn_g * attn_h * visit_emb  # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0)  # (1, emb)

        return self.output(c)


class Leap_batch(nn.Module):
    def __init__(self, voc_size, emb_dim=64, device=torch.device('cpu:0')):
        super(Leap_batch, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        self.SOS_TOKEN = voc_size[2]        # start of sentence
        self.END_TOKEN = voc_size[2]+1      # end   新增的两个编码，两者均是针对于药物的embedding
        self.MED_PAD_TOKEN = voc_size[2]+2      # 用于embedding矩阵中的padding（全为0）
        self.DIAG_PAD_TOKEN = voc_size[0]+2
        self.PROC_PAD_TOKEN = voc_size[1]+2
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
            nn.Embedding(voc_size[2] + 3, emb_dim, self.MED_PAD_TOKEN),
            nn.Dropout(0.3)
        )

        # 用于对上一个visit进行编码
        self.enc_gru = nn.GRU(emb_dim, emb_dim, batch_first=True, bidirectional=True)

        # 用于生成药物序列
        self.dec_gru = nn.GRU(emb_dim*2, emb_dim, batch_first=True)

        self.attn = nn.Linear(emb_dim*2, 1)

        # self.output = nn.Linear(emb_dim, voc_size[2]+2)
        # self.output2 = nn.Linear(emb_dim, voc_size[2]+2)

        # weights
        self.Ws = nn.Linear(emb_dim*2, emb_dim)  # only used at initial stage
        self.Wo = nn.Linear(emb_dim, voc_size[2]+2)  # generate mode
        self.Wc = nn.Linear(emb_dim*2, emb_dim)  # copy mode

    def encoder(self, x):
        # input: (med_num)
        embedded = self.dec_embedding(x)
        out, h = self.enc_gru(embedded)  # out: [b x seq x hid*2] (biRNN)
        return out, h

    # def forward(self, input, last_input=None, max_len=20):
    def forward(self, diseases, procedures, medications, d_mask_matrix, m_mask_matrix, seq_length, max_len=20):
        device = self.device
        # batch维度以及seq维度上并行计算（现在不考虑时间序列信息），每一个medication序列仍然按顺序预测
        batch_size, max_seq_length, max_med_num = medications.size()
        max_diag_num = diseases.size()[2]
        hidden_state = None
        
        # print("diasease size", diseases.size())
        # print("proc size", procedures.size())
        # print("med size", medications.size())
        input_disease_embdding = self.diag_embedding(diseases)      # [batch, seq, max_d_num, emb]
        input_med_embedding = self.med_embedding(medications)       # [batch, seq, max_med_num, emb]
        
        # 拼接一个last_seq_medication，表示对应seq对应的上一次的medication，第一次的由于没有上一次medication，用0填补（用啥填补都行，反正不会用到）
        last_seq_medication = torch.full((batch_size, 1, max_med_num), 0).to(device)
        last_seq_medication = torch.cat([last_seq_medication, medications[:, :-1, :]], dim=1)
        # m_mask_matrix矩阵同样也需要后移
        last_m_mask = torch.full((batch_size, 1, max_med_num), -1e9).to(device) # 这里用较大负值，避免softmax之后分走了概率
        last_m_mask = torch.cat([last_m_mask, m_mask_matrix[:, :-1, :]], dim=1)

        last_seq_medication_emb = self.med_embedding(last_seq_medication)
        # print(last_seq_medication.size(), input_med_embedding.size())
        # 对last_visit进行编码，注意这里用的是last_seq_medication的编码结果
        # (batch*seq, max_med_num, emb_dim*2)
        encoded_disease, _ = self.enc_gru(last_seq_medication_emb.view(batch_size * max_seq_length, max_med_num, self.emb_dim))

        # 同样拼接一个last_medication，用于进行序列生成，注意维度上增加了一维
        last_medication = torch.full((batch_size, max_seq_length, 1), self.SOS_TOKEN).to(device)    # [batch_size, seq, 1]
        last_medication = torch.cat([last_medication, medications], dim=2)      # [batch_size, seq, max_med_num + 1]
        # print(last_medication.size(), medications.size())
        
        hidden_state = None
        # 预定义结果矩阵
        if self.training:
            output_logits = torch.zeros(batch_size, max_seq_length, max_med_num+1, self.voc_size[2]+2).to(device)
            loop_size=max_med_num+1
        else:
            output_logits = torch.zeros(batch_size, max_seq_length, max_len, self.voc_size[2]+2).to(device)
            loop_size=max_len

        # 开始遍历生成每一个位置的结果
        for i in range(loop_size):
            if self.training:
                dec_input = self.med_embedding(last_medication[:,:,i])  # (batch, seq, emb_dim)  取上一个药物的embedding
            else:
                if i==0: 
                    dec_input = self.med_embedding(last_medication[:,:,0])
                elif i==max_len:
                    break
                else:
                    # 非训练时，只能取上一次的输出
                    dec_input=self.med_embedding(dec_input)

            if hidden_state is None:
                    hidden_state=dec_input
            
            # 根据当前的疾病做attention，计算hidden_state (batch, seq, emb_dim)
            # print(dec_input.size())
            hidden_state_repeat = hidden_state.unsqueeze(dim=2).repeat(1,1,max_diag_num,1)  # (batch, seq, max_diag_num, emb_dim)
            # print(hidden_state_repeat.size())
            combined_input=torch.cat([hidden_state_repeat, input_disease_embdding], dim=-1) # (batch, seq, max_diag_num, emb_dim*2)
            """这里attn_score结果需要根据mask来加上一个较大的负值，使得对应softmax值接近0，来避免分散注意力"""
            attn_score = self.attn(combined_input).squeeze(dim=-1) # (batch, seq, max_diag_num, 1) -> (batch, seq, max_diag_num)
            attn_score = attn_score + d_mask_matrix

            attn_weight=F.softmax(attn_score, dim=-1).unsqueeze(dim=2)   # (batch, seq, 1, max_diag_num) 注意力权重
            # print(attn_weight.size())
            input_embedding=torch.matmul(attn_weight, input_disease_embdding).squeeze(dim=2)    # (batch, seq, emb_dim)

            # 为了送到dec_gru中进行reshape
            input_embedding_buf = input_embedding.view(batch_size * max_seq_length, 1, -1)
            dec_input_buf = dec_input.view(batch_size * max_seq_length, 1, -1)
            # print(input_embedding_buf.size())
            # print(dec_input_buf.size())
            hidden_state_buf = hidden_state.view(1, batch_size * max_seq_length, -1)
            _, hidden_state_buf = self.dec_gru(torch.cat([input_embedding_buf, dec_input_buf], dim=-1), hidden_state_buf)
            # print(hidden_state_buf.size())
            hidden_state = hidden_state_buf.view(batch_size, max_seq_length, -1)    # (batch, seq, emb_dim)
            # print(hidden_state.size())

            score_g = self.Wo(hidden_state) # (batch, seq, voc_size[2]+2)

            prob = torch.log_softmax(score_g, dim=-1)
            output_logits[:, :, i, :] = prob

            if not self.training:
                # data是直接取数据，这里直接获取当前位置上最有可能的logits
                _, topi = torch.topk(prob, 1, dim=-1)
                dec_input=topi.detach()

        return output_logits


class MICRON(nn.Module):
    def __init__(self, vocab_size, ddi_adj, emb_dim=256, device=torch.device('cpu:0')):
        super(MICRON, self).__init__()

        self.device = device

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        
        self.health_net = nn.Sequential(
                nn.Linear(2 * emb_dim, emb_dim)
        )

        # 
        self.prescription_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, vocab_size[2])
        )

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.init_weights()

    def forward(self, input):

	    # patient health representation
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        
        diag_emb = sum_embedding(self.dropout(self.embeddings[0](torch.LongTensor(input[-1][0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
        prod_emb = sum_embedding(self.dropout(self.embeddings[1](torch.LongTensor(input[-1][1]).unsqueeze(dim=0).to(self.device))))
        # diag_emb = torch.cat(diag_emb, dim=1) #(1,seq,dim)
        # prod_emb = torch.cat(prod_emb, dim=1) #(1,seq,dim)

        if len(input) < 2:
            diag_emb_last = diag_emb * torch.tensor(0.0)
            prod_emb_last = diag_emb * torch.tensor(0.0)
        else:
            diag_emb_last = sum_embedding(self.dropout(self.embeddings[0](torch.LongTensor(input[-2][0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            prod_emb_last = sum_embedding(self.dropout(self.embeddings[1](torch.LongTensor(input[-2][1]).unsqueeze(dim=0).to(self.device))))
            # diag_emb_last = torch.cat(diag_emb_last, dim=1) #(1,seq,dim)
            # prod_emb_last = torch.cat(prod_emb_last, dim=1) #(1,seq,dim)

        health_representation = torch.cat([diag_emb, prod_emb], dim=-1).squeeze(dim=0) # (seq, dim*2)
        health_representation_last = torch.cat([diag_emb_last, prod_emb_last], dim=-1).squeeze(dim=0) # (seq, dim*2)

        health_rep = self.health_net(health_representation)[-1:, :] # (seq, dim)
        health_rep_last = self.health_net(health_representation_last)[-1:, :] # (seq, dim)
        health_residual_rep = health_rep - health_rep_last

	    # drug representation
        drug_rep = self.prescription_net(health_rep)
        drug_rep_last = self.prescription_net(health_rep_last)
        drug_residual_rep = self.prescription_net(health_residual_rep)  

        # reconstructon loss
        rec_loss = 1 / self.tensor_ddi_adj.shape[0] * torch.sum(torch.pow((F.sigmoid(drug_rep) - F.sigmoid(drug_rep_last + drug_residual_rep)), 2))
        
        # ddi_loss
        neg_pred_prob = F.sigmoid(drug_rep)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)

        batch_neg = 1 / self.tensor_ddi_adj.shape[0] *  neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return drug_rep, drug_rep_last, drug_residual_rep, batch_neg, rec_loss

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
