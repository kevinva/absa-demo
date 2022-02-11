import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, pack_sequence
import math

DEBUG_ON = False

class ATAE_LSTM(nn.Module):

    def __init__(self, embed_dim, vocab_size, hidden_dim, polarities_dim=3, rnn_num_layers=1, batch_first=True, embedding_matrix=None):
        super(ATAE_LSTM, self).__init__()

        if embedding_matrix is None:
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
            
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicRNN(embed_dim * 2, hidden_dim, num_layers=rnn_num_layers, batch_first=batch_first)
        self.attention = NoQueryAttention(hidden_dim + embed_dim, score_function='bi_linear')
        self.dense = nn.Linear(hidden_dim, polarities_dim)

    def forward(self, inputs):
        text_indices, aspect_indices = inputs[0], inputs[1]

        if DEBUG_ON:
            print(f'text_indices: {text_indices}, \naspect_indices: {aspect_indices}')

        x_len = torch.sum(text_indices != 0, dim=-1)
        if DEBUG_ON:
            print(f'x_len: {x_len}')
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1).float()

        x = self.embed(text_indices)
        x = self.squeeze_embedding(x, x_len)
        if DEBUG_ON:
            print(f'x embed: {x}')

        aspect = self.embed(aspect_indices)
        if DEBUG_ON:
            print(f'aspect embed: {aspect}')
        
        if DEBUG_ON:
            print(f'torch.sum(aspect, dim=1): {torch.sum(aspect, dim=1)}')
            print(f'aspect_len: {aspect_len}, aspect_len.unsqueeze(1): {aspect_len.unsqueeze(1)}')
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.unsqueeze(1))
        if DEBUG_ON:
            print(f'aspect_pool: {aspect_pool}')
            print(f'aspect_pool.unsqueeze(1): {aspect_pool.unsqueeze(1)}')
        aspect = aspect_pool.unsqueeze(1).expand(-1, x_len_max, -1)
        if DEBUG_ON:
            print(f'aspect formatted: {aspect}')
        x = torch.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        if DEBUG_ON:
            print(f'lstm h size: {h.size()}, \nha size: {ha.size()}')

        _, score = self.attention(ha)
        if DEBUG_ON:
            print(f'score : {score}')

        score_h = torch.bmm(score, h)
        if DEBUG_ON:
            print(f'score_h size: {score_h.size()}')

        output = torch.squeeze(score_h, dim=1)
        if DEBUG_ON:
            print(f'output size: {output.size()}')

        out = self.dense(output)

        return out

class DynamicRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        super(DynamicRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bias=bias,
                               batch_first=batch_first,
                               dropout=dropout,
                               bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bias=bias,
                              batch_first=batch_first,
                              dropout=dropout,
                              bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bias=bias,
                              batch_first=batch_first,
                              dropout=dropout,
                              bidirectional=bidirectional)


    def forward(self, x, x_len):
        # x为一个batch的样本序列
        # x_len为x中每个样本的长度序列
        if DEBUG_ON:
            print('start rnn forward ====>')
            print('x raw: ', x)
            print('x len: ', x_len)

        x_sort_idx = torch.sort(-x_len)[1].long() # x由大到小的排列的索引号
        if DEBUG_ON:
            print('x_sort_idx: ', x_sort_idx)

        x_unsort_idx = torch.sort(x_sort_idx)[1].long()  # 用于恢复序列最初的排序（对序列的负数sort一次，再对返回的索引sort一次）
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        if DEBUG_ON:
            print('x_sort: ', x)

        # pack_padded_sequence会把RNN每一步要处理的batch中每个样本的数据结构整理好，不用输入embedding层之后自行调整数据结构
        x_emb_p = pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        if DEBUG_ON:
            print('x_emb_p: ', x_emb_p)

        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None

        ht = torch.transpose(ht, 0, 1)[x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            if DEBUG_ON:
                print('out_pack: ', out_pack)
            out = pad_packed_sequence(out_pack, batch_first=self.batch_first)
            if DEBUG_ON:
                print('out1: ', out)

            out = out[0]
            if DEBUG_ON:
                print('out2: ', out)
            out = out[x_unsort_idx]
            if DEBUG_ON:
                print('out3: ', out)

            if DEBUG_ON:
                print('ct: ', ct)
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[x_unsort_idx]
                ct = torch.transpose(ct, 0, 1)
        if DEBUG_ON:
            print('<===== end rnn forward!')
        return out, (ht, ct)


class SqueezeEmbedding(nn.Module):

    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, x_len):
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]

        x_emb_p = pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        out = pad_packed_sequence(x_emb_p, batch_first=self.batch_first)
        out = out[0]
        out = out[x_unsort_idx]

        return out

class Attention(nn.Module):

    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if self.score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot product或scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameter()
    
    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2: # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2: # k_len missing
            k = torch.unsqueeze(k, dim=1)

        mb_size = k.shape[0]  # batch_size?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim)
        # q: (?, q_len, embed_dim)

        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        # kx: (n_head * ?, k_len, hidden_dim)

        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        # qx: (n_head * ?, q_len, hidden_dim)

        if self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(kx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        elif self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)  # bmm: 带batch维的矩阵乘法
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)  # bmm: 带batch维的矩阵乘法
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        else:
            raise RuntimeError('invalid score function')

        score = F.softmax(score, dim=-1)
        # score: (n_head*?, q_len, k_len)
        output = torch.bmm(score, kx)  # kx当是value
        # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score

class NoQueryAttention(Attention):

    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]  # batch_size?
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)



if __name__ == '__main__':
    # test_seq = torch.tensor([2, 4, 5, 1, 9, 6, 4])
    # print('source seq: ', test_seq)
    # sorted = torch.sort(-test_seq)
    # print(sorted)
    # print(sorted[1].long())
    # sort_seq = test_seq[sorted[1].long()]
    # print('sort seq: ', sort_seq)

    # unsorted = torch.sort(sorted[1].long())
    # print(unsorted[1].long())
    # print('unsort seq: ', sort_seq[unsorted[1].long()])

    a = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    print('unpermute size: ', a.size())
    a_permute = a.permute(2, 0, 1)
    print('permute size: ', a_permute.size())
    print('a_permute: ', a_permute)
