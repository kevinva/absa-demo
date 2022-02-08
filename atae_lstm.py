import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence, pack_sequence

class ATAE_LSTM(nn.Module):

    def __init__(self):
        pass


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
        print('rnn forward....')
        print('x raw: ', x)
        print('x len: ', x_len)

        x_sort_idx = torch.sort(-x_len)[1].long() # x由大到小的排列的索引号
        print('x_sort_idx: ', x_sort_idx)

        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        print('x_sort: ', x)

        # pack_padded_sequence会把RNN每一步要处理的batch中每个样本的数据结构整理好，不用输入embedding层之后自行调整数据结构
        x_emb_p = pack_padded_sequence(x, x_len, batch_first=self.batch_first)
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
            print('out_pack: ', out_pack)
            out = pad_packed_sequence(out_pack, batch_first=self.batch_first)
            print('out1: ', out)

            out = out[0]
            print('out2: ', out)
            out = out[x_unsort_idx]
            print('out3: ', out)

            print('ct: ', ct)
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[x_unsort_idx]
                ct = torch.transpose(ct, 0, 1)

        return out, (ht, ct)



if __name__ == '__main__':
    test_seq = torch.tensor([2, 4, 5, 1, 9, 6, 4])
    print('source seq: ', test_seq)
    sorted = torch.sort(-test_seq)
    print(sorted)
    print(sorted[1].long())
    sort_seq = test_seq[sorted[1].long()]
    print('sort seq: ', sort_seq)

    unsorted = torch.sort(sorted[1].long())
    print(unsorted[1].long())
    print('unsort seq: ', sort_seq[unsorted[1].long()])