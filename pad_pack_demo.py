import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence


class MyData(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_data(data):
    # # v1
    # data.sort(key=lambda x: len(x), reverse=True)
    # seq_len = [s.size(0) for s in data]
    # data = pad_sequence(data, batch_first=True, padding_value=0)
    # data = pack_padded_sequence(data, seq_len, batch_first=True, enforce_sorted=False) # enforce_sorted默认为True
    
    # # v2
    # data.sort(key=lambda x: len(x), reverse=True)
    # data = pack_sequence(data)
    
    # # v3
    print('0. data:', data)
    data.sort(key=lambda x: len(x), reverse=True)
    print('1. data: ', data)
    seq_len = [s.size(0) for s in data]
    data = pad_sequence(data, batch_first=True).float()
    print('2. data: ', data)
    data = data.unsqueeze(-1)
    print('3. data: ', data)
    data = pack_padded_sequence(data, seq_len, batch_first=True)

    return data


a = torch.tensor([1,2,3,4])
b = torch.tensor([5,6,7])
c = torch.tensor([7,8])
d = torch.tensor([9])
train_x = [a, b, c, d]

data = MyData(train_x)
data_loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate_data)
batch_x = iter(data_loader).next()

print(batch_x)

# test_list = [[1], [2, 34], [23], [342, 14, 0]]
# test_list.sort(key=lambda x: len(x), reverse=True)  # 默认排序是由小到大，即reverse=False
# print(test_list)

rnn = nn.LSTM(1, 4, 1, batch_first=True)
h0 = torch.rand(1, 2, 4).float()
c0 = torch.rand(1, 2, 4).float()
out, (h1, c1) = rnn(batch_x, (h0, c0))

# print(h0)
print('out packed: ', out)

out_pad, out_len = pad_packed_sequence(out, batch_first=True)
print('out paded: ', out_pad)
print('out len: ', out_len)