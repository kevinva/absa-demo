import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from utils import *
from atae_lstm import *
import logging
import sys

EPOCHS = 10
EMBEDDING_SIZE = 3
HIDDEN_SIZE = 6
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 85
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
DROPOUT = 0.1

TRAIN_FILE_PATH = '../data/absa/SemEval14/abas-pytorch/Laptops_Train.xml.seg'
TEST_FILE_PATH = '../data/absa/SemEval14/abas-pytorch/Laptops_Test_Gold.xml.seg'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.addHandler(logging.StreamHandler(sys.stdout))

def train():
   pass

def print_model_param(model):
   n_trainable_params, n_nontrainable_params = 0, 0
   for p in model.parameters():
      n_params = torch.prod(torch.tensor(p.shape))
      if p.requires_grad:
            n_trainable_params += n_params
      else:
            n_nontrainable_params += n_params
   logger.info(f'> n_trainable_params: {n_trainable_params}, n_nontrainable_params: {n_nontrainable_params}')

def reset_model_param(model):
   for child in model.children():
      for p in child.parameters():
         if p.requires_grad:
            if len(p.shape) > 1:
                  nn.init.xavier_uniform_(p)
            else:
                  stdv = 1. / math.sqrt(p.shape[0])
                  nn.init.uniform_(p, a=-stdv, b=stdv)


if __name__ == '__main__':
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   tokenizer = build_tokenizer(
      fnames=[TRAIN_FILE_PATH, TEST_FILE_PATH], 
      max_seq_len=MAX_SEQ_LENGTH, 
      dat_fname='./output/laptop_tokenizer.dat'
   )
   trainset = ABSADataset(TRAIN_FILE_PATH, tokenizer)
   testset = ABSADataset(TEST_FILE_PATH, tokenizer)
   val_len = int(len(trainset) * 0.1)
   trainset, valset = random_split(trainset, [len(trainset) - val_len, val_len])
   # logger.info(trainset[0])
   train_data_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)
   test_data_loader = DataLoader(dataset=testset, batch_size=BATCH_SIZE, shuffle=True)
   val_data_loader = DataLoader(dataset=valset, batch_size=BATCH_SIZE, shuffle=True)

   vocab_size = len(tokenizer.word2idx)  # hoho todo
   # logger.info(f'vocab size={vocab_size}')
   model = ATAE_LSTM(EMBEDDING_SIZE, vocab_size, HIDDEN_SIZE)
   # print_model_param(model)

   criterion = nn.CrossEntropyLoss()
   params = filter(lambda p: p.requires_grad, model.parameters())
   optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
   reset_model_param(model)

   