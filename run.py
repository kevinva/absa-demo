import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from utils import *
from atae_lstm import *
import time
import os
import math
from sklearn import metrics
# import logging
# import sys

EPOCHS_NUM = 10
EPOCHS_PATIENCE = 5
EMBEDDING_SIZE = 3
HIDDEN_SIZE = 6
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 85
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
LOG_INTERVAL = 10

TRAIN_FILE_PATH = '../data/absa/SemEval14/abas-pytorch/Laptops_Train.xml.seg'
TEST_FILE_PATH = '../data/absa/SemEval14/abas-pytorch/Laptops_Test_Gold.xml.seg'

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger.addHandler(logging.StreamHandler(sys.stdout))

def train(model, criterion, optimizer, train_data, val_data, device):
   max_val_acc = 0
   max_val_f1 = 0
   max_val_epoch = 0
   global_step = 0
   path = ''
   train_loss_list = []
   start_time = int(time.time() / 1000)
   for i_epoch in range(EPOCHS_NUM):
      n_corrent, n_total, loss_total = 0, 0, 0
      model.train()
      for i_batch, batch in enumerate(train_data):
         global_step += 1
         optimizer.zero_grad()

         text_tokens = batch['text_indices'].to(device)
         aspect_tokens =  batch['aspect_indices'].to(device)
         targets =  batch['polarity'].to(device)
         # print(f'{i_batch} | text: {text_tokens}')
         # print(f'{i_batch} | aspect: {aspect_tokens}')
         # print(f'{i_batch} | target: {targets}')

         outputs = model((text_tokens, aspect_tokens))
         # print(f'{i_batch} | outputs: {outputs}')
         loss = criterion(outputs, targets)
         # print(f'{i_batch} | loss: {loss}')
         loss.backward()
         optimizer.step()

         n_corrent += (torch.argmax(outputs, -1) == targets).sum().item()
         n_total += len(outputs)
         loss_total += loss.item() * len(outputs)
         if global_step % LOG_INTERVAL == 0:
            train_acc = n_corrent / n_total
            train_loss = loss_total / n_total
            print('epoch: {} | step: {} | elpased time: {} | loss: {:.4f} | acc: {:.4f}'.format(i_epoch, global_step, int(time.time() / 1000) - start_time), train_loss, train_acc)
            start_time = int(time.time() / 1000)

            train_loss_list.append(train_loss)

         if i_batch >= 2:
            break
      
      val_acc, val_f1 = evaluate_acc_f1(model, val_data, device)
      print('===> epoch: {} | step: {} | val_acc: {:.4f} | val_f1: {:.4f}'.format(i_epoch, global_step, val_acc, val_f1)

      if val_acc > max_val_acc:
         max_val_acc = val_acc 
         max_val_epoch = i_epoch
         if not os.path.exists('./output/state_dict/'):
            os.mkdir('./output/state_dict/')
         path = './output/state_dict/{}_val_acc_{}'.format(global_step, round(val_acc, 4))
         torch.save(model.state_dict(), path)

      if val_f1 > max_val_f1:
         max_val_f1 = val_f1

      if i_epoch - max_val_epoch >= EPOCHS_PATIENCE:
         print('early stop.')
         break

      break

def evaluate_acc_f1(model, data_loader, device):
   n_correct, n_total = 0, 0
   t_targets_all, t_outputs_all = None, None
   model.eval()
   with torch.no_grad():
      for i_batch, batch in enumerate(data_loader):
         text_tokens = batch['text_indices'].to(device)
         aspect_tokens =  batch['aspect_indices'].to(device)
         targets =  batch['polarity'].to(device)
         outputs = model((text_tokens, aspect_tokens))

         n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
         n_total += len(outputs)

            if t_targets_all is None:
               t_targets_all = targets
               t_outputs_all = outputs
            else:
               t_targets_all = torch.cat((t_targets_all, targets), dim=0)
               t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)

   acc = n_correct / n_total
   f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
   return acc, f1


def print_model_param(model):
   n_trainable_params, n_nontrainable_params = 0, 0
   for p in model.parameters():
      n_params = torch.prod(torch.tensor(p.shape))
      if p.requires_grad:
            n_trainable_params += n_params
      else:
            n_nontrainable_params += n_params
   # logger.info(f'> n_trainable_params: {n_trainable_params}, n_nontrainable_params: {n_nontrainable_params}')

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
   model = ATAE_LSTM(EMBEDDING_SIZE, vocab_size, HIDDEN_SIZE).to(device)
   # print_model_param(model)

   criterion = nn.CrossEntropyLoss()
   params = filter(lambda p: p.requires_grad, model.parameters())
   optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
   reset_model_param(model)
   train(model, criterion, optimizer, train_data_loader, val_data_loader, device)

   print(f'time: {time.time()}')
   