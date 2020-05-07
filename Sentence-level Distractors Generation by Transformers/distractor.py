# -*- coding: utf-8 -*-
"""The code is modified based on “6 - Attention is All You Need.ipynb”

Original file is located at
    https://colab.research.google.com/drive/1GHv1abE5PPpsP-fI2jAJhyuuN-Kl5oPf

"""

datasetDir = '/home/ubuntu/DistractorTransformer/distractor_package/RACE_BLEU'
batch_size = 32
N_EPOCHS = 10
CLIP = 1.0
LEARNING_RATE = 1e-4
LR_DECAY = 0.5
LR_DECAY_EPOCH = 2

import json
def getmaxlen(field):
  maxlen = 0
  with open(datasetDir+'/race_train.json') as f:
    for line in f:
      d = json.loads(line)
      if len(d[field]) > maxlen:
        maxlen = len(d[field])
  print(field, 'train maxlen', maxlen)
  maxlen = 0
  with open(datasetDir+'/race_dev.json') as f:
    for line in f:
      d = json.loads(line)
      if len(d[field]) > maxlen:
        maxlen = len(d[field])
  print(field, 'dev maxlen', maxlen)
  maxlen = 0
  with open(datasetDir+'/race_test.json') as f:
    for line in f:
      d = json.loads(line)
      if len(d[field]) > maxlen:
        maxlen = len(d[field])
  print(field, 'test maxlen', maxlen)

getmaxlen('article')
getmaxlen('question')
getmaxlen('answer_text')
getmaxlen('distractor')


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchtext
from torchtext import data
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT = data.Field(init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)
SCORE = data.Field(sequential=False, dtype=torch.float, batch_first=True, use_vocab=False, preprocessing=float)

train_set, valid_set, test_set = data.TabularDataset.splits(
    path=datasetDir, train='race_train.json',
    validation='race_dev.json', test='race_test.json', format='json',
    fields={'question': ('question', TEXT),
            'answer_text': ('answer_text', TEXT),
            'article': ('article', TEXT),
            'distractorBleu': ('distractor', TEXT),
            'bleu-1': ('bleu1', SCORE)})


train_iterator = data.Iterator(train_set, batch_size=batch_size, sort_key=lambda x: len(x.question), shuffle=True, device=device)
valid_iterator = data.Iterator(valid_set, batch_size=batch_size, sort_key=lambda x: len(x.question), shuffle=True, device=device)
test_iterator = data.Iterator(test_set, batch_size=batch_size, sort_key=lambda x: len(x.question), shuffle=False, device=device)

TEXT.build_vocab(train_set, valid_set, min_freq=1)
print('len(TEXT.vocab)', len(TEXT.vocab))

example_idx = 8
ans = vars(train_set.examples[example_idx])['answer_text']
ques = vars(train_set.examples[example_idx])['question']
doc = vars(train_set.examples[example_idx])['article']
dis = vars(train_set.examples[example_idx])['distractor']
bleu1 = vars(train_set.examples[example_idx])['bleu1']
print(f'question = {ques}')
print(f'answer = {ans}')
print(f'distractor = {dis}')
print(f'bleu1 score = {bleu1}')


class QuestionEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 72):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([QuestionEncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):

        #src = [batch size, src len]
        #src_mask = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        #src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        #src = [batch size, src len, hid dim]

        return src


class DocumentEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 800):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DocumentEncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, ques_enc, src_mask, ques_mask):

        #src = [batch size, src len]
        #src_mask = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        #src = [batch size, src len, hid dim]

        for layer in self.layers:
            src, att = layer(src, ques_enc, src_mask, ques_mask)

        #src = [batch size, src len, hid dim]

        return src, att

class AnswerEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 42):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([AnswerEncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, doc_enc, src_mask, doc_mask):

        #src = [batch size, src len]
        #src_mask = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        #src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, doc_enc, src_mask, doc_mask)

        #src = [batch size, src len, hid dim]

        return src



class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]

        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        return src

class QuestionEncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]

        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        return src

class DocumentEncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, enc_ques, src_mask, ques_mask):

        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]

        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]
        #encoder attention
        _src, attention = self.encoder_attention(src, enc_ques, enc_ques, ques_mask)

        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        return src, attention

class AnswerEncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, enc_doc, src_mask, doc_mask):

        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]

        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #encoder attention
        _src, attention = self.encoder_attention(src, enc_doc, enc_doc, doc_mask)

        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        #energy = [batch size, n heads, seq len, seq len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim = -1)

        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        #x = [batch size, n heads, seq len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, seq len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x = [batch size, seq len, hid dim]

        x = self.fc_o(x)

        #x = [batch size, seq len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        #x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        #x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        #x = [batch size, seq len, hid dim]

        return x


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 80):
        super().__init__()

        self.device = device
        self.hid_dim = hid_dim
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  self.device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)

    def forward(self, trg, bleu, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        #pos = [batch size, trg len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        #trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        #output = [batch size, trg len, output dim]

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]

        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        #dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]

        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        #dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        #dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        return trg, attention



class Seq2Seq(nn.Module):
    def __init__(self,
                 ans_encoder,
                 doc_encoder,
                 ques_encoder,
                 decoder,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()
        self.ques_encoder = ques_encoder
        self.doc_encoder = doc_encoder
        self.ans_encoder = ans_encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):

        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        #trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        #trg_pad_mask = [batch size, 1, trg len, 1]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()


        #trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        #trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, ans_src, doc_src, ques_src, trg, bleu):

        #src = [batch size, src len]
        #trg = [batch size, trg len]
        ques_src_mask = self.make_src_mask(ques_src)
        doc_src_mask = self.make_src_mask(doc_src)
        ans_src_mask = self.make_src_mask(ans_src)
        trg_mask = self.make_trg_mask(trg)

        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        ques_enc_src = self.ques_encoder(ques_src, ques_src_mask)
        doc_enc_src, att = self.doc_encoder(doc_src, ques_enc_src, doc_src_mask, ques_src_mask)
        ans_enc_src = self.ans_encoder(ans_src, doc_enc_src, ans_src_mask, doc_src_mask)

        #enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, bleu, ans_enc_src, trg_mask, ans_src_mask)

        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]

        return output, attention

""" Training the Seq2Seq Model
"""

INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(TEXT.vocab)
HID_DIM = 512
ENC_LAYERS = 1
DEC_LAYERS = 1
ENC_HEADS = 1
DEC_HEADS = 1
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

ans_enc = AnswerEncoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

doc_enc = DocumentEncoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

ques_enc = QuestionEncoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)


TEXT_PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = Seq2Seq(ans_enc, doc_enc, ques_enc, dec, TEXT_PAD_IDX, TEXT_PAD_IDX, device).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights);


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_EPOCH*(len(train_set)//batch_size), gamma=LR_DECAY)


criterion = nn.CrossEntropyLoss(ignore_index = TEXT_PAD_IDX)


def train(model, iterator, optimizer, scheduler, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        ans_src = batch.answer_text
        doc_src = batch.article
        ques_src = batch.question
        trg = batch.distractor
        bleu = batch.bleu1

        optimizer.zero_grad()

        output, _ = model(ans_src, doc_src, ques_src, trg[:,1:-1], bleu)

        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,2:].contiguous().view(-1)

        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):
            ans_src = batch.answer_text
            doc_src = batch.article
            ques_src = batch.question
            trg = batch.distractor
            bleu = batch.bleu1

            output, _ = model(ans_src, doc_src, ques_src, trg[:,1:-1], bleu)

            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,2:].contiguous().view(-1)

            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, scheduler, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
    torch.save(model.state_dict(), 'tut6-model-epoch'+str(epoch)+'.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')



model.load_state_dict(torch.load('tut6-model-epoch6.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

""" Inference
"""

def create_src_tensor(sentence, src_field):
    tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    return src_tensor, src_mask

def translate_sentence(answer, question, document, bleu, src_field, trg_field, model, device, target_bleu, max_len = 80):

    model.eval()
    ans_tensor, ans_src_mask = create_src_tensor(answer, src_field)
    ques_tensor, ques_src_mask = create_src_tensor(question, src_field)
    doc_tensor, doc_src_mask = create_src_tensor(document, src_field)

    with torch.no_grad():
        ques_enc_src = model.ques_encoder(ques_tensor, ques_src_mask)
        doc_enc_src, att = model.doc_encoder(doc_tensor, ques_enc_src, doc_src_mask, ques_src_mask)
        ans_enc_src = model.ans_encoder(ans_tensor, doc_enc_src, ans_src_mask, doc_src_mask)

    trg_indexes = [trg_field.vocab.stoi[target_bleu]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)



        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, None, ans_enc_src, trg_mask, ans_src_mask)

        pred_token = output.argmax(2)[:,-1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def display_attention(sentence, translation, attention, n_heads = 1, n_rows = 1, n_cols = 1):

    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15,655))

    for i in range(n_heads):

        ax = fig.add_subplot(n_rows, n_cols, i+1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone', aspect="auto")

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'],
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.savefig("fig.png")
    # plt.close()


example_idx = 8

ques = vars(train_set.examples[example_idx])['question']
doc = vars(train_set.examples[example_idx])['article']
dis = vars(train_set.examples[example_idx])['distractor']
ans = vars(train_set.examples[example_idx])['answer_text']
bleu = vars(train_set.examples[example_idx])['bleu1']
print(f'ques = {ques}')
print(f'ans = {ans}')
print(f'dis = {dis}')
print(f'bleu = {bleu}')

translation, attention = translate_sentence(ans, ques, doc, bleu, TEXT, TEXT, model, device, 'b@4')

print(f'predicted trg = {translation}')
translation, attention = translate_sentence(ans, ques, doc, bleu, TEXT, TEXT, model, device, 'b@5')

print(f'predicted trg = {translation}')
translation, attention = translate_sentence(ans, ques, doc, bleu, TEXT, TEXT, model, device, 'b@6')

print(f'predicted trg = {translation}')
translation, attention = translate_sentence(ans, ques, doc, bleu, TEXT, TEXT, model, device, 'b@7')

print(f'predicted trg = {translation}')


# display_attention(ans, doc, att)


example_idx = 6

ques = vars(valid_set.examples[example_idx])['question']
doc = vars(valid_set.examples[example_idx])['article']
dis = vars(valid_set.examples[example_idx])['distractor']
ans = vars(train_set.examples[example_idx])['answer_text']
bleu = vars(train_set.examples[example_idx])['bleu1']
print(f'ques = {ques}')
print(f'ans = {ans}')
print(f'dis = {dis}')
print(f'bleu = {bleu}')


translation, attention = translate_sentence(ans, ques, doc, bleu, TEXT, TEXT, model, device, 'b@4')

print(f'predicted trg = {translation}')

# display_attention(src, translation, attention)


example_idx = 10

ques = vars(test_set.examples[example_idx])['question']
doc = vars(test_set.examples[example_idx])['article']
dis = vars(test_set.examples[example_idx])['distractor']
ans = vars(train_set.examples[example_idx])['answer_text']
bleu = vars(train_set.examples[example_idx])['bleu1']
print(f'ques = {ques}')
print(f'ans = {ans}')
print(f'dis = {dis}')
print(f'bleu = {bleu}')

translation, attention = translate_sentence(ans, ques, doc, bleu, TEXT, TEXT, model, device, 'b@4')

print(f'predicted trg = {translation}')

# display_attention(src, translation, attention)
ques = ['which', 'of', 'the', 'following', 'is', 'true', 'according', 'to', 'the', 'survey', '?']
ans = ['there', 'is', 'the', 'same', 'percentage', 'about', 'people', 'preferring', 'a', 'weekend', 'all', 'by', 'themselves', 'and', 'people', 'spending', 'no', 'more', 'than', '500', 'yuan', 'during', 'weekends', '.']
dis = ['b@0', 'most', 'office', 'workers', 'ca', "n't", 'afford', 'things', 'in', 'supermarkets', ',', 'so', 'they', 'prefer', 'to', 'attend', 'other', 'stores', ',', 'especially', 'when', 'discounts', 'are', 'offered', '.']

translation, attention = translate_sentence(ans, ques, doc, bleu, TEXT, TEXT, model, device, 'b@4')

print(f'special predicted trg = {translation}')


"""## BLEU

Finally we calculate the BLEU score for the Transformer.
"""

from torchtext.data.metrics import bleu_score

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 80):

    trgs = []
    pred_trgs = []
    trgs_filter = []
    pred_trgs_filter = []
    target_bleu_list = ['b@0', 'b@1', 'b@2', 'b@3', 'b@4', 'b@5', 'b@6', 'b@7', 'b@8', 'b@9']

    for datum in data:

        ques = vars(datum)['question']
        ans = vars(datum)['answer_text']
        doc = vars(datum)['article']
        trg = vars(datum)['distractor']
        bleu = vars(datum)['bleu1']


        print('ques = '+' '.join(ques))
        print('ans = '+' '.join(ans))

        trg = trg[1:]
        print('trg = '+' '.join(trg))
        max_bleu = 0
        max_pred_trg = ''
        for target_bleu in target_bleu_list:
            pred_trg, _ = translate_sentence(ans, ques, doc, bleu, src_field, trg_field, model, device, target_bleu, max_len)
            print(target_bleu + ' : '+' '.join(pred_trg))
            #cut off <eos> token, cut off special char "b@n"
            pred_trg = pred_trg[:-1]
            bleu_filter = bleu_score([pred_trg], [[ans]], max_n=1, weights=[1.0])
            print(bleu_filter)
            if bleu_filter > max_bleu:
                max_pred_trg = pred_trg
                max_bleu = bleu_filter
        pred_trgs.append(max_pred_trg)
        trgs.append([trg])
        print('predicted = '+' '.join(max_pred_trg))
        print()
        bleu_filter = bleu_score([pred_trg], [[ans]], max_n=1, weights=[1.0])
        if 0.2<bleu_filter<0.6:
            pred_trgs_filter.append(pred_trg)
            trgs_filter.append([trg])

    orinum = len(pred_trgs)
    newnum = len(pred_trgs_filter)
    print(f'original number = {orinum}')
    print(f'new number = {newnum}')
    return bleu_score(pred_trgs, trgs, max_n=1, weights=[1.0]), \
        bleu_score(pred_trgs, trgs, max_n=2, weights=[1.0/2]*2), \
        bleu_score(pred_trgs, trgs, max_n=3, weights=[1.0/3]*3), \
        bleu_score(pred_trgs, trgs, max_n=4, weights=[1.0/4]*4), \
        bleu_score(pred_trgs_filter, trgs_filter, max_n=1, weights=[1.0]), \
        bleu_score(pred_trgs_filter, trgs_filter, max_n=2, weights=[1.0/2]*2), \
        bleu_score(pred_trgs_filter, trgs_filter, max_n=3, weights=[1.0/3]*3), \
        bleu_score(pred_trgs_filter, trgs_filter, max_n=4, weights=[1.0/4]*4)


bleu1_score, bleu2_score, bleu3_score, bleu4_score, bleu1_score_filter, bleu2_score_filter, bleu3_score_filter, bleu4_score_filter = calculate_bleu(test_set, TEXT, TEXT, model, device)

print(f'BLEU-1 score = {bleu1_score*100:.2f}')
print(f'BLEU-2 score = {bleu2_score*100:.2f}')
print(f'BLEU-3 score = {bleu3_score*100:.2f}')
print(f'BLEU-4 score = {bleu4_score*100:.2f}')

print(f'BLEU-1 filter score = {bleu1_score_filter*100:.2f}')
print(f'BLEU-2 filter score = {bleu2_score_filter*100:.2f}')
print(f'BLEU-3 filter score = {bleu3_score_filter*100:.2f}')
print(f'BLEU-4 filter score = {bleu4_score_filter*100:.2f}')
