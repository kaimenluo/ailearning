import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import time

# set the random seeds for deterministic 14 results
SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# must first download data on the command line
# python -m spacy download en
# python -m spacy download de
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


# tokenize data
def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens)
    """
    # return [tok.text for tok in spacy_de.tokenizer(text)][::-1]
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


# add init token and eos token, and other operations
# code of Field:
# https://github.com/pytorch/text/blob/master/torchtext/data/field.py#L61

SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

# Multi30k is the name of dataset
# This is a dataset with ~30,000 parallel English, German and French sentences, each with ~12 words per sentence.
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

# build vocab for source and target language for the tokens appear at least 2 times
# words not exist in vocab is transformed to <unk>
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# create iterators
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use gpu if we have

BATCH_SIZE = 128

# BucketIterator.splits() shuffle and padding automatically
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        '''

        :param input_dim: 输入源词库的大小
        :param emb_dim:  输入单词Embedding的维度
        :param hid_dim: 隐层的维度
        :param n_layers: 几个隐层
        :param dropout:  dropout参数 0.5
        '''

        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers = n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src sent len, batch size] 这句话的长度和batch大小

        embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        output, (hidden, cell) = self.rnn.forward(embedded)

        # x = [scr len, emb_dim]
        # w_xh = [emb_dim, hid_dim, n_layers]

        # scr sen len, batch size, hid dim, n directions, n layers
        # outputs: [src sent len, batch size, hid dim * n directions]
        # hidden, cell: [n layers* n directions, batch size, hid dim]
        # outputs are always from the top hidden layer

        # The RNN returns:
        # outputs (the top-layer hidden state for each time-step)
        # hidden (the final hidden state for each layer, stacked on top of
        # each other)

        # and cell (the final cell state for each layer, stacked on top of
        # each other)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0) #增维操作

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # !! sent len and n directions will always be 1 in the decoder,therefore:

        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        output_new = output.squeeze(0)
        # output_new = [batch size, hid dim]

        prediction = self.out(output_new)

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


# seq2seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size,
                              trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder

        hidden, cell = self.encoder.forward(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, max_len):
            # insert input token embedding, previous hidden and previous cell states

            # receive output tensor (predictions) and new hidden and cell states

            output, hidden, cell = self.decoder.forward(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token

            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, 203 use predicted token
            # 在 模型训练速度 和 训练测试差别不要太大 作一个均衡
            input = trg[t] if teacher_force else top1

        return outputs

# Training
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

# print(torch.cuda.is_available())

# init weights
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

# calculate the number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# optimizer
optimizer = optim.Adam(model.parameters())

# index of <pad>
PAD_IDX = TRG.vocab.stoi['<pad>']
# criterion
# we ignore the loss whenever the target token is a padding token
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

#训练
def train(model, iterator, optimizer, criterion, clip):
     model.train()

     epoch_loss = 0

     for i, batch in enumerate(iterator):
         src = batch.src
         print(src.size())
         trg = batch.trg

         optimizer.zero_grad()

         output = model.forward(src, trg)

         #trg = [trg sent len, batch size]
         #output = [trg sent len, batch size, output dim]

         output = output[1:].view(-1, output.shape[-1])
         trg = trg[1:].view(-1)

         #output = [(trg sent len - 1) * batch size, output dim]
         #trg = [(trg sent len - 1) * batch size]

         loss = criterion(output, trg)

         loss.backward()

         #gradient clipping 防止梯度爆炸问题
         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

         optimizer.step()

         epoch_loss += loss.item()

     return epoch_loss / len(iterator)


# 测试
def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model.forward(src, trg, 0)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# calculate time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# N_EPOCHS = 10
N_EPOCHS = 1
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')

model.load_state_dict(torch.load('tut1-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f}')


