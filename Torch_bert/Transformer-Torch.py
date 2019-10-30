'''
  code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
  Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
              https://github.com/JayParks/transformer
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

dtype = torch.FloatTensor
# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

# Transformer Parameters
# Padding Should be Zero
src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'S' : 5, 'E' : 6}
number_dict = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5
tgt_len = 5

d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]

    input_batch_tensor = torch.LongTensor(input_batch)  # tensor([[1, 2, 3, 4, 0]]) shape:(1,5)
    output_batch_tensor = torch.LongTensor(output_batch)
    target_batch_tensor = torch.LongTensor(target_batch)

    return Variable(input_batch_tensor), Variable(output_batch_tensor), Variable(target_batch_tensor)

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token
    # [1,1,5]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking

    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1) # 返回函数的上三角矩阵
    # [[0,1,1,1,1],[0,0,1,1,1],[0,0,0,1,1],[0,0,0,0,1],[0,0,0,0,0]]
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''

        :param Q: [1,8,5,64]
        :param K:[1,8,5,64]
        :param V: [1,8,5,64]
        :param attn_mask:[1,8,5,5]
        :return:
        '''

        # test_data_1 = K.transpose(-1, -2) #[1,8,64,5]
        # test_data_2 = torch.matmul(Q, K.transpose(-1, -2)) #[1,8,5,5]
        # test_data_3 = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) #[1,8,5,5]

        # [batch_size, n_heads, len_q, len_k] [1,8,5,5]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]

        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.

        attn = nn.Softmax(dim=-1)(scores) #[1,8,5,5]
        context = torch.matmul(attn, V)  #[1,8,5,64]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads) #shape[512, 512]
        self.W_K = nn.Linear(d_model, d_k * n_heads) #shape[512, 512]
        self.W_V = nn.Linear(d_model, d_v * n_heads) #shape[512, 512]
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]

        # residual:[1,5,512]
        # batch_size:1
        residual, batch_size = Q, Q.size(0)

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # test_data_1 = self.W_Q(Q) #(1,5,512)
        # test_data_2 = self.W_Q(Q).view(batch_size, -1, n_heads, d_k) #[1,5,8,64]
        # test_data_3 = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  #[1,8,5,64]

        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k] [1,8,5,64]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # test_data_4 = attn_mask.unsqueeze(1) #[1,1,5,5]

        #[1,8,5,5]
        #重复head位置
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]

        # enc_outputs : [1,8,5,64]
        # enc_outputs : [1,8,5,5]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # test_data_5 = context.transpose(1, 2) #[1,5,8,64]
        # test_data_6 = context.transpose(1, 2).contiguous() #[1,5,8,64]
        # test_data_7 = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) #[1,5,512]

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]

        # test_data_8 = nn.Linear(n_heads * d_v, d_model)
        # test_data_9 = nn.Linear(n_heads * d_v, d_model)(context)

        # 线性转换
        output = nn.Linear(n_heads * d_v, d_model)(context) #wx+b [1,5,512]

        # test_data_10 =  nn.LayerNorm(d_model)
        # test_data_11 = output + residual
        # test_data_12 = nn.LayerNorm(d_model)(output + residual)

        # nn.LayerNorm 零均值归一化
        return nn.LayerNorm(d_model)(output + residual), attn # output: [batch_size x len_q x d_model]
        # return nn.Linear(n_heads * d_v, d_model)(output + residual), attn  # output: [batch_size x len_q x d_model]
        # return output + residual, attn  # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1) #d_ff:2048
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # test =  np.array(self.conv2.weight.detach().numpy())
        # test = test.reshape(test.shape[0], test.shape[1])
        # print(test)

        # torch reshape view
        # test_weight = self.conv2.weight.view(d_ff, d_model)
        # print(test_weight.size())


    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model] [1,5,512]

        # test_data_1 = inputs.transpose(1, 2) # [1,512,5]
        # test_data_2 = self.conv1(inputs.transpose(1, 2)) # [1,2048,5]
        # test_data_3 = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))  # [1,2048,5]

        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2))) #[1,2048,5]
        output = self.conv2(output).transpose(1, 2)  # [1,512,5] ---- > [1,5,512]

        out_emd = output + residual
        result = nn.LayerNorm(d_model)(out_emd)

        # return nn.LayerNorm(d_model)(output + residual)
        return result

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):

        # enc_outputs : [1,5,512]
        # enc_outputs : [1,8,5,5]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V

        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]

        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model) #[5,512]
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs): # enc_inputs : [batch_size x source_len]

        src_emb = self.src_emb(enc_inputs) #[1,5,512]
        pos_emb = self.pos_emb(torch.LongTensor([[1,2,3,4,0]])) #[1,5,512]
        enc_outputs = src_emb + pos_emb #[1,5,512]

        # [1,5,5]
        # [[[False,False,False,False,True],
        #   [False,False,False,False,True],
        #   [False,False,False,False,True],
        #   [False,False,False,False,True],
        #   [False,False,False,False,True]]]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        # test = np.array(enc_self_attn_mask)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]])) #dec_outputs:[1,5,512]

        # [1,5,5]
        # [[[False,False,False,False,True],
        #   [False,False,False,False,True],
        #   [False,False,False,False,True],
        #   [False,False,False,False,True],
        #   [False,False,False,False,True]]]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)

        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs) #[1,5,5]

        # 为了不重复计算
        # dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_self_attn_mask = (dec_self_attn_pad_mask.int() + dec_self_attn_subsequent_mask.int()).bool()

        # test_Data_1  = dec_self_attn_mask.data.numpy()[0]

        # tensor([[[False, False, False, False, True],
        #          [False, False, False, False, True],
        #          [False, False, False, False, True],
        #          [False, False, False, False, True],
        #          [False, False, False, False, True]]])
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        # tgt_vocab_size:[512,7]
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs) #enc_self_attns:[[1,8,5,5],[1,8,5,5]，[1,8,5,5]，[1,8,5,5]，[1,8,5,5]]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

model = Transformer()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def showgraph(attn):
    # test_data_1 = attn[-1]#[1,8,5,5]
    # test_data_2 = attn[-1].squeeze(0)#[8,5,5]
    # test_data_3 = attn[-1].squeeze(0)[0]#[5,5]

    attn = attn[-1].squeeze(0)[0] #[5,5] 取mutihead attention的第一个矩阵
    attn = attn.squeeze(0).data.numpy()

    print(attn)

    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()

for epoch in range(20):
    optimizer.zero_grad()
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    #outputs:[5,7]
    #enc_self_attns, dec_self_attns, dec_enc_attns:[[1,8,5,5],[1,8,5,5],[1,8,5,5],[1,8,5,5],[1,8,5,5],[1,8,5,5]] 6层
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

    test_data_1 = target_batch.contiguous().view(-1)

    loss = criterion(outputs, target_batch.contiguous().view(-1))
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

# Test
predict, _, _, _ = model(enc_inputs, dec_inputs)

# tensor([[-2.0008, 10.2820,  0.0470, -1.5183, -0.2899, -1.2601, -3.2617],
#         [-2.8301, -3.5005, 10.3755, -1.6809, -0.1652, -1.6929, -1.8569],
#         [-1.8316, -4.4791, -1.2669, 10.6068,  0.0435, -0.2920, -0.6336],
#         [-1.2841, -2.7595, -1.0153, -1.2816, 10.4920, -0.8805, -1.5439],
#         [-2.1730, -1.6929, -2.2679, -2.3579, -0.9170, -1.2622, 10.3462]])

predict = predict.data.max(1, keepdim=True)[1]
print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

print('first head of last state enc_self_attns')
showgraph(enc_self_attns)

print('first head of last state dec_self_attns')
showgraph(dec_self_attns)

print('first head of last state dec_enc_attns')
showgraph(dec_enc_attns)