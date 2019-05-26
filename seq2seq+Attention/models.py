import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
import torch.nn.functional as F



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # 注意：和前面的实现不同，这里没有时刻t的for循环，而是一次输入GRU直接计算出来

        embedded = self.embedding(input_seqs)  # max_length , b , hidden_size
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        outputs, hidden = self.gru(packed,
                                   hidden)  # outputs.data (non_empty, hidden_size x layers) / hidden (layesr x direction , bs, hidden_size)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            outputs)  # unpack -> max_length , b , hidden_size x layers

        hidden = self._cat_directions(hidden)

        return outputs, hidden  # (num_layers, batch_size, hidden_size * num_directions)

    def _cat_directions(self, hidden):
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
        """

        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size * 2)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 4, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
            torch.nn.init.normal_(self.v, mean=0, std=1)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # 创建变量来存储注意力能量
        attn_energies = torch.zeros(this_batch_size, max_len)  # B x S

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        # 计算
        for b in range(this_batch_size):
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # softmax并且resize成B x 1 x S
        return F.softmax(attn_energies,dim = 1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = torch.mm(hidden, encoder_output.view(-1, 1)).item()
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.mm(hidden, energy.view(-1, 1)).item()
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.tanh(energy)
            energy = torch.mm(self.v, energy.view(-1, 1)).item()
            return energy


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # 保存变量
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 定义网络层
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size * 2, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 4, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # 选择注意力计算方法
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # 注意：我们encoder一次计算所有时刻的数据，但是decoder我们目前还是一次计算一个时刻的（但是是一个batch）
        # 因为Teacher Forcing可以一次计算但是Random Sample必须逐个计算
        # 得到当前输入的embedding

        # last_hidden / layers ,b ,hidden_size x directions
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N

        # 计算gru的输出和新的隐状态，输入是当前词的embedding和之前的隐状态。
        rnn_output, hidden = self.gru(embedded,
                                      last_hidden)  # 1 x b x (num_directions = 1) * (hiddensize * 2)  / num_layers*num_directions=1 x b x (hidden_size * 2)
        # rnn_output : 1 x b x  (hiddensize * 2)  / 2 x b x (hidden_size * 2)
        # 根据当前的RNN状态和encoder的输出计算注意力。
        # 根据注意力计算context向量
        attn_weights = self.attn(rnn_output, encoder_outputs)  # B x 1 x S

        context = attn_weights.bmm(encoder_outputs.transpose(0,
                                                             1))  # (B x 1 x S)  bmm (B , max_length , hidden_size x layers) -> B x 1 x hidden_size x layers
        # 把gru的输出和context vector拼接起来
        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # 预测下一个token，这里没有softmax，只有计算loss的时候才需要。
        output = self.out(concat_output)

        # 返回最终的输出，GRU的隐状态和attetion（用于可视化）
        return output, hidden, attn_weights