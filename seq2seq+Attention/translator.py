from models import *
from helper import *
from read_data import *
import random
from masked_cross_entropy import *
import os
from bleu import *
from sklearn.model_selection import train_test_split


# 训练的超参数
clip = 50.0
# teacher_forcing_ratio = 0.5
learning_rate = 0.0001
decoder_learning_ratio = 5.0

class Translator():
    def __init__(self,attention_mode = DOT_MODEL):
        self.attention_img_count = 1
        pairs, input_lang, output_lang = get_trim_data(min_count=5)

        X_train, X_valid = train_test_split(pairs, test_size = TEST_SPLIT, random_state = RNG_SEED)
        self.valid_pairs =X_valid
        self.pairs = X_train

        self.input_lang = input_lang
        self.output_lang = output_lang

        # 模型的配置
        attn_model = attention_mode
        self.attn_model = attn_model
        hidden_size = 500
        n_layers = 2
        dropout = 0.1
        learning_rate = 0.0001
        decoder_learning_ratio = 5.0
        self.batch_size = 100

        # 初始化模型
        self.encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers, dropout=dropout)
        self.decoder = LuongAttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout=dropout)

        # 初始化optimizers和criterion
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
        self.criterion = nn.CrossEntropyLoss()

        # 把模型放到GPU上
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def random_batch(self):
        input_seqs = []
        target_seqs = []

        # 随机选择pairs
        for i in range(self.batch_size):
            pair = random.choice(self.pairs)
            input_seqs.append(indexes_from_sentence(self.input_lang, pair[0]))
            target_seqs.append(indexes_from_sentence(self.output_lang, pair[1]))

        # 把输入和输出序列zip起来，通过输入的长度降序排列，然后unzip
        seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs = zip(*seq_pairs)

        # 对输入和输出序列都进行padding。
        input_lengths = [len(s) for s in input_seqs]
        input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]

        target_lengths = [len(s) for s in target_seqs]
        target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

        # padding之后的shape是(batch_size x max_len)，我们需要把它转置成(max_len x batch_size)
        input_var = torch.LongTensor(input_padded).transpose(0, 1)
        target_var = torch.LongTensor(target_padded).transpose(0, 1)

        if USE_CUDA:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        return input_var, input_lengths, target_var, target_lengths

    def train(self,input_batches, input_lengths, target_batches, target_lengths):
        # 梯度清空
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss = 0  # Added onto for each word

        # 进行encoding
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)

        # 准备输入和输出变量
        decoder_input = torch.LongTensor([SOS_token] * self.batch_size)
        decoder_hidden = encoder_hidden
        # decoder_hidden = decoder_hidden.contiguous()

        max_target_length = max(target_lengths)
        all_decoder_outputs = torch.zeros(max_target_length, self.batch_size, self.decoder.output_size)

        if USE_CUDA:
            decoder_input = decoder_input.cuda()
            all_decoder_outputs = all_decoder_outputs.cuda()

        # 解码
        for t in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attn = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            all_decoder_outputs[t] = decoder_output
            decoder_input = target_batches[t]  # 当前输出是下一个时刻的输入。

        # 计算loss和反向计算梯度
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
            target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
            target_lengths,
            USE_CUDA
        )
        loss.backward()

        # clip梯度
        nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # 更新参数
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()

    def evaluate(self,input_seq, max_length=MAX_LENGTH):
        input_seqs = [indexes_from_sentence(self.input_lang, input_seq)]
        input_lengths = [len(sen) for sen in input_seqs]
        input_batches = torch.LongTensor(input_seqs).transpose(0, 1)

        if USE_CUDA:
            input_batches = input_batches.cuda()

        # 预测的时候不需要更新参数
        self.encoder.eval()
        self.decoder.eval()

        # encoding
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)

        # 创建开始的SOS
        decoder_input = torch.LongTensor([SOS_token])  # SOS
        decoder_hidden = encoder_hidden
        # decoder_hidden = decoder_hidden.contiguous()

        if USE_CUDA:
            decoder_input = decoder_input.cuda()

        # Store output words and attention states
        decoded_words = []
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

        # Run through decoder
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            #  decoder_attention (B x 1 x S)
            decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0].item()
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.output_lang.index2word[ni])

            # Next input is chosen word
            decoder_input = torch.LongTensor([ni])
            if USE_CUDA: decoder_input = decoder_input.cuda()

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]

    def show_attention(self,input_sentence, output_words, attentions):
        '''
        :param input_sentence: string
        :param output_words: string
        :param attentions: [output_words x input_sentence_length]
        :return:
        '''
        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        words = input_sentence.split(' ')
        ax.set_xticklabels([''] + words + ['<EOS>'])
        ax.set_yticklabels([''] + output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # plt.show()
        plt.savefig(os.path.join(ATTENTION_IMG_PATH,self.attn_model,str(self.attention_img_count))+'.jpg')


        # we only save the past 5 attention images.
        self.attention_img_count+=1
        if self.attention_img_count > 5:
            self.attention_img_count = 1
        plt.close()


    def evaluate_and_show_attention(self,input_sentence, target_sentence=None):
        output_words, attentions = self.evaluate(input_sentence)
        output_sentence = ' '.join(output_words)
        print('>', input_sentence)
        if target_sentence is not None:
            print('=', target_sentence)
        print('<', output_sentence)

        self.show_attention(input_sentence, output_words, attentions)



    def evaluate_oneSample(self,is_random = True):
        '''
        :param is_random: whether we randomly select a pair for evaluation
        :return:
        '''
        if is_random:
            [input_sentence, target_sentence] = random.choice(self.pairs)
            self.evaluate_and_show_attention(input_sentence, target_sentence)
        else:
            [input_sentence, target_sentence] = self.pairs[100]
            self.evaluate_and_show_attention(input_sentence, target_sentence)

    def evaluate_unseen(self,input_sentence):
        self.evaluate_and_show_attention(input_sentence)

    def save_model(self):
        torch.save(self.encoder, os.path.join(MODEL_SAVE_PATH,self.attn_model,ENCODER_MODEL))
        torch.save(self.decoder, os.path.join(MODEL_SAVE_PATH,self.attn_model,DECODER_MODEL))


    def load_model(self):
        self.encoder = torch.load(os.path.join(MODEL_SAVE_PATH,self.attn_model,ENCODER_MODEL))
        self.encoder.eval()
        self.decoder = torch.load(os.path.join(MODEL_SAVE_PATH,self.attn_model,DECODER_MODEL))
        self.decoder.eval()


    def evaluate_bleu(self):
        candidates = []
        translations = []
        for input_sentence, target_sentence in self.valid_pairs:
            output_words, _ = self.evaluate(input_sentence)
            output_sentence = ' '.join(output_words)
            candidates.append(target_sentence)
            translations.append(output_sentence)

        return get_bleu(translations, candidates)





