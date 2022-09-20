import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler

from queue import PriorityQueue
import operator
import random


class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input = [batch size, sequence len, vocab size]
        embedding = self.embedding(input)
        outputs, hidden = self.rnn(self.dropout(embedding))
        # outputs = [batch size, sequence len, hid dim * directions]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        # outputs 是最上層RNN的輸出
        
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
        super().__init__()
        self.cn_vocab_size = cn_vocab_size
        self.hid_dim = hid_dim * 2
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_vocab_size, emb_dim)
        self.isatt = isatt
        self.attention = Attention(hid_dim)
        # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
        # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
        self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
        # self.input_dim = emb_dim
        self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout = dropout, batch_first=True)
        self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size, vocab size]
        # hidden = [batch size, n layers * directions, hid dim]
        # Decoder 只會是單向，所以 directions=1
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, emb dim]
        if self.isatt:
            attn_weights = self.attention(encoder_outputs, hidden)
            
            # 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化                 
            # Multiplying the Attention weights with encoder outputs to get the context vector
            context_vector = torch.bmm(attn_weights, encoder_outputs)
            
            # Concatenating context vector with embedded input word
            embedded = torch.cat([embedded, context_vector], dim=2).cuda()
                
        output, hidden = self.rnn(embedded, hidden)
        # output = [batch size, 1, hid dim]
        # hidden = [num_layers, batch size, hid dim]

        # 將 RNN 的輸出轉為每個詞出現的機率
        output = self.embedding2vocab1(output.squeeze(1))
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)       
        return prediction, hidden




class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim
        
        self.enc_dim = self.dec_hid_dim = hid_dim * 2
        self.attn = nn.Linear(self.enc_dim + self.dec_hid_dim, self.dec_hid_dim)
        self.v = nn.Linear(self.dec_hid_dim, 1, bias = False)
        
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)
  
    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs = [batch size, sequence len, hid dim * directions]
        # decoder_hidden = [num_layers, batch size, hid dim]
        # 一般來說是取 Encoder 最後一層的 hidden state 來做 attention
        ########
        # TODO #
        ########
        attention=None  

        # Linear
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        #hidden = [batch size, dec hid dim]
        decoder_hidden = decoder_hidden.permute(1, 0, 2)[:, -1:, :]        
        
        #repeat decoder hidden state src_len times
        #hidden = [batch size, src len, dec hid dim] 
        hidden = decoder_hidden.repeat(1, src_len, 1)                    
      
        # encoder_outputs = [batch size, sequence len, hid dim * directions]        
        # energy = [batch size, src len, dec hid dim]      
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))                         
        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]  
        attention = F.softmax(attention, dim=1)    # torch.Size([60, 50])  
        attention = attention.unsqueeze(1)         # torch.Size([60, 1, 50])  

        # Scaled Dot Product Attention    
        # torch.Size([60, 1, 50])  
        attention2 = torch.bmm(decoder_hidden / (self.dec_hid_dim ** 0.5) , 
                              encoder_outputs.permute(0, 2, 1))     
        attention2 = F.softmax(attention2, dim=2)    # torch.Size([60, 1, 50])  
        
        # cosine similarity attention
        attention3 = self.cos(hidden, encoder_outputs)
        attention3 = F.softmax(attention3, dim=1)    # torch.Size([60, 50]) 
        attention3 = attention3.unsqueeze(1) 
                                       
        return attention

class BeamSearchNode(object):
    # https://github.com/312shan/Pytorch-seq2seq-Beam-Search/blob/master/model.py
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward  # 注意这里是有惩罚参数的，参考恩达的 beam-search

    def __lt__(self, other):
        return self.leng < other.leng  # 这里展示分数相同的时候怎么处理冲突，具体使用什么指标，根据具体情况讨论

    def __gt__(self, other):
        return self.leng > other.leng
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers, \
                "Encoder and decoder must have equal number of layers!"
        
            
    def forward(self, input, target, teacher_forcing_ratio):
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)       
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            # 決定是否用正確答案來做訓練
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
            input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def greedy(self, input, target):       
        # 此函式的 batch size = 1  
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        batch_size = input.shape[0]
        input_len = input.shape[1]        # 取得最大字數
        vocab_size = self.decoder.cn_vocab_size              

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []     
        
        for t in range(1, input_len):
            # Greedy
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # 將預測結果存起來
            outputs[:, t] = output
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            input = top1
            preds.append(top1.unsqueeze(1))           
        ##############################################################  
        preds = torch.cat(preds, 1)                    # preds:  torch.Size([1, 49])            
        return outputs, preds
 
    def inference(self, input, target, BOS_token, EOS_token, vocab):
        ########
        # TODO #
        ########
        # 在這裡實施 Beam Search
        # 此函式的 batch size = 1  
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        beam_width = 8
        batch_size = input.shape[0]
        input_len = input.shape[1]        # 取得最大字數
        vocab_size = self.decoder.cn_vocab_size              

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []    
        ##########################################################   
        # Start with the start of the sentence token
        decoder_input = input

        # Number of sentence to generate
        topk = 1 # how many sentence do you want to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes)) # number_required = topk

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(hidden, None, input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1
        
        for t in range(1, input_len):           
        # Beam Search
            # give up when decoding takes too long
            if qsize > 2000: 
                print("qsize break", end="\r")
                break

            # fetch the best node
            score, n = nodes.get()
            # print('--best node seqs len {} '.format(n.leng), end="\r")
            decoder_input = n.wordid
            hidden = n.h

            # print("{} : {} ".format(vocab[str(decoder_input.cpu().numpy()[0])], 
            #                     str(decoder_input.cpu().numpy()[0]) ), 
            #                     end="\r")

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder          
            decoder_output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            # decoder_output : torch.Size([1, 3805]) 


            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []           

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(-1)
                log_p = log_prob[0][new_k].item()

                # print("decoded_t : {} ".format(vocab[str(decoded_t.cpu().numpy()[0])] ),  end="\r") 

                node = BeamSearchNode(hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1
            # print('qsize: {} '.format(qsize), end="\r")
        
        ##############################################################        
        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]            
        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):              
            wordid = n.wordid.unsqueeze(0)
            utterance = []
            utterance.append(wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                wordid = n.wordid.unsqueeze(0)
                utterance.append(wordid)

            utterance = utterance[::-1]
            utterance = torch.cat(utterance, 1) 
            utterances.append(utterance)

        preds = utterances[0]     
  
        return outputs, preds
