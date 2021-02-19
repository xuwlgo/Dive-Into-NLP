import re
import torch
import gensim
import numpy as np
import pandas as pd
from hanziconv import HanziConv   # 繁简互换库
from collections import Counter   # 计数器
from torch.utils.data import Dataset


class LCQMC_Dataset(Dataset):

    def __init__(self, LCQMC_file, vocab_file, max_char_len):

        p, h, self.label = load_sentences(LCQMC_file)
        word2idx, _, _ = load_vocab(vocab_file)

        self.p_list, self.p_lengths, self.h_list, self.h_lengths = word_index(p, h, word2idx, max_char_len)
        self.p_list = torch.from_numpy(self.p_list).type(torch.long)  #返回的张量和ndarray共享同一内存
        self.h_list = torch.from_numpy(self.h_list).type(torch.long)
        self.max_length = max_char_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.p_list[idx], self.p_lengths[idx], self.h_list[idx], self.h_lengths[idx], self.label[idx]

def load_sentences(file, data_size=None):
    """
    输入：原始数据
    输出：每组数据分词后的结果
    """
    df = pd.read_csv(file, sep='\t', names=['sentence1', 'sentence2', 'label'])

    p = map(get_word_list, df['sentence1'].values[0:data_size])
    h = map(get_word_list, df['sentence2'].values[0:data_size])
    label = df['label'].values[0:data_size]

    return p, h, label

# 加载字典
def load_vocab(vocab_file):
    """
    输入：词汇表
    输出：词汇表中的单词及其索引
    """
    vocab = [line.strip() for line in open(vocab_file, encoding='utf-8').readlines()]  # 按行读取
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word, vocab

# word->index
def word_index(p_sentences, h_sentences, word2idx, max_char_len):
    """
    输入：分词后的数据、词汇索引表
    输出：索引值数据集（限定句子长度，且所有序列一样长）
    """
    p_list, p_length, h_list, h_length = [], [], [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        # 将句中单词切换为词汇表中的索引
        p = [word2idx[word] for word in p_sentence if word in word2idx.keys()]
        h = [word2idx[word] for word in h_sentence if word in word2idx.keys()]
        p_list.append(p)
        p_length.append(min(len(p), max_char_len))   # 限定句子长度
        h_list.append(h)
        h_length.append(min(len(h), max_char_len))
    # 将序列长度转变为一样长（不够补0，长了截断）
    p_list = pad_sequences(p_list, maxlen = max_char_len)
    h_list = pad_sequences(h_list, maxlen = max_char_len)
    return p_list, p_length, h_list, h_length

def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences
    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。填充补0
    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值
    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        # x[idx] = trunc
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x

#@TODO
def load_embeddings(embdding_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(embdding_path, binary=False)
    embedding_matrix = np.zeros((len(model.index2word) + 1, model.vector_size))
    #填充向量矩阵
    for idx, word in enumerate(model.index2word):
        embedding_matrix[idx + 1] = model[word]#词向量矩阵
    return embedding_matrix

def get_word_list(query):
    ''' 把句子按字分开，中文按字分，英文数字按空格, 大写转小写，繁体转简体'''
    query = HanziConv.toSimplified(query.strip())  # 去除字符串首尾空格与换行后进行繁简转换
    str_list = []

    # 用除字母，数字，下划线外的任意字符分割字符串
    regEx = re.compile('[\\W]+')#规则是除单词，数字，下划线外的任意字符串
    sentences = regEx.split(query.lower())

    # 用中文字符按字分割
    res = re.compile(r'([\u4e00-\u9fa5])')  # [\u4e00-\u9fa5]中文范围
    for sentence in sentences:
        if res.split(sentence) == None:   # 纯英文与数字
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]

def build_vocab(train_dir, vocab_size=5000):
    """构建词汇表(纯净版)"""
    df = pd.read_csv(train_dir, sep='\t', names=['sentence1', 'sentence2', 'label'])

    p = map(get_word_list, df['sentence1'].values)
    h = map(get_word_list, df['sentence2'].values)

    # 连接成字符串
    p2 = ''.join(''.join(i) for i in p)
    h2= ''.join(''.join(i) for i in h)
    text = p2 + h2

    # 统计字符出现次数
    counter = Counter(text)
    count_pairs = counter.most_common(vocab_size - 1)  #统计最常出现的字
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    open('./data/vocab.txt', mode='w', encoding='utf-8', errors='ignore').write('\n'.join(words) + '\n')

if __name__ == '__main__':
    build_vocab('./data/atec_nlp_sim_train_all.csv')