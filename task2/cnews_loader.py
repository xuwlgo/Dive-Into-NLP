import sys
from collections import Counter

import numpy as np
from tensorflow import keras as kr
import os


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename,encoding='utf-8') as f:
        for line in f:
            label, content = line.split('\t')
            contents.append(list(content))
            labels.append(label)
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)  #统计最常出现的字
    # print(count_pairs)
    words, _ = list(zip(*count_pairs))  # 得出最常出现的词words
    # print(words)
    # 添加一个 <PAD> 来将所有文本填补为同一长度
    words = ['<PAD>'] + list(words)

    with open(vocab_dir,'w',encoding='utf-8') as f:
        f.write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    words = []
    with open(vocab_dir,encoding='utf-8') as fp:
        for word in fp:
            word,_=word.split('\n')
            words.append(word)
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    # 转换为{类别: id}表示
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本填充为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    # print('inter')
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


# if __name__ == '__main__':
#     base_dir = 'data'
#     train_dir = os.path.join(base_dir, 'cnews.train.txt') # 训练集 50000
#     test_dir = os.path.join(base_dir, 'cnews.test.txt')   # 测试集 10000
#     val_dir = os.path.join(base_dir, 'cnews.val.txt')     # 验证集 5000 
#     vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt') # 词汇表
#     categories, cat_to_id = read_category()  # 将分类目录固定

#     words, word_to_id = read_vocab(vocab_dir) # 读取词汇表，转换为{词：id}表示
#     x_val, y_val = process_file(val_dir, word_to_id, cat_to_id,600)
#     print('start')
#     batch_val = batch_iter(x_val, y_val, 64)
#     f = 0
#     for i,j in batch_val:
#         f+=1
#     print(f)

