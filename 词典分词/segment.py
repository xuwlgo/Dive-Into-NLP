# Author:wl
# 词典分词——基于规则

def fully_segment(text,dict):
    """朴素完全切分"""
    word_lists=[]
    for i in range(len(text)):
        for j in range(i+1,len(text)+1):
            word = text[i:j]
            if word in dict:
                word_lists.append(word)
    return word_lists

def forward_segment(text,dict):
    """正向最长匹配"""
    word_lists=[]
    i = 0
    while i<len(text):
        longest_word = text[i]
        for j in range(i+1,len(text)+1):
            word = text[i:j]
            if word in dict:
                if len(longest_word) < len(word):
                    longest_word = word
        word_lists.append(longest_word)
        i += len(longest_word)
    return word_lists

def backward_segment(text,dict):
    """逆向最长匹配"""
    word_lists=[]
    i = len(text)-1
    while i>=0:
        longest_word = text[i]
        for j in range(0,i):
            word = text[j:i+1]
            if word in dict:
                if len(word) > len(longest_word):
                    longest_word = word
        word_lists.insert(0,longest_word) # 前插，最先出现的词应在结果的最后
        i -= len(longest_word)
    return word_lists

def bidirectional_segment(text,dict):
    """双向最长匹配"""
    def count_single_char(word_lists):
        """统计单字成词的个数"""
        return sum(1 for word in word_lists if len(word)==1)

    f = forward_segment(text,dict)
    b = backward_segment(text,dict)
    if len(f)<len(b):
        return f
    elif len(f)>len(b):
        return b
    else:
        if count_single_char(f) < count_single_char(b):
            return f
        else:
            return b

if __name__=='__main__':

    dt = {'空调':1,'调和':1,'风扇':1,
          '空':1,'调':1,'和':2,'风':1,'扇':1}

    sentence = '空调和风扇'

    # result = fully_segment(sentence,dt)
    # result = forward_segment(sentence,dt)
    # result = backward_segment(sentence,dt)
    result = bidirectional_segment(sentence,dt)

    print(result)