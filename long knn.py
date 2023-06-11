from math import sqrt
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from time import *
import jieba
import codecs
from sklearn.model_selection import train_test_split
from collections import defaultdict  # 导入collections用于创建空白词典
#pip install pyltp-0.2.1-cp39-cp39m-win_amd64.whl
def seg_word(sentence):
    seg_list = jieba.cut(sentence)
    seg_result = []
    for word in seg_list:
        seg_result.append(word)
        stopwords = set()
        stopword = codecs.open('data/stopwords.txt', 'r',
                               encoding='utf-8')  # 加载停用词
    for word in stopword:
        stopwords.add(word.strip())
    stopword.close()
    return list(filter(lambda x: x not in stopwords, seg_result))

def sort_word(word_dict):
    sen_file = open('data/BosonNLP_sentiment_score.txt', 'r+',
                    encoding='utf-8')  # 加载Boson情感词典
    sen_list = sen_file.readlines()
    sen_dict = defaultdict()  # 创建词典
    for s in sen_list:
        s = re.sub('\n', '', s)  # 去除每行最后的换行符
        if s:
            # 构建以key为情感词，value为对应分值的词典
            sen_dict[s.split(' ')[0]] = s.split(' ')[1]
    not_file = open('data/否定词.txt', 'r+',
                    encoding='utf-8')  # 加载否定词词典
    not_list = not_file.readlines()
    for i in range(len(not_list)):
        not_list[i] = re.sub('\n', '', not_list[i])
    degree_file = open('data/程度副词（中文）.txt', 'r+',
                       encoding='utf-8')  # 加载程度副词词典
    degree_list = degree_file.readlines()
    degree_dic = defaultdict()
    for d in degree_list:
        d = re.sub('\n', '', d)
        if d:
            degree_dic[d.split(' ')[0]] = d.split(' ')[1]
    sen_file.close()
    degree_file.close()
    not_file.close()
    sen_word = dict()
    not_word = dict()
    degree_word = dict()
    # 分类
    for word in word_dict.keys():
        if word in sen_dict.keys() and word not in not_list and word not in degree_dic.keys():
            sen_word[word_dict[word]] = sen_dict[word]  # 情感词典中的包含分词结果的词
        elif word in not_list and word not in degree_dic.keys():
            not_word[word_dict[word]] = -1  # 程度副词词典中的包含分词结果的词
        elif word in degree_dic.keys():
            # 否定词典中的包含分词结果的词
            degree_word[word_dict[word]] = degree_dic[word]
    return sen_word, not_word, degree_word  # 返回分类结果

def list_to_dict(word_list):
    data = {}
    for x in range(0, len(word_list)):
        data[word_list[x]] = x
    return data
def setiment(sentence):
    # 对文本进行分词和去停用词，去除跟情感词无关的词语
    seg_list = seg_word(sentence)
    # 对分词结果进行分类，找出其中的情感词、程度副词和否定词
    sen_word, not_word, degree_word = sort_word(list_to_dict(seg_list))
    return sen_word, not_word, degree_word
    # 计算并汇总情感词的得分
    #score = socre_sentiment(sen_word, not_word, degree_word, seg_list)
    #return seg_list, sen_word, not_word, degree_word

def delete_stopwords(word_lst):
    text_collection = []
    new_word_lst = []
    for j in range(len(word_lst)):
        lst1 = []
        sen_word, not_word, degree_word=setiment(word_lst[j])
        text_collection.append (sen_word)
        new_word_lst.append(jieba.cut(word_lst[j]))
    return new_word_lst,text_collection
#new_word是词典，text_collection是分词后的列表
# 将文本向量化
def text_CountVectorizer(new_word_lst,text_collection):
    num_collection = len(text_collection)
    num_text = len(new_word_lst)
    array_text = np.zeros((num_text,num_collection), dtype=int)
    for i in range(num_text):
        for j in range(len(new_word_lst[i])):
            if new_word_lst[i][j] in text_collection:
                place = text_collection.index(new_word_lst[i][j])
                array_text[i][place] += 1
    return array_text

# 皮尔森相关系数法
def Person(array_text,text_collection, len_negative, len_positive):
    yuzhi = 0.06
    delete_lst = []
    array_negative = np.zeros(len_negative)
    array_positive = np.ones(len_positive)
    vb = np.hstack((array_negative, array_positive))
    num = array_text.shape[1]
    for i in range(num):
        vc = array_text[:,i]
        co = np.mean(np.multiply((vc - np.mean(vc)), (vb - np.mean(vb)))) / (np.std(vb) * np.std(vc))
        if abs(co) < yuzhi:
            delete_lst.append(i)
    array_text = np.delete(array_text, delete_lst, axis=1)
    lst_word = [text_collection[i] for i in range(len(text_collection)) if i not in delete_lst]
    return array_text,lst_word



# 将文本文件以数组的形式输出
def input_test(where_from,num):
    s1 = open(where_from, 'r',encoding='utf-8')
    X = s1.readlines()
    return X

def KNN(array_train, array_text,k):
    negative_correct = 0 ; positive_correct =0
    train_len = len(array_train)
    text_len = len(array_text)
    for i in range(text_len):     # 计算每一组数的距离
        dis = []
        for j in range(train_len):
            temp_dis = sqrt(np.sum((array_train[j] - array_text[i])**2))  # 计算距离
            dis.append(temp_dis)
        dis = np.array(dis)
        sort_id = dis.argsort()
        print(i)#返回原数组的下标
        dic = {'negative' : 0, 'positive' : 0}
        # 对于前排在前面k个元素进行统计，从而判断是negative还是positive。
        for w in range(k):
            num = sort_id[w]  # 为对应的标签记数
            if num <= train_len/2:
                dic['negative'] += 1
            else:
                dic['positive'] += 1
        if dic['negative'] > dic['positive'] and i < text_len/2 :
            negative_correct += 1
        if dic['negative'] < dic['positive'] and i >= text_len/2:
            positive_correct += 1
    print(negative_correct,positive_correct)
    return negative_correct/(text_len/2), positive_correct/(text_len/2)


def main():
    begin_time = time()
    pos=input_test('data/pos.txt',2142)
    neg=input_test('data/neg.txt',947)
    postrain, postest = train_test_split(np.array(pos), test_size=0.25)
    negtrain, negtest = train_test_split(np.array(neg), test_size=0.25)
    train_lst = list(negtrain) + list(postrain)
    text_lst = list(negtest) + list(postest)
    # 输出训练集的向量组
    lst1, word_lst_train = delete_stopwords(train_lst)
    print(len(word_lst_train[1]))
    array_train = text_CountVectorizer(lst1, word_lst_train)
    array_train, word_train = Person(array_train, word_lst_train, len(postrain), len(negtrain))
    print('训练集的特征数目是：', len(word_train))
    end_time_train = time()
    run_time_train = end_time_train - begin_time
    print('训练的时间是：', str(run_time_train)+'s')
    # 输出测试集的向量组
    lst2, word_lst_text = delete_stopwords(text_lst)
    array_text = text_CountVectorizer(lst2, word_train)    # 这里的向量化是根据训练集的词库生成的
    k = 25
    negative_correct, positive_correct = KNN(array_train, array_text,k)
    print('此时的k值是',k)
    end_time_text = time()
    run_time_text = end_time_text - end_time_train
    print('测试的时间是：', str(run_time_text)+'s')
    run_time = end_time_text - begin_time
    print('总运行的时间是', str(run_time)+'s')
    print('消极和积极文本分析的正确率分别是：',negative_correct, positive_correct)
    print('总正确率是',end = '')
    return (negative_correct+positive_correct)/2

if __name__ == '__main__':
    print(main())

