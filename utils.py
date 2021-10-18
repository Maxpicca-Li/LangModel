# -*- codeing = utf-8 -*-
# @Author: Maxpicca
# @Description: 工具包

from pyhanlp import *
import jieba
import time
from collections import Counter

# ==========导入hanlp类==========
# HanLP统计单个单词词频，使用DictionaryMaker
DictionaryMaker = SafeJClass('com.hankcs.hanlp.corpus.dictionary.DictionaryMaker')
# HanLP统计两个单词词频，使用NGramDictionaryMaker
NGramDictionaryMaker = SafeJClass('com.hankcs.hanlp.corpus.dictionary.NGramDictionaryMaker')
# HanLP语料库加载类
CorpusLoader = SafeJClass("com.hankcs.hanlp.corpus.document.CorpusLoader")  
# HanLP Word类
Word =  SafeJClass("com.hankcs.hanlp.corpus.document.sentence.word.Word")

def get_word_freq(corpus_path="./训练语料utf-8.txt", unigram_savename='unigram.txt', bigram_savename='bigram'):
    '''
    获取语料库词频\n
    :param corpus_path: 语料库路径
    :param unigram_savename: 单元，词频统计结果存储文件名
    :param bigram_savename: 双元，词频统计结果存储文件名
    :return:
    '''
    print("词频统计中...")
    # ==========读取语料==========
    sentences = CorpusLoader.convert2SentenceList(corpus_path)  # 返回List<List<IWord>>类型
    for sent in sentences:
        # 设置头、尾部
        sent[0] = Word("BOS", "begin")
        sent.addLast(Word("EOS", "end"))

    # ==========创建词频统计对象=========
    dict_maker = DictionaryMaker()
    ngram_maker = NGramDictionaryMaker()

    # =========统计词频=========
    for sent in sentences:
        # 一阶频次，只需要统计
        dict_maker.add(sent[0])
        for i in range(1, len(sent)):
            dict_maker.add(sent[i])
            ngram_maker.addPair(sent[i - 1], sent[i])

    # =========本地存储词频统计结果=========
    dict_maker.saveTxtTo(unigram_savename)
    ngram_maker.saveTxtTo(bigram_savename)


def my_read_dict(filepath):
    '''
    文件读取词频结果，返回字典索引\n
    :param filepath: 文件存储路径\n
    :return: 结果字典
    '''
    res_dict = {}
    with open(filepath,encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(" ")
            res_dict[line[0]] = int(line[-1])
    # num_list = list(unigram_counter.values())
    return res_dict

def get_unigram_list(sent):
    '''
    获取句子分词结果的单元列表\n
    :param sent:
    :return:
    '''
    temp_list = jieba.lcut(sent)
    unigram_list = ['BOS']+temp_list+['EOS']
    return unigram_list

if __name__=="__main__":
    corpus_path = "./训练语料utf-8.txt"    # 语料库Corpus所在路径
    unigram_savename = 'unigram.txt'
    bigram_savename = 'bigram'
    get_word_freq(corpus_path,unigram_savename,bigram_savename)