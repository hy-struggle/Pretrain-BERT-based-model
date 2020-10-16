# /usr/bin/env python
# coding=utf-8
"""generate corpus"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import re

import logging
from utils import set_logger

set_logger()

DATA_SRC = Path('../data_src')
NGRAM_CUT_OFF = 3


def _generate_vocab_ngram(merge_df):
    # generate vocab.txt
    char_list = list(merge_df['char'])
    char_str_list = [str(c) for char in char_list for c in eval(char)]
    char_set = set(char_str_list)
    # 生成词典映射
    w2i = {"[PAD]": 0, "[CLS]": 1, "[MASK]": 2, "[SEP]": 3,
           "[UNK]": 4, '机': 5, '人': 6}
    count = 4
    for c in char_set:
        if c not in w2i.keys():
            w2i[c] = count
            count += 1
    with open("./vocab.txt", 'w', encoding='utf-8') as fw:
        for key in w2i.keys():
            fw.write(key + '\n')

    # generate ngram.txt
    word_list = list(merge_df['word'])
    word_str_list = [str(w) for word in word_list for w in eval(word) if '-' in w]
    counter = Counter(word_str_list).most_common()
    with open('./ngram.txt', 'w', encoding='utf-8') as fn:
        for cou in counter:
            if cou[1] >= NGRAM_CUT_OFF:
                fn.write(f'{cou[0]},{cou[1]}\n')


def generate_src_corups():
    # read data_src
    train_df = pd.read_excel(DATA_SRC / "train.xlsx")
    test_df = pd.read_excel(DATA_SRC / "public_test.xlsx")
    # 修改列名
    test_df.rename(columns={'catgory': 'category'}, inplace=True)
    merge_df = train_df.append(test_df)

    logging.info('generate vocab and ngram...')
    # 生成vocab字表和ngram词表
    _generate_vocab_ngram(merge_df)
    logging.info('done')

    with open('./corpus_src.txt', 'w', encoding='utf-8') as c_src:
        # init
        total_len = 0
        max_len = 0

        # num of ids: 13781
        # 一段对话对应一段语料
        for comu_id in tqdm(merge_df['id'].unique(), desc='NUM OF IDS', unit='id'):
            # 取出同一段对话
            merge_id = merge_df[merge_df['id'] == comu_id]
            char_list = list(merge_id['char'])
            category_list = list(merge_id['category'])

            char_str_list = []
            # 取一条语句
            for char in char_list:
                char = eval(char)
                char_str_list.append(' '.join([str(c) for c in char]))
            # 融合一段对话，语句间用特殊token分割
            char_line = ''
            for idx, category in enumerate(category_list):
                # 0：客户；1：机器人
                if category == 1:
                    char_line += char_str_list[idx] + ' 机 '
                else:
                    char_line += char_str_list[idx] + ' 人 '
            c_src.write(char_line + '\n')

            # 记录总长度和最大长度
            total_len += len(char_line.split(' '))
            if len(char_line.split(' ')) > max_len:
                max_len = len(char_line.split(' '))

    """
    序列平均长度：264
    序列最大长度：1510
    """
    print(f'序列平均长度：{total_len // len(merge_df["id"].unique())}')
    print(f'序列最大长度：{max_len}')


def _split_text(text, max_len, split_pat, greedy=True):
    """文本分片
    将超过长度的文本分片成多段满足最大长度要求的最长连续子文本
    约束条件：1）每个子文本最大长度不超过max_len；
             2）所有的子文本的合集要能覆盖原始文本。
    Arguments:
        text {str} -- 原始文本
        max_len {int} -- 最大长度

    Keyword Arguments:
        split_pat {str or re pattern} -- 分割符模式 (default: {SPLIT_PAT})
        greedy {bool} -- 是否选择贪婪模式 (default: {False})
                         贪婪模式：在满足约束条件下，选择子文本最多的分割方式

    Returns:
        tuple -- 返回子文本列表以及每个子文本在原始文本中对应的起始位置列表
    """
    if len(text.split(' ')) <= max_len:
        return [text], [0]

    segs = re.split(split_pat, text)
    segs_list = []
    # 收集单个子片段
    for i in range(0, len(segs) - 1, 2):
        segs_list.append(segs[i] + segs[i + 1])
    if segs[-1]:
        segs_list.append(segs[-1])

    num_segs = len(segs_list)
    seg_lens = [len(s.strip().split(' ')) for s in segs_list]
    # 所有满足约束条件的最长子片段组合
    alls = []
    # 获取最长子片段组合
    for i in range(num_segs):
        length = 0
        sub = []
        for j in range(i, num_segs):
            if length + seg_lens[j] <= max_len or not sub:
                sub.append(j)
                length += seg_lens[j]
            else:
                break
        alls.append(sub)

        # 如果囊括了所有子片段
        if j == num_segs - 1:
            # 如果最后一个片段没加进去
            if sub[-1] != j:
                alls.append(sub[1:] + [j])
            else:
                break

    if len(alls) == 1:
        return [text], [0]

    # 贪婪模式返回所有子文本
    if greedy:
        sub_texts = [''.join([segs_list[i] for i in sub]) for sub in alls]
        return sub_texts


def generate_split_corups(max_len, split_pat):
    """获取分割后的语料"""
    text_list = []
    with open("./corpus_src.txt", 'r',
              encoding='utf-8') as f_src:
        for line in f_src:
            char_list = line.strip().split(' ')
            if len(char_list) > max_len:
                # 切割文本
                split_list = _split_text(line.strip(), max_len=max_len, split_pat=split_pat, greedy=True)
                for s in split_list:
                    text_list.append(s.strip().split(' '))
            else:
                text_list.append(char_list)
            # 加空行
            text_list.append([])

    # 生成分割后的语料
    with open("./corpus_256.txt", 'w',
              encoding='utf-8') as f_split:
        for line in text_list:
            f_split.write(' '.join(line) + '\n')


if __name__ == '__main__':
    MAX_LEN = 256
    SPLIT_PAT = '([机人]”?)'
    logging.info('generate src corpus...')
    # 生成原始语料和词表
    generate_src_corups()
    logging.info('done')
    logging.info('do split...')
    # 生成分割后的语料
    generate_split_corups(split_pat=SPLIT_PAT, max_len=MAX_LEN)
    logging.info('done')
