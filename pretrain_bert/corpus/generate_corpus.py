# /usr/bin/env python
# coding=utf-8
"""generate corpus"""
from pathlib import Path
import re

import logging
from utils import set_logger

set_logger()

DATA_SRC = Path('../data_src')


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
        return [text]

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
        return [text]

    # 贪婪模式返回所有子文本
    if greedy:
        sub_texts = [''.join([segs_list[i] for i in sub]) for sub in alls]
        return sub_texts


def generate_split_corups(max_len, split_pat):
    with open(DATA_SRC / 'military.json', 'r', encoding='utf-8') as f:
        # 获取过滤后的文本
        sentences = [' '.join(list(eval(line)['简介'].strip().replace('\n\n', '').replace('\n', '').replace(' ', ''))) for line in f if len(eval(line)['简介'].strip()) != 0]
    with open(DATA_SRC / 'sentences.txt', 'r', encoding='utf-8') as f:
        sentences += [line.strip() for line in f]

    text_list = []
    for line in sentences:
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

    # 写入
    with open(f"./corpus_{max_len}.txt", 'w', encoding='utf-8') as f_split:
        for line in text_list:
            f_split.write(' '.join(line) + '\n')


if __name__ == '__main__':
    MAX_LEN = 256
    SPLIT_PAT = '([，。！]+)'
    logging.info('generate split corpus...')
    # 生成原始语料和词表
    generate_split_corups(max_len=MAX_LEN, split_pat=SPLIT_PAT)
    logging.info('done')
