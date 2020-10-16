# /usr/bin/env python
# coding=utf-8
"""generate corpus"""
import json
import shelve
from tempfile import TemporaryDirectory
from pathlib import Path
from tqdm import tqdm, trange
import collections
from argparse import ArgumentParser
from random import random, randrange, randint, shuffle, choice
import re

import numpy as np
import sys
import jieba

sys.path.insert(0, '../')
from NEZHA import BertTokenizer
from utils import Params
import os

parser = ArgumentParser()
parser.add_argument("--max_seq_len", type=int, default=512)
parser.add_argument("--max_word_len", type=int, default=7,
                    help='检索词时的最大长度')

parser.add_argument("--do_whole_word_mask", action="store_true",
                    help="Whether to use whole word masking rather than per-WordPiece masking.")
parser.add_argument("--remove_nsp", action="store_true",
                    help="是否移除随机NSP任务负样本")

parser.add_argument("--reduce_memory", action="store_true",
                    help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")
parser.add_argument("--epochs_to_generate", type=int, default=3,
                    help="Number of epochs of data to pregenerate")
parser.add_argument("--short_seq_prob", type=float, default=0.1,
                    help="Probability of making a short sentence as a training example")
parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                    help="Probability of masking each token for the LM task")
parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                    help="Maximum number of tokens to mask in each sequence")
args = parser.parse_args()

CORPUS_PATH = Path('../corpus')


class DocumentDatabase():
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def get_wwm_tokens(segment, word_length=args.max_word_len):
    """输入一句话，返回一句经过处理的话
    中文wwm，将被分开的词，将上特殊标记("##")，使得后续处理模块能够知道哪些字是属于同一个词的。
    :param segment (List[str]): tokens
    :param cws_set (Tuple[str] or List[str]): 词表
    :param word_length (Int): 检索word的最大长度
    :return wwm_segment (List[str]): wwm tokens, 加了特殊标记
    """
    word_segment = jieba.lcut(''.join(segment))
    wwm_segment = []

    i = 0
    while i < len(segment):
        # token不是中文的直接加入
        if len(re.findall('[\u4E00-\u9FA5]', segment[i])) == 0:
            wwm_segment.append(segment[i])
            i += 1
            continue

        has_add = False
        # 检索word
        for length in range(word_length, 0, -1):
            if i + length > len(segment):
                continue
            if ''.join(segment[i:i + length]) in word_segment:
                wwm_segment.append(segment[i])
                for l in range(1, length):
                    wwm_segment.append('##' + segment[i + l])
                i += length
                has_add = True
                break
        if not has_add:
            wwm_segment.append(segment[i])
            i += 1
    return wwm_segment


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if whole_word_mask and len(cand_indices) >= 1 and token.startswith("##"):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    # 不带##标识的tokens
    output_tokens = [t[2:] if t.startswith('##') else t for t in tokens]

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = output_tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)

            # TODO:标签不带##标识
            masked_lms.append(MaskedLmInstance(index=index, label=output_tokens[index]))
            output_tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return output_tokens, mask_indices, masked_token_labels


def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document = doc_database[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        # 只有一行或总长度大于目标长度
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = randrange(1, len(current_chunk))

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []

                # Random next
                # 只有一行，随机拼接
                if len(current_chunk) == 1 or (random() < 0.5 and args.remove_nsp):
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # Sample a random document, with longer docs being sampled more frequently
                    random_document = doc_database.sample_doc(current_idx=doc_idx, sentence_weighted=True)

                    random_start = randrange(0, len(random_document))
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                # 有多行，连续拼接
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]

                # get whole-word-mask tokens
                if whole_word_mask:
                    # get cws set
                    tokens = get_wwm_tokens(tokens)

                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list)

                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels,
                }
                instances.append(instance)
            # recover
            current_chunk = []
            current_length = 0
        i += 1

    return instances


if __name__ == '__main__':
    params = Params()
    tokenizer = BertTokenizer(vocab_file=os.path.join(params.bert_model_dir, 'vocab.txt'),
                              do_lower_case=True)
    # 读取自定义词典
    jieba.load_userdict(str(CORPUS_PATH) + '/dict.txt')

    vocab_list = list(tokenizer.vocab.keys())
    # 读取以空格分割的文档
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        with open(CORPUS_PATH / 'corpus_256.txt', 'r', encoding='utf-8') as f:
            doc = []
            for line in tqdm(f, desc="Loading Dataset", unit="lines"):
                line = line.strip()
                if line == "":
                    docs.add_document(doc)
                    doc = []
                else:
                    tokens = tokenizer.tokenize(line)
                    doc.append(tokens)
            if doc:
                docs.add_document(doc)  # If the last doc didn't end on a newline, make sure it still gets added
        if len(docs) <= 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")

        # write pre-train data
        for epoch in trange(args.epochs_to_generate, desc="Epoch"):
            epoch_filename = f"./epoch_{epoch}.json"
            num_instances = 0
            with open(epoch_filename, 'w', encoding='utf-8') as epoch_file:
                for doc_idx in trange(len(docs), desc="Document"):
                    doc_instances = create_instances_from_document(
                        docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                        masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                        whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list)
                    doc_instances = [json.dumps(instance, ensure_ascii=False) for instance in doc_instances]
                    for instance in doc_instances:
                        epoch_file.write(instance + '\n')
                        num_instances += 1
            # write metrics file
            metrics_file = f"./epoch_{epoch}_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": args.max_seq_len,
                }
                metrics_file.write(json.dumps(metrics, indent=4))
