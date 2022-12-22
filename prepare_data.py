import os
import json
import torch
import numpy as np

from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable

from parser import args

from utils import seq_padding, subsequent_mask


class PrepareData:
    def __init__(self):
        # load data and tokenizer
        self.train_en, self.train_cn = self.load_data(args.train_file)
        self.dev_en, self.dev_cn = self.load_data(args.dev_file)

        # create vocab
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en, args.src)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn, args.tgt)

        # convert tokens to ids
        self.train_en, self.train_cn = self.word_to_id(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.word_to_id(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict)

        # split data batch + padding + mask
        self.train_data = self.split_batch(self.train_en, self.train_cn, args.batch_size)
        self.dev_data = self.split_batch(self.dev_en, self.dev_cn, args.batch_size)

    def load_data(self, path):
        """
        load data form specific file, and tokenizer at sentence level。
        Each sentence start with ‘BOS’ and end with ‘EOS’。
        """
        en = []
        cn = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                sent_en, sent_cn = line.strip().split('\t')
                sent_en = ["BOS"] + word_tokenize(sent_en.lower()) + ["EOS"]

                # 或按字符切分   sent_cn = ["BOS"] + [w for w in sent_cn] + ["EOS"]
                sent_cn = ["BOS"] + word_tokenize(" ".join([w for w in sent_cn])) + ["EOS"]
                en.append(sent_en)
                cn.append(sent_cn)

        return en, cn

    def build_dict(self, sentences, vocab_name, max_words=50000):
        """
        built vocab from trainning data.
        """
        # 统计词频
        word_count = Counter()
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2

        # 构建词典
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = args.UNK
        word_dict['PAD'] = args.PAD

        # 构建id2word映射
        index_dict = {v: k for k, v in word_dict.items()}

        # FIXME How to write vocab to file
        index_dict_sort_key = sorted(index_dict.items(), key=lambda d: d[0], reverse=False)
        with open("./data" + "/vocab_" + vocab_name + ".txt", 'w') as f:
            for key, value in index_dict_sort_key:
                f.write(str(key) + ': ' + value + '\n')

        return word_dict, total_words, index_dict

    def word_to_id(self, en, cn, en_dict, cn_dict, sort=True):
        """
        convert tokens to ids.
        """
        length = len(en)

        # 单词映射为索引
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        # 按照语句长度排序
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 把中文和英文按照同样的顺序排序
        if sort:
            sorted_index = len_argsort(out_en_ids)  # 以英文句子长度排序
            out_en_ids = [out_en_ids[i] for i in sorted_index]
            out_cn_ids = [out_cn_ids[i] for i in sorted_index]

        return out_en_ids, out_cn_ids

    def split_batch(self, en, cn, batch_size, shuffle=True):
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))

        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            batch_cn = seq_padding(batch_cn)
            batch_en = seq_padding(batch_en)
            batches.append(Batch(batch_en, batch_cn))

        return batches


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        src = torch.from_numpy(src).to(args.device).long()
        trg = torch.from_numpy(trg).to(args.device).long()

        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)  # padding处为False，其余为True
        if trg is not None:
            self.trg = trg[:, :-1]  # 去掉 EOS
            self.trg_y = trg[:, 1:]  # 去掉 BOS
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)  #
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
