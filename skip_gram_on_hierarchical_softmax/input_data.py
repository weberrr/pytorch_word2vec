import numpy as np
from collections import deque
from skip_gram_on_hierarchical_softmax.huffman_tree import HuffmanTree


class InputData:
    def __init__(self, input_file_name, min_count):
        self.input_file_name = input_file_name
        self.input_file = open(self.input_file_name)  # 数据文件
        self.min_count = min_count  # 要淘汰的低频数据的频度
        self.wordId_frequency_dict = dict()  # 词id-出现次数 dict
        self.word_count = 0  # 单词数（重复的词只算1个）
        self.word_count_sum = 0  # 单词总数 （重复的词 次数也累加）
        self.sentence_count = 0  # 句子数
        self.id2word_dict = dict()  # 词id-词 dict
        self.word2id_dict = dict()  # 词-词id dict
        self._init_dict()  # 初始化字典
        self.huffman_tree = HuffmanTree(self.wordId_frequency_dict)  # 霍夫曼树
        self.huffman_pos_path, self.huffman_neg_path = self.huffman_tree.get_all_pos_and_neg_path()
        self.word_pairs_queue = deque()
        # 结果展示
        print('Word Count is:', self.word_count)
        print('Word Count Sum is', self.word_count_sum)
        print('Sentence Count is:', self.sentence_count)
        print('Tree Node is:', len(self.huffman_tree.huffman))

    def _init_dict(self):
        word_freq = dict()
        # 统计 word_frequency
        for line in self.input_file:
            line = line.strip().split(' ')  # 去首尾空格
            self.word_count_sum += len(line)
            self.sentence_count += 1
            for word in line:
                try:
                    word_freq[word] += 1
                except:
                    word_freq[word] = 1
        word_id = 0
        # 初始化 word2id_dict,id2word_dict, wordId_frequency_dict字典
        for per_word, per_count in word_freq.items():
            if per_count < self.min_count:  # 去除低频
                self.word_count_sum -= per_count
                continue
            self.id2word_dict[word_id] = per_word
            self.word2id_dict[per_word] = word_id
            self.wordId_frequency_dict[word_id] = per_count
            word_id += 1
        self.word_count = len(self.word2id_dict)

    # 获取mini-batch大小的 正采样对 (w,v) w为目标词id,v为上下文中的一个词的id。上下文步长为window_size，即2c = 2*window_size
    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pairs_queue) < batch_size:
            for _ in range(10000):  # 先加入10000条，减少循环调用次数
                self.input_file = open(self.input_file_name, encoding="utf-8")
                sentence = self.input_file.readline()
                if sentence is None or sentence == '':
                    continue
                wordId_list = []  # 一句中的所有word 对应的 id
                for word in sentence.strip().split(' '):
                    try:
                        word_id = self.word2id_dict[word]
                        wordId_list.append(word_id)
                    except:
                        continue
                # 寻找正采样对 (w,v) 加入正采样队列
                for i, wordId_w in enumerate(wordId_list):
                    for j, wordId_v in enumerate(wordId_list[max(i - window_size, 0):i + window_size + 1]):
                        assert wordId_w < self.word_count
                        assert wordId_v < self.word_count
                        if i == j:  # 上下文=中心词 跳过
                            continue
                        self.word_pairs_queue.append((wordId_w, wordId_v))
        result_pairs = []  # 返回mini-batch大小的正采样对
        for _ in range(batch_size):
            result_pairs.append(self.word_pairs_queue.popleft())
        return result_pairs

    def get_pairs(self, pos_pairs):
        neg_word_pair = []
        pos_word_pair = []
        for pair in pos_pairs:
            pos_word_pair += zip([pair[0]] * len(self.huffman_pos_path[pair[1]]), self.huffman_pos_path[pair[1]])
            neg_word_pair += zip([pair[0]] * len(self.huffman_neg_path[pair[1]]), self.huffman_neg_path[pair[1]])
        return pos_word_pair, neg_word_pair


    # 估计数据中正采样对数，用于设定batch
    def evaluate_pairs_count(self, window_size):
        return self.word_count_sum * (2 * window_size - 1) - (self.sentence_count - 1) * (1 + window_size) * window_size


# 测试所有方法
def test():
    test_data = InputData('./data.txt', 1)
    test_data.evaluate_pairs_count(2)
    pos_pairs = test_data.get_batch_pairs(10, 2)
    print('正采样:')
    print(pos_pairs)
    pos_word_pairs = []
    for pair in pos_pairs:
        pos_word_pairs.append((test_data.id2word_dict[pair[0]], test_data.id2word_dict[pair[1]]))
    print(pos_word_pairs)
    neg_pair = test_data.get_negative_sampling(pos_pairs, 3)
    print('负采样:')
    print(neg_pair)
    neg_word_pair = []
    for pair in neg_pair:
        neg_word_pair.append(
            (test_data.id2word_dict[pair[0]], test_data.id2word_dict[pair[1]], test_data.id2word_dict[pair[2]]))
    print(neg_word_pair)


if __name__ == '__main__':
    test()
