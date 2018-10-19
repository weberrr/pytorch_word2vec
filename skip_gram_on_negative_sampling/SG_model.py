import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.w_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self._init_emb()

    def _init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.w_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_w, pos_v, neg_v):
        emb_w = self.w_embeddings(torch.LongTensor(pos_w))  # 转为tensor 大小 [ mini_batch_size * emb_dimension ]
        emb_v = self.v_embeddings(torch.LongTensor(pos_v))
        neg_emb_v = self.v_embeddings(torch.LongTensor(neg_v))  # 转换后大小 [ negative_sampling_number * mini_batch_size * emb_dimension ]
        score = torch.mul(emb_w, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_score = torch.bmm(neg_emb_v, emb_v.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)
        # L = log sigmoid (Xw.T * θv) + ∑neg(v) [log sigmoid (-Xw.T * θneg(v))]
        loss = -1 * (torch.sum(score) + torch.sum(neg_score))
        return loss

    def save_embedding(self, id2word, file_name):
        embedding = self.w_embeddings.weight.data.numpy()
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


def test():
    model = SkipGramModel(100, 10)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    pos_w = [0, 0, 1, 1, 1]
    pos_v = [1, 2, 0, 2, 3]
    neg_v = [[23, 42, 32], [32, 24, 53], [32, 24, 53], [32, 24, 53], [32, 24, 53]]
    model.forward(pos_w, pos_v, neg_v)
    model.save_embedding(id2word, 'test.txt')


if __name__ == '__main__':
    test()
