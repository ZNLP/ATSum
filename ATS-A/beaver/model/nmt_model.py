# -*- coding: utf-8 -*-
from typing import Dict

import torch
import torch.nn as nn

from beaver.model.embeddings import Embedding
from beaver.model.transformer import Decoder, Encoder, TransAttn


class Generator(nn.Module):
    def __init__(self, hidden_size: int, tgt_vocab_size: int):
        self.vocab_size = tgt_vocab_size
        super(Generator, self).__init__()
        self.linear_hidden = nn.Linear(hidden_size, tgt_vocab_size)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.sm = nn.Softmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_hidden.weight)

    def forward(self, dec_out):
        score = self.linear_hidden(dec_out)
        sm_score = self.sm(score)
        return sm_score


class NMTModel(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, generator: Generator, transattn: TransAttn):
        super(NMTModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.transattn = transattn

    def forward(self, src, tgt, probs, idxes):
        tgt = tgt[:, :-1]  # shift left
        src_pad = src.eq(self.encoder.embedding.word_padding_idx)
        tgt_pad = tgt.eq(self.decoder.embedding.word_padding_idx)

        enc_out = self.encoder(src, src_pad)
        decoder_outputs, _, p_trans, weights = self.decoder(tgt, enc_out, src_pad, tgt_pad)
        scores = self.generator(decoder_outputs)
        bsize, _, tgtvocab = scores.size()
        _, srclen, _ = enc_out.size()
        tmp_trans_scores = torch.zeros(bsize, srclen, tgtvocab).cuda()
        trans_probs = self.transattn(enc_out, idxes)
        prob_mask = (probs > 0.05).float()
        trans_probs = trans_probs * prob_mask
        tmp_trans_scores.scatter_add_(2, idxes, trans_probs)
        trans_scores = torch.matmul(weights, tmp_trans_scores)

        # tmp_score = (1 - p_trans) * scores
        # tmp_score.scatter_add_(2, idxes, p_trans * trans_scores)
        final_scores = torch.log(p_trans * trans_scores + (1 - p_trans) * scores)
        return final_scores

    @classmethod
    def load_model(cls, model_opt,
                   pad_ids: Dict[str, int],
                   vocab_sizes: Dict[str, int],
                   checkpoint=None):
        src_embedding = Embedding(embedding_dim=model_opt.hidden_size,
                                  dropout=model_opt.dropout,
                                  padding_idx=pad_ids["src"],
                                  vocab_size=vocab_sizes["src"])

        if len(model_opt.vocab) == 2:
            tgt_embedding = Embedding(embedding_dim=model_opt.hidden_size,
                                      dropout=model_opt.dropout,
                                      padding_idx=pad_ids["tgt"],
                                      vocab_size=vocab_sizes["tgt"])
        else:
            # use shared word embedding for source and target
            tgt_embedding = src_embedding

        encoder = Encoder(model_opt.layers,
                          model_opt.heads,
                          model_opt.hidden_size,
                          model_opt.dropout,
                          model_opt.ff_size,
                          src_embedding)

        decoder = Decoder(model_opt.layers,
                          model_opt.heads,
                          model_opt.hidden_size,
                          model_opt.dropout,
                          model_opt.ff_size,
                          tgt_embedding)

        transattn = TransAttn(model_opt.heads,
                              model_opt.hidden_size,
                              model_opt.dropout,
                              tgt_embedding)

        generator = Generator(model_opt.hidden_size, vocab_sizes["tgt"])

        model = cls(encoder, decoder, generator, transattn)

        if model_opt.train_from and checkpoint is None:
            checkpoint = torch.load(model_opt.train_from, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["model"])
        elif checkpoint is not None:
            model.load_state_dict(checkpoint)
        return model
