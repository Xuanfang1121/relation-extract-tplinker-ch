# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 15:33
# @Author  : zxf
import json

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, BertTokenizerFast

from tplinker import TPLinkerBert
from tplinker import TPLinkerBiLSTM
from tplinker import DataMaker4Bert
from tplinker import DataMaker4BiLSTM


def get_tokenizer_bert(bert_path):
    tokenizer = BertTokenizerFast.from_pretrained(bert_path,
                                                  add_special_tokens=False,
                                                  do_lower_case=False)
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(text,
                                                               return_offsets_mapping=True,
                                                               add_special_tokens=False)["offset_mapping"]
    return tokenize, get_tok2char_span_map


def get_tokenizer_lstm():
    tokenize = lambda text: text.split(" ")

    def get_tok2char_span_map(text):
        tokens = text.split(" ")
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append((char_num, char_num + len(tok)))
            char_num += len(tok) + 1  # +1: whitespace
        return tok2char_span

    return tokenize, get_tok2char_span_map


def get_tokenizer(encoder_name, bert_path):
    if encoder_name == "BERT":
        tokenize, get_tok2char_span_map = get_tokenizer_bert(bert_path)
    else:
        tokenize, get_tok2char_span_map = get_tokenizer_lstm()
    return tokenize, get_tok2char_span_map


def get_data_bert_data_maker(bert_path, handshaking_tagger):
    tokenizer = BertTokenizerFast.from_pretrained(bert_path,
                                                  add_special_tokens=False,
                                                  do_lower_case=False)
    data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)
    return data_maker


def get_data_bilstm_data_maker(token2idx_path, handshaking_tagger):

    token2idx = json.load(open(token2idx_path, "r", encoding="utf-8"))
    idx2token = {idx: tok for tok, idx in token2idx.items()}

    def text2indices(text, max_seq_len):
        input_ids = []
        tokens = text.split(" ")
        for tok in tokens:
            if tok not in token2idx:
                input_ids.append(token2idx['<UNK>'])
            else:
                input_ids.append(token2idx[tok])
        if len(input_ids) < max_seq_len:
            input_ids.extend([token2idx['<PAD>']] * (max_seq_len - len(input_ids)))
        input_ids = torch.tensor(input_ids[:max_seq_len])
        return input_ids

    def get_tok2char_span_map(text):
        tokens = text.split(" ")
        tok2char_span = []
        char_num = 0
        for tok in tokens:
            tok2char_span.append((char_num, char_num + len(tok)))
            char_num += len(tok) + 1  # +1: whitespace
        return tok2char_span

    data_maker = DataMaker4BiLSTM(text2indices, get_tok2char_span_map, handshaking_tagger)
    return data_maker, token2idx


def get_token_num(all_data, tokenize):
    max_tok_num = 0
    for sample in tqdm(all_data, desc="Calculate the max token number"):
        tokens = tokenize(sample["text"])
        max_tok_num = max(len(tokens), max_tok_num)
    return max_tok_num


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_tplinker_bert_model(bert_path, rel2id, hyper_parameters):
    roberta = AutoModel.from_pretrained(bert_path)
    hidden_size = roberta.config.hidden_size
    rel_extractor = TPLinkerBert(roberta,
                                 len(rel2id),
                                 hyper_parameters["shaking_type"],
                                 hyper_parameters["inner_enc_type"],
                                 hyper_parameters["dist_emb_size"],
                                 hyper_parameters["ent_add_dist"],
                                 hyper_parameters["rel_add_dist"],
                                 )
    return rel_extractor


def get_tplinker_lstm_model(token2idx, hyper_parameters, rel2id):
    word_embedding_init_matrix = np.random.normal(-1, 1, size=(len(token2idx),
                                                               hyper_parameters["word_embedding_dim"]))
    word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)

    rel_extractor = TPLinkerBiLSTM(word_embedding_init_matrix,
                                   hyper_parameters["emb_dropout"],
                                   hyper_parameters["enc_hidden_size"],
                                   hyper_parameters["dec_hidden_size"],
                                   hyper_parameters["rnn_dropout"],
                                   len(rel2id),
                                   hyper_parameters["shaking_type"],
                                   hyper_parameters["inner_enc_type"],
                                   hyper_parameters["dist_emb_size"],
                                   hyper_parameters["ent_add_dist"],
                                   hyper_parameters["rel_add_dist"],
                                   )
    return rel_extractor


def filter_duplicates(rel_list):
    rel_memory_set = set()
    filtered_rel_list = []
    for rel in rel_list:
        rel_memory = "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                 rel["subj_tok_span"][1],
                                                                 rel["predicate"],
                                                                 rel["obj_tok_span"][0],
                                                                 rel["obj_tok_span"][1])
        if rel_memory not in rel_memory_set:
            filtered_rel_list.append(rel)
            rel_memory_set.add(rel_memory)
    return filtered_rel_list


def get_test_prf(pred_sample_list, gold_test_data, metrics, pattern="only_head_text"):
    text_id2gold_n_pred = {}
    for sample in gold_test_data:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id] = {
            "gold_relation_list": sample["relation_list"],
        }

    for sample in pred_sample_list:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id]["pred_relation_list"] = sample["relation_list"]

    correct_num, pred_num, gold_num = 0, 0, 0
    for gold_n_pred in text_id2gold_n_pred.values():
        gold_rel_list = gold_n_pred["gold_relation_list"]
        pred_rel_list = gold_n_pred["pred_relation_list"] if "pred_relation_list" in gold_n_pred else []
        if pattern == "only_head_index":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                            rel["predicate"], rel["obj_tok_span"][0])
                                for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                            rel["predicate"], rel["obj_tok_span"][0])
                                for rel in pred_rel_list])
        elif pattern == "whole_span":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                            rel["subj_tok_span"][1],
                                                                            rel["predicate"], rel["obj_tok_span"][0],
                                                                            rel["obj_tok_span"][1])
                                for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                            rel["subj_tok_span"][1],
                                                                            rel["predicate"],
                                                                            rel["obj_tok_span"][0],
                                                                            rel["obj_tok_span"][1])
                                for rel in pred_rel_list])
        elif pattern == "whole_text":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"],
                                                            rel["predicate"],
                                                            rel["object"]) for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"],
                                                            rel["predicate"],
                                                            rel["object"]) for rel in pred_rel_list])
        elif pattern == "only_head_text":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0],
                                                            rel["predicate"],
                                                            rel["object"].split(" ")[0]) for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0],
                                                            rel["predicate"],
                                                            rel["object"].split(" ")[0]) for rel in pred_rel_list])

        for rel_str in pred_rel_set:
            if rel_str in gold_rel_set:
                correct_num += 1

        pred_num += len(pred_rel_set)
        gold_num += len(gold_rel_set)
    #     print((correct_num, pred_num, gold_num))
    prf = metrics.get_prf_scores(correct_num, pred_num, gold_num)
    return prf


def predict(config, test_data, data_maker, max_seq_len, batch_size, device,
            rel_extractor, split_test_data, handshaking_tagger):
    '''
    test_data: if split, it would be samples with subtext
    ori_test_data: the original data has not been split, used to get original text here
    '''
    indexed_test_data = data_maker.get_indexed_data(test_data, max_seq_len, data_type="test")  # fill up to max_seq_len
    test_dataloader = DataLoader(MyDataset(indexed_test_data),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 drop_last=False,
                                 collate_fn=lambda data_batch: data_maker.generate_batch(data_batch,
                                                                                         data_type="test"),
                                 )

    pred_sample_list = []
    for batch_test_data in tqdm(test_dataloader, desc="Predicting"):
        if config["encoder"] == "BERT":
            sample_list, batch_input_ids, batch_attention_mask, \
            batch_token_type_ids, tok2char_span_list, _, _, _ = batch_test_data

            batch_input_ids, batch_attention_mask, batch_token_type_ids = (batch_input_ids.to(device),
                                                                           batch_attention_mask.to(device),
                                                                           batch_token_type_ids.to(device))

        elif config["encoder"] in {"BiLSTM", }:
            sample_list, batch_input_ids, tok2char_span_list, _, _, _ = batch_test_data
            batch_input_ids = batch_input_ids.to(device)

        with torch.no_grad():
            if config["encoder"] == "BERT":
                batch_ent_shaking_outputs, batch_head_rel_shaking_outputs, \
                batch_tail_rel_shaking_outputs = rel_extractor(batch_input_ids,
                                                               batch_attention_mask,
                                                               batch_token_type_ids,
                                                               )
            elif config["encoder"] in {"BiLSTM", }:
                batch_ent_shaking_outputs, batch_head_rel_shaking_outputs, \
                batch_tail_rel_shaking_outputs = rel_extractor(batch_input_ids)

        batch_ent_shaking_tag, batch_head_rel_shaking_tag, \
        batch_tail_rel_shaking_tag = torch.argmax(batch_ent_shaking_outputs, dim=-1), \
                                     torch.argmax(batch_head_rel_shaking_outputs, dim=-1), \
                                     torch.argmax(batch_tail_rel_shaking_outputs, dim=-1)

        for ind in range(len(sample_list)):
            gold_sample = sample_list[ind]
            text = gold_sample["text"]
            text_id = gold_sample["id"]
            tok2char_span = tok2char_span_list[ind]
            ent_shaking_tag, head_rel_shaking_tag, tail_rel_shaking_tag = batch_ent_shaking_tag[ind], \
                                                                          batch_head_rel_shaking_tag[ind], \
                                                                          batch_tail_rel_shaking_tag[ind]

            tok_offset, char_offset = 0, 0
            if split_test_data:
                tok_offset, char_offset = gold_sample["tok_offset"], gold_sample["char_offset"]
            rel_list = handshaking_tagger.decode_rel_fr_shaking_tag(text,
                                                                    ent_shaking_tag,
                                                                    head_rel_shaking_tag,
                                                                    tail_rel_shaking_tag,
                                                                    tok2char_span,
                                                                    tok_offset=tok_offset,
                                                                    char_offset=char_offset)
            pred_sample_list.append({
                "text": text,
                "id": text_id,
                "relation_list": rel_list,
            })

    return pred_sample_list