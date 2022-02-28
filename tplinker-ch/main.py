# -*- coding: utf-8 -*-
# @Time: 2022/2/25 17:22
# @Author: zxf
import os
import json
import time

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from common.common import logger
from utils.util import bias_loss
from utils.util import MyDataset
from utils.util import train_step
from utils.util import model_evaluate
from common.utils import Preprocessor
from config.config import train_config
from utils.util import get_token_max_num
from models.tplinker import DataMaker4Bert
from models.tplinker import MetricsCalculator
from utils.util import get_tplinker_bert_model
from utils.util import get_model_training_scheduler
from models.tplinker import HandshakingTaggingScheme


def main():
    # config.train_config
    hyper_parameters = train_config["hyper_parameters"]
    os.environ["CUDA_VISIBLE_DEVICES"] = train_config["device_num"]
    device = "cpu" if train_config["device_num"] == "-1" else "cuda:{}".format(train_config["device_num"])
    # make save model path
    model_state_dict_dir = train_config["path_to_save_model"]
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)

    # get tokenzier
    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(train_config["bert_path"],
                                                  add_special_tokens=False,
                                                  do_lower_case=False)
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(text,
                                                               return_offsets_mapping=True,
                                                               add_special_tokens=False)["offset_mapping"]
    # preprocessor
    preprocessor = Preprocessor(tokenize_func=tokenize,
                                get_tok2char_span_map_func=get_tok2char_span_map)
    # get data path
    train_data_path = os.path.join(train_config["data_home"], train_config["train_data"])
    valid_data_path = os.path.join(train_config["data_home"], train_config["valid_data"])
    rel2id_path = os.path.join(train_config["data_home"], train_config["rel2id"])
    # read data and relation
    train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
    valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))
    rel2id = json.load(open(rel2id_path, "r", encoding="utf-8"))
    logger.info("train data size: {}".format(len(train_data)))
    logger.info("valid data size: {}".format(len(valid_data)))
    logger.info("relation number: {}".format(len(rel2id)))
    # get data max token num
    max_tok_num = get_token_max_num(train_data, valid_data, tokenize)
    logger.info("数据集中的最大长度为:{}".format(max_tok_num))
    # 对数据进行截断处理
    if max_tok_num > hyper_parameters["max_seq_len"]:
        train_data = preprocessor.split_into_short_samples(train_data,
                                                           hyper_parameters["max_seq_len"],
                                                           sliding_len=hyper_parameters["sliding_len"],
                                                           encoder=train_config["encoder"]
                                                           )
        valid_data = preprocessor.split_into_short_samples(valid_data,
                                                           hyper_parameters["max_seq_len"],
                                                           sliding_len=hyper_parameters["sliding_len"],
                                                           encoder=train_config["encoder"]
                                                           )
        logger.info("train: {}".format(len(train_data)), "valid: {}".format(len(valid_data)))
    # 更新max token num
    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])
    # 定义handshaking
    handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id, max_seq_len=max_seq_len)
    data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)
    indexed_train_data = data_maker.get_indexed_data(train_data, max_seq_len)
    indexed_valid_data = data_maker.get_indexed_data(valid_data, max_seq_len)
    # get dataloader
    train_dataloader = DataLoader(MyDataset(indexed_train_data),
                                  batch_size=hyper_parameters["batch_size"],
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=False,
                                  collate_fn=data_maker.generate_batch,
                                  )
    valid_dataloader = DataLoader(MyDataset(indexed_valid_data),
                                  batch_size=hyper_parameters["batch_size"],
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=False,
                                  collate_fn=data_maker.generate_batch,
                                  )
    # get tplinker model
    rel_extractor = get_tplinker_bert_model(train_config["bert_path"], rel2id,
                                            hyper_parameters)
    rel_extractor = rel_extractor.to(device)
    metrics = MetricsCalculator(handshaking_tagger)
    loss_func = bias_loss(device)
    init_learning_rate = float(hyper_parameters["lr"])
    optimizer = torch.optim.Adam(rel_extractor.parameters(), lr=init_learning_rate)
    # get scheduler
    scheduler = get_model_training_scheduler(optimizer, len(train_dataloader),
                                             hyper_parameters)
    logger.info("开始训练")
    best_f1 = 0.0
    for epoch in range(hyper_parameters["epochs"]):
        rel_extractor.train()
        t_ep = time.time()
        # start_lr = optimizer.param_groups[0]['lr']
        total_loss, total_ent_sample_acc, total_head_rel_sample_acc, \
        total_tail_rel_sample_acc = 0., 0., 0., 0.
        for batch_ind, batch_train_data in enumerate(train_dataloader):
            t_batch = time.time()
            z = (2 * len(rel2id) + 1)
            steps_per_ep = len(train_dataloader)
            total_steps = hyper_parameters["loss_weight_recover_steps"] + 1  # + 1 avoid division by zero error
            current_step = steps_per_ep * epoch + batch_ind
            w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
            w_rel = min((len(rel2id) / z) * current_step / total_steps, (len(rel2id) / z))
            loss_weights = {"ent": w_ent, "rel": w_rel}

            loss, ent_sample_acc, head_rel_sample_acc, \
            tail_rel_sample_acc = train_step(rel_extractor, batch_train_data, optimizer, loss_weights, loss_func,
                                             metrics, device)
            scheduler.step()

            total_loss += loss
            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            avg_loss = total_loss / (batch_ind + 1)
            avg_ent_sample_acc = total_ent_sample_acc / (batch_ind + 1)
            avg_head_rel_sample_acc = total_head_rel_sample_acc / (batch_ind + 1)
            avg_tail_rel_sample_acc = total_tail_rel_sample_acc / (batch_ind + 1)

            batch_print_format = "\rrun_name: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + \
                                 "t_ent_sample_acc: {}, t_head_rel_sample_acc: {}, t_tail_rel_sample_acc: {}," + \
                                 "lr: {}, batch_time: {}, total_time: {} -------------"
            if (batch_ind + 1) % hyper_parameters["log_interval"] == 0:
                logger.info(batch_print_format.format(train_config["run_name"],
                                                      epoch + 1, hyper_parameters["epochs"],
                                                      batch_ind + 1, len(train_dataloader),
                                                      avg_loss, avg_ent_sample_acc,
                                                      avg_head_rel_sample_acc,
                                                      avg_tail_rel_sample_acc,
                                                      optimizer.param_groups[0]['lr'],
                                                      time.time() - t_batch,
                                                      time.time() - t_ep,
                                                      ))

        # model eval
        f1 = model_evaluate(valid_dataloader, rel_extractor, metrics, logger, device,
                            hyper_parameters["match_pattern"])
        if f1 >= best_f1:
            best_f1 = f1
            torch.save(rel_extractor.state_dict(),
                       os.path.join(model_state_dict_dir, "model.pt"))


if __name__ == "__main__":
    main()