# -*- coding: utf-8 -*-
# @Time    : 2021/5/10 15:30
# @Author  : zxf
import os
import json

import torch
from transformers import BertTokenizerFast

from utils.util import predict
from common.common import logger
from config.config import eval_config
from common.utils import Preprocessor
from models.tplinker import DataMaker4Bert
from utils.util import get_tplinker_bert_model
from models.tplinker import HandshakingTaggingScheme, MetricsCalculator


def tplinker_predict(config, test_data_path, model_state_path):
    hyper_parameters = eval_config["hyper_parameters"]

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
    device = "cpu" if eval_config["device_num"] == "-1" else "cuda:{}".format(eval_config["device_num"])

    # test_data_path = os.path.join(data_home, experiment_name, config["test_data"])
    batch_size = hyper_parameters["batch_size"]
    rel2id_path = os.path.join(eval_config["data_home"], eval_config["rel2id"])
    max_test_seq_len = hyper_parameters["max_test_seq_len"]
    sliding_len = hyper_parameters["sliding_len"]
    force_split = hyper_parameters["force_split"]
    # for reproductivity
    torch.backends.cudnn.deterministic = True
    if force_split:
        split_test_data = True
        print("force to split the test dataset!")
    # read test data
    test_data = json.load(open(test_data_path, "r", encoding="utf-8"))
    logger.info("test data size：{}".format(len(test_data)))
    # read relation
    rel2id = json.load(open(rel2id_path, "r", encoding="utf-8"))
    logger.info("rel2id num: {}".format(len(rel2id)))
    # get tokenizer
    # tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(eval_config["bert_path"],
                                                  add_special_tokens=False,
                                                  do_lower_case=False)
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(text,
                                                               return_offsets_mapping=True,
                                                               add_special_tokens=False)["offset_mapping"]
    # get data token num
    # max_tok_num = get_token_num(test_data, tokenize)
    # max_seq_len = min(max_tok_num, max_test_seq_len)
    max_seq_len = max_test_seq_len

    # data prpcessor
    preprocessor = Preprocessor(tokenize_func=tokenize,
                                get_tok2char_span_map_func=get_tok2char_span_map)
    data = preprocessor.split_into_short_samples(test_data,
                                                 max_seq_len,
                                                 sliding_len=sliding_len,
                                                 encoder=config["encoder"],
                                                 data_type="test")
    # get handshaking
    handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id, max_seq_len=max_seq_len)

    # get data maker and model
    data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)
    rel_extractor = get_tplinker_bert_model(config["bert_path"], rel2id, hyper_parameters)

    # load model
    rel_extractor.load_state_dict(torch.load(model_state_path,
                                             map_location=torch.device('cpu')))
    rel_extractor.eval()

    result = predict(data, data_maker, max_seq_len, batch_size, device, rel_extractor, True,
                     handshaking_tagger)

    with open("./results/predict_result.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_data_path = "./data4bert/baidu_relation_test_demo/test_data.json"
    model_state_path = "./default_log_dir/sKQAhyrx/model.pt"
    tplinker_predict(eval_config, test_data_path, model_state_path)