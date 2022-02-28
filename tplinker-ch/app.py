# -*- coding: utf-8 -*-
# @Time    : 2021/5/13 17:08
# @Author  : zxf
import os
import json
import requests
import traceback

import torch
from flask import Flask
from flask import jsonify
from flask import request
from transformers import BertTokenizerFast

from utils.util import predict
from common.common import logger
from config.config import eval_config
from common.utils import Preprocessor
from models.tplinker import DataMaker4Bert
from utils.util import get_tplinker_bert_model
from models.tplinker import HandshakingTaggingScheme, MetricsCalculator


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# model parameters
hyper_parameters = eval_config["hyper_parameters"]
device = "cpu" if eval_config["device_num"] == "-1" else "cuda:{}".format(eval_config["device_num"])

# test_data_path = os.path.join(data_home, experiment_name, config["test_data"])
batch_size = 1
rel2id_path = os.path.join(eval_config["data_home"], eval_config["rel2id"])
max_seq_len = hyper_parameters["max_test_seq_len"]
sliding_len = hyper_parameters["sliding_len"]
force_split = hyper_parameters["force_split"]
# read rel2id
rel2id = json.load(open(rel2id_path, "r", encoding="utf-8"))
# for reproductivity
torch.backends.cudnn.deterministic = True
if force_split:
    split_test_data = True
    print("force to split the test dataset!")

model_state_path = "./default_log_dir/sKQAhyrx/model.pt"
# get tokenizer
tokenizer = BertTokenizerFast.from_pretrained(eval_config["bert_path"],
                                              add_special_tokens=False,
                                              do_lower_case=False)
tokenize = tokenizer.tokenize
get_tok2char_span_map = lambda text: tokenizer.encode_plus(text,
                                                           return_offsets_mapping=True,
                                                           add_special_tokens=False)["offset_mapping"]

preprocessor = Preprocessor(tokenize_func=tokenize,
                            get_tok2char_span_map_func=get_tok2char_span_map)

handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id, max_seq_len=max_seq_len)
metrics = MetricsCalculator(handshaking_tagger)

# get data maker and model
data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)
rel_extractor = get_tplinker_bert_model(eval_config["bert_path"],
                                        rel2id, hyper_parameters)


# load model
rel_extractor.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
rel_extractor.eval()


@app.route("/tplinker", methods=['POST'])
def tplinker_predict():
    try:
        data_para = json.loads(request.get_data(), encoding="utf-8")
        id = data_para.get("id")
        text = data_para.get("text")
        test_data = [{"id": id,
                      "text": text}]

        data = preprocessor.split_into_short_samples(test_data,
                                                     max_seq_len,
                                                     sliding_len=sliding_len,
                                                     encoder=eval_config["encoder"],
                                                     data_type="test")
        result = predict(data, data_maker, max_seq_len,
                         batch_size, device, rel_extractor, True,
                         handshaking_tagger)

        return_result = {"errCode": 200,
                         "data": result}
        return jsonify(return_result)
    except Exception:
        return_result = {"errCode": 200,
                         "data": traceback.format_exc()}
        return jsonify(return_result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)