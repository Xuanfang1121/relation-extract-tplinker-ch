# -*- coding: utf-8 -*-
# @Time    : 2021/5/13 17:08
# @Author  : zxf
import os
import json
import requests

import torch
from flask import Flask
from flask import jsonify
from flask import request

import config
from utils import predict
from utils import get_token_num
from utils import get_tokenizer
from common.utils import Preprocessor
from utils import get_tplinker_bert_model
from utils import get_tplinker_lstm_model
from utils import get_data_bert_data_maker
from utils import get_data_bilstm_data_maker
from tplinker import HandshakingTaggingScheme, MetricsCalculator


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# model parameters
config = config.eval_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_home = config["data_home"]
experiment_name = config["exp_name"]
# test_data_path = os.path.join(data_home, experiment_name, config["test_data"])
batch_size = 1
rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])
save_res_dir = os.path.join(config["save_res_dir"], experiment_name)
max_seq_len = hyper_parameters["max_test_seq_len"]
sliding_len = hyper_parameters["sliding_len"]
force_split = hyper_parameters["force_split"]
# for reproductivity
torch.backends.cudnn.deterministic = True
if force_split:
    split_test_data = True
    print("force to split the test dataset!")

model_state_path = "./default_log_dir/LQAsWXoc/model_state_dict_6.pt"
# get tokenizer
tokenize, get_tok2char_span_map = get_tokenizer(config["encoder"], config["bert_path"])

preprocessor = Preprocessor(tokenize_func=tokenize,
                            get_tok2char_span_map_func=get_tok2char_span_map)

rel2id = json.load(open(rel2id_path, "r", encoding="utf-8"))
handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id, max_seq_len=max_seq_len)
metrics = MetricsCalculator(handshaking_tagger)

# get data maker and model
if config["encoder"] == "BERT":
    data_maker = get_data_bert_data_maker(config["bert_path"], handshaking_tagger)
    rel_extractor = get_tplinker_bert_model(config["bert_path"], rel2id, hyper_parameters)
elif config["encoder"] == "BiLSTM":
    token2idx_path = os.path.join(data_home, experiment_name, config["token2idx"])
    data_maker, token2idx = get_data_bilstm_data_maker(token2idx_path, handshaking_tagger)
    rel_extractor = get_tplinker_lstm_model(token2idx, hyper_parameters, rel2id)

# load model
rel_extractor.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
rel_extractor.eval()


@app.route("/tplinker", methods=['POST'])
def tplinker_predict():
    data_para = json.loads(request.get_data(), encoding="utf-8")
    id = data_para.get("id")
    text = data_para.get("text")
    test_data = [{"id": id,
                  "text": text}]

    data = preprocessor.split_into_short_samples(test_data,
                                                 max_seq_len,
                                                 sliding_len=sliding_len,
                                                 encoder=config["encoder"],
                                                 data_type="test")
    result = predict(config, data, data_maker, max_seq_len, batch_size, device, rel_extractor, True,
                     handshaking_tagger)

    return_result = {"errCode": 200,
                     "data": result}
    return jsonify(return_result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)