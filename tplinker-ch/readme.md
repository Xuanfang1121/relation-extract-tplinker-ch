## TPLinker
这个代码是TPLinker bert版本的，模型预测可以得到关系主客体中的类型，TPLinker是发表于2020年的ACL中的论文
TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking提出的，
代码地址为：https://github.com/131250208/TPlinker-joint-extraction

### 环境依赖
```
onnxruntime==1.7.0
numpy==1.19.2
transformers==4.6.0
requests==2.26.0
tqdm==4.62.0
torch==1.8.1
Flask==2.0.1
glove==1.0.2
PyYAML==6.0
```

### 代码结构
```
  |__common  工具类函数
  |__postprocess
  |__preprocess data处理，生成模型需要的格式
  |__app.py  flask 服务
  |__config 参数配置文件 这里的参数可以优化一下 
  |__predict.py  模型预测
  |__models 模型文件
  |__main.py  主函数，模型训练
  |__utils 工具类函数
  |__data4bert 数据目录
  |__log 日志目录
```
备注：目前不支持多卡训练，

### 数据
本次代码训练是基于百度关系抽取数据集训练的，执行preprocess/BuildData.py生成模型需要的数据格式.
1. 修改preprocess/build_data_config.yaml中的参数，其中ori_data_format为tplinker
2. 执行preprocess/BuildData.py
输入的数据格式为
```
{
    "id": 0,
    "text": "《邪少兵王》是冰火未央写的网络小说连载于旗峰天下",
    "relation_list": [
      {
        "object": "冰火未央",
        "subject": "邪少兵王",
        "predicate": "作者"
      }
    ],
    "entity_list": [
      {
        "text": "冰火未央",
        "type": "人物"
      },
      {
        "text": "邪少兵王",
        "type": "图书作品"
      }
    ]
  }
```
生成的数据格式如下：
```
{
    "id": 0,
    "text": "《邪少兵王》是冰火未央写的网络小说连载于旗峰天下",
    "relation_list": [
      {
        "subject": "邪少兵王",
        "object": "冰火未央",
        "subj_char_span": [
          1,
          5
        ],
        "obj_char_span": [
          7,
          11
        ],
        "predicate": "作者",
        "subj_tok_span": [
          1,
          5
        ],
        "obj_tok_span": [
          7,
          11
        ]
      }
    ],
    "entity_list": [
      {
        "text": "冰火未央",
        "type": "人物",
        "char_span": [
          7,
          11
        ],
        "tok_span": [
          7,
          11
        ]
      },
      {
        "text": "邪少兵王",
        "type": "图书作品",
        "char_span": [
          1,
          5
        ],
        "tok_span": [
          1,
          5
        ]
      }
    ]
  }
```

### 模型训练
修改config.py中的相应参数后，进行模型训练
```
python main.py
```

### flask 服务
```
curl --location --request POST 'http://127.0.0.1:5000/tplinker' \
--header 'Content-Type: application/json' \
--data-raw '{
    "id": 100,
    "text": "《父老乡亲》是由是由由中国人民解放军海政文工团创作的军旅歌曲，石顺义作词，王锡仁作曲，范琳琳演唱"
}'
```

### 模型训练结果
百度2020关系抽取数据集，训练10个epoch的结果为
```
{'time': 520.0066828727722,
 'val_ent_seq_acc': 0.5921271881659563,
 'val_f1': 0.74142521155895,
 'val_head_rel_acc': 0.5819920530308212,
 'val_prec': 0.7324275875082538,
 'val_recall': 0.7506466507119183,
 'val_tail_rel_acc': 0.5767825752712821}
Current avf_f1: 0.74142521155895, Best f1: 0.74142521155895
```