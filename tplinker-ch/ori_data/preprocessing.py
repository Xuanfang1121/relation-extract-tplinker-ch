# -*- coding: utf-8 -*-
# @Time    : 2021/5/4 14:14
# @Author  : zxf
import os
import re
import codecs
import json


def process(data_path, target_path):
    target_file = open(target_path, 'w', encoding='utf8')
    with codecs.open(data_path, "r", encoding="utf-8") as read_data:
        data_list = []
        for num, line in enumerate(read_data):
            line = re.sub('\n', '', line)
            data_dict = {}
            relation_list = []
            entity_list = []
            line_json = json.loads(line)
            data_dict["id"] = num
            # data_dict["text"] = line_json['text'].replace(' ', '#')   # '✈'
            text = line_json['text'].replace(' ', '#')
            text = re.sub('[💜＝●\xad★ ⃣ ╬😎🤘🏻®←═∝◇◆━┅┉😱『』✎]+', '', text)
            data_dict["text"] = text
            # c = line_json['spo_list']
            for rel_num in range(len(line_json['spo_list'])):
                relation = {}
                entity_obj = {}
                entity_sub = {}
                if line_json['spo_list'][rel_num]['object']['@value'] != '' and \
                        line_json['spo_list'][rel_num]['subject'] != '':
                    relation['object'] = line_json['spo_list'][rel_num]['object']['@value'].strip().replace(' ', '#')

                    relation['subject'] = line_json['spo_list'][rel_num]['subject'].strip().replace(' ', '#')
                    relation['predicate'] = line_json['spo_list'][rel_num]['predicate']
                    entity_obj['text'] = line_json['spo_list'][rel_num]['object']['@value'].strip().replace(' ', '#')
                    entity_obj['type'] = line_json['spo_list'][rel_num]['object_type']['@value']
                    entity_sub['text'] = line_json['spo_list'][rel_num]['subject'].strip().replace(' ', '#')
                    entity_sub['type'] = line_json['spo_list'][rel_num]['subject_type']
                    relation_list.append(relation)
                    if entity_obj not in entity_list:
                        entity_list.append(entity_obj)
                    if entity_sub not in entity_list:
                        entity_list.append(entity_sub)
                    data_dict['relation_list'] = relation_list
                    data_dict['entity_list'] = entity_list
            data_list.append(data_dict)

    target_file.write(json.dumps(data_list, ensure_ascii=False, indent=2))
    target_file.close()


if __name__ == "__main__":
    data_path = "./ori_baidu_relation/train_data.json"
    target_path = "./baidu_relation/train_data.json"
    process(data_path, target_path)
    data_path = "./ori_baidu_relation/dev_data.json"
    target_path = "./baidu_relation/dev_data.json"
    process(data_path, target_path)

