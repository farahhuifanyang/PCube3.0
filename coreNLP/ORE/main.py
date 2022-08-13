'''
Author: Huifan Yang
Date: 2021-06-02
LastEditTime: 2021-06-05 17:59:12
LastEditors: Please set LastEditors
Description: Define algorithm interface
FilePath: /PCube3.0/coreNLP/ORE/main.py
'''

from coreNLP.ORE.config import FLAGS
from configure import globalFLAGS
from coreNLP.Algorithm import Algorithm
from coreNLP.ORE.dataParser import evaluate

import os
import json
import torch
import itertools
from pytorch_transformers import (BertConfig, BertForQuestionAnswering, BertTokenizer)


class OpenRelExtractor():
    def __init__(self) -> None:
        self.no_cuda = FLAGS.no_cuda

        self.n_gpu = FLAGS.n_gpu
        self.per_gpu_eval_batch_size = FLAGS.per_gpu_eval_batch_size
        self.eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)
        self.device = FLAGS.device

        self.tokenizer_name = FLAGS.tokenizer_name
        self.model_name_or_path = FLAGS.model_name_or_path
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name if self.tokenizer_name else self.model_name_or_path, do_lower_case=False)

        self.config = BertConfig.from_pretrained(self.tokenizer_name if self.tokenizer_name else self.model_name_or_path)
        self.model = BertForQuestionAnswering(self.config)
        self.state_dict = FLAGS.state_dict
        self.model_state_dict = self.state_dict
        self.model.load_state_dict(torch.load(self.model_state_dict))
        self.model.to(self.device)
        self.model.eval()

    def convert_to_squad_form(self, context, question):
        orig_data = {
            "version": "v1.0",
            "data": [
                {
                    "paragraphs": [
                        {
                            "id": "",
                            "context": "",
                            "qas": [
                                {
                                    "question": "",
                                                "id": "",
                                                "answers": []
                                }
                            ]
                        }
                    ],
                    "id": "",
                    "title": ""
                }
            ]
        }

        orig_data["data"][0]['paragraphs'][0]['context'] = context
        for i in range(len(question)):
            orig_data["data"][0]['paragraphs'][0]['qas'][i][
                'question'] = question[i]

        return orig_data

    def run(self, inputs=None):
        triple_id = 0
        outputs = {}
        for key, value in inputs.items():
            sent_list = value
            for sid, sent_ent in enumerate(sent_list):
                sent = sent_ent["sentence"]
                entity_list = []
                for ent in sent_ent["entities"]:
                    if "wikiID" in ent:
                        entity_list.append(ent)
                triples = []
                if len(entity_list) > 1:
                    entity_pairs = list(itertools.combinations(entity_list, 2))
                    for pair in entity_pairs:
                        triple = {}
                        triple["sentence"] = sent
                        triple['entity1'] = pair[0]
                        triple['entity2'] = pair[1]
                        triple['relation'] = ''
                        triple['tripleID'] = str(triple_id)
                        triple_id += 1

                        entity_1_text = pair[0]["form"]
                        entity_2_text = pair[1]["form"]
                        context = sent
                        question = []
                        ques = "\"" + entity_1_text + "\"" + "?" + "\"" + entity_2_text + "\""
                        question.append(ques)
                        # print(ques)

                        input_json = self.convert_to_squad_form(context, question)
                        evaluate(FLAGS, input_json, self.model, self.tokenizer)

                        predict_file = os.path.join(FLAGS.output_dir, "predictions_.json")
                        with open(predict_file, "r", encoding='utf-8') as reader:
                            orig_data = json.load(reader)
                            # print(orig_data)
                            rel = orig_data[""]
                            triple['relation'] = ''.join(rel.split())

                        triples.append(triple)
                if triples:
                    outputs[key] = triples
        return outputs


if __name__ == "__main__":
    openRelExtractor = OpenRelExtractor()
    json_folder_path = FLAGS.json_folder_path
    json_files = [x for x in os.listdir(json_folder_path)]
    for json_file in json_files:
        json_file_path = os.path.join(json_folder_path, json_file)
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
            outputs = openRelExtractor.run(json_data)
            print(outputs)
