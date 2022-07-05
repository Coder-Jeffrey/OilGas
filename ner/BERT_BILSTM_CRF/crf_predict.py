# -*- encoding: utf-8 -*-

import torch
import os
# import flask
# from flask import request
import sys
# sys.path.append('/root/workspace/Bert-BiLSTM-CRF-pytorch')
from ner.BERT_BILSTM_CRF.utils import tag2idx, idx2tag
from ner.BERT_BILSTM_CRF.crf import Bert_BiLSTM_CRF
from pytorch_pretrained_bert import BertTokenizer
from typing import NamedTuple

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CRF_MODEL_PATH = 'checkpoints/03/15.pt'
BERT_PATH = 'chinese_L-12_H-768_A-12'

class CRF(object):
    def __init__(self, crf_model, bert_model, device='cpu'):
        self.device = torch.device(device)
        self.model = Bert_BiLSTM_CRF(tag2idx)
        self.model.load_state_dict(torch.load(crf_model))
        self.model.to(device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)


    def predict(self, text):
        """Using CRF to predict label

        Arguments:
            text {str} -- [description]
        """
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']
        xx = self.tokenizer.convert_tokens_to_ids(tokens)
        xx = torch.tensor(xx).unsqueeze(0).to(self.device)
        _, y_hat = self.model(xx)
        pred_tags = []
        for tag in y_hat.squeeze():
            pred_tags.append(idx2tag[tag.item()])
        print(pred_tags)
        print('*'*30)
        print(tokens)
        return pred_tags, tokens

    def parse(self, tokens, pred_tags):
        """Parse the predict tags to real word

        Arguments:
            x {List[str]} -- the origin text
            pred_tags {List[str]} -- predicted tags

        Return:
            entities {List[str]} -- a list of entities
        """
        entities = {}
        entity = None
        for idx, st in enumerate(pred_tags):
            if entity is None:
                if st.startswith('B'):
                    tag = st[2:]
                    entity = {}
                    entity['start'] = idx
                else:
                    continue
            else:
                if st == 'O':
                    entity['end'] = idx
                    name = ''.join(tokens[entity['start'] : entity['end']])
                    entities[tag] = name
                    entity = None
                elif st.startswith('B'):
                    entity['end'] = idx
                    name = ''.join(tokens[entity['start'] : entity['end']])
                    entities[tag] = name
                    entity = {}
                    entity['start'] = idx
                else:
                    continue
        return entities


crf = CRF(CRF_MODEL_PATH, BERT_PATH, 'cuda')

def get_crf_ners(text):
    # text = '罗红霉素和头孢能一起吃吗'
    pred_tags, tokens = crf.predict(text)
    entities = crf.parse(tokens, pred_tags)
    return entities



if __name__ == "__main__":
    # app = flask.Flask(__name__)
    # @app.route("/api/MedicalNer",methods=["GET"])
    # def get_crf_ners():
    #     text = request.args.get("data")
    #     pred_tags, tokens = crf.predict(text)
    #     entities = crf.parse(tokens, pred_tags)
    #     return entities
    # app.run(host='127.0.0.1',port=8181,debug=True)


    text = '达西定律是什么'
    print(get_crf_ners(text))


# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.current_device())