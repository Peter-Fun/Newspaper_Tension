import transformers
import torch.nn as nn

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased", hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3)
        self.out = nn.Linear(768, 20)
        
        
    def forward(self, ids, mask, token_type_ids):

        _, o2 = self.bert_model(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        
        out= self.out(o2)
        
        return out