"""
from FlagEmbedding import BGE_M3   
model = BGE_M3('BAAI/bge-m3')
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

embeddings_1 = model.encode(sentences_1, 
                            batch_size=12, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
"""

from FlagEmbedding import FlagModel
import torch.nn as nn
import torch

from transformers import AutoModel, AutoTokenizer

from typing import List

class BGE(nn.Module):
    def __init__(self, freeze_bge: bool = True, use_flag_model: bool = False):
        super(BGE, self).__init__()

        self.use_flag_model = use_flag_model
        if not use_flag_model:
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
            self.model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")

            if freeze_bge:
                # Freeze all layers
                for param in self.model.parameters():
                    param.requires_grad = False
        
        else:
            self.bge_model = FlagModel("BAAI/bge-large-en-v1.5")  # Not a nn.Module. 
            self.double() # get dtype the same

        self.out = nn.Linear(1024, 20)
        
        
    
    def forward(self, x: List[str]):
        """
        Takes a list of strings and classifies them.

        Defining Shape:
            B: batch size
            D: embedding dim (default: 1024)
        """
        if not self.use_flag_model:
            inputs = self.tokenizer(
                x,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            )

            inputs['input_ids'] = inputs['input_ids'].cuda()
            inputs['token_type_ids'] = inputs['token_type_ids'].cuda()
            inputs['attention_mask'] = inputs['attention_mask'].cuda()

            x = self.model(**inputs, return_dict=True).last_hidden_state
            x = x[:, 0]

        else:
            x = self.bge_model.encode(x)  # Numpy Array
            x = torch.tensor(x).cuda()  # Tensor, SHAPE: B D
            x = x.to(torch.float64) # get the dtypes the same

        x = self.out(x)
        return x


    # def forward(self, ids, mask, token_type_ids):

    #     _, o2 = self.bge_model(
    #         ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
    #     )
        
    #     out= self.out(o2)
        
    #     return out