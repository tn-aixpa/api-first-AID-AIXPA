from llama_index.retrievers.bm25 import BM25Retriever
from typing import List, Dict

from rank_bm25 import BM25Okapi

from dataclasses import dataclass

from tools.chunker import TextNode
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


retriever_bge = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
tokenizer_bge = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')

class Retriever_llamaindex_bm25:
    """
    BM25 retriever to be used if knowledge base consist of Llamaindex chunks
    """
    def __init__(self,
                 knowledge_base,
                 name='BM25',
                 top_k=3):

        self.name = name
        self.top_k = top_k
        self.knowledge_base = knowledge_base.nodes
        self.retriever = self.set_retriever()
    def set_retriever(self):
        if self.name == 'BM25':
            retriever = BM25Retriever.from_defaults(nodes=self.knowledge_base,
                                                    similarity_top_k=self.top_k)
        else:
            print("RetrieverModule Error: Select a valid retriever.")
            retriever = None
        return retriever

    def retrieve(self, query: str):
        """
        TODO
        """
        return self.retriever.retrieve(query.strip())

    def set_top_k(self, top_k):
        """
        TODO
        """
        self.top_k = top_k
        self.retriever = self.set_retriever()


class Retriever_bm25:
    """
    BM25 retriever that can be used with any knowledge base 
    (both llama index nodes and generic text nodes as defined in chunker)
    """
    def __init__(self,
                 knowledge_base,
                 name='BM25',
                 top_k=3):

        self.name = name
        self.top_k = top_k
        self.knowledge_base = knowledge_base.nodes
        self.retriever = self.set_retriever()
        
    def set_retriever(self):
        if self.name == 'BM25':
            corpus = [doc.text for doc in self.knowledge_base]
            tokenized_docs = [doc.split(" ") for doc in corpus]
            retriever = BM25Okapi(tokenized_docs)

        else:
            print("RetrieverModule Error: Select a valid retriever.")
            retriever = None
        return retriever
    
    def retrieve(self, query: str):
        """
        Retrieve top n most similar chunks to the query
        """      
        text_metadata_dict = {node.dict()['text']:node.dict()['metadata'] for node in self.knowledge_base}
        corpus = list(text_metadata_dict.keys())
        tokenized_query = query.split(" ")
        retrieved_docs = self.retriever.get_top_n(tokenized_query, corpus, n=self.top_k)
        return [TextNode(text = retrieved_doc.strip(), metadata = text_metadata_dict[retrieved_doc]) for retrieved_doc in retrieved_docs]

class Retriever_BGE_v2_m3:
    """
    bge-reranker-v2-m3
    """
    def __init__(self,
                 knowledge_base,
                 name='BGE',
                 top_k=3):

        self.name = name
        self.top_k = top_k
        self.knowledge_base = knowledge_base.nodes
        self.retriever = self.set_retriever()
        self.tokenizer = self.set_tokenizer()

    def set_tokenizer(self):
        tokenizer = tokenizer_bge
        return tokenizer

    def set_retriever(self):
        if self.name == 'BGE':
            retriever = retriever_bge
            retriever.eval()
        else:
            print("RetrieverModule Error: Select a valid retriever.")
            retriever = None
        return retriever

    def retrieve(self, query: str):
        """
        TODO
        """
        text_metadata_dict = {node.dict()['text']:node.dict()['metadata'] for node in self.knowledge_base}
        corpus = list(text_metadata_dict.keys())
        pairs = [[query, k] for k in text_metadata_dict.keys()]
        # scores = {k:self.retriever.compute_score([query, k])[0] for k in text_metadata_dict.keys()}

        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.retriever(**inputs, return_dict=True).logits.view(-1, ).float()

        scores_dict = dict(zip(text_metadata_dict.keys(), scores.tolist()))
        sorted_values = sorted(scores_dict.values(), reverse=True)
        retrieved_docs = [next((k for k, v in scores_dict.items() if v == value), None) for value in sorted_values[:self.top_k]]
        return [TextNode(text = retrieved_doc.strip(), metadata = text_metadata_dict[retrieved_doc]) for retrieved_doc in retrieved_docs]

    def set_top_k(self, top_k):
        """
        TODO
        """
        self.top_k = top_k
        self.retriever = self.set_retriever()






# from llama_index.retrievers.bm25 import BM25Retriever
# from typing import List, Dict

# from rank_bm25 import BM25Okapi

# from dataclasses import dataclass

# from tools.chunker import TextNode
# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer


# retriever_bge = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
# tokenizer_bge = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')

# class Retriever_llamaindex_bm25:
#     """
#     BM25 retriever to be used if knowledge base consist of Llamaindex chunks
#     """
#     def __init__(self,
#                  knowledge_base,
#                  name='BM25',
#                  top_k=3):

#         self.name = name
#         self.top_k = top_k
#         self.knowledge_base = knowledge_base.nodes
#         self.retriever = self.set_retriever()
#     def set_retriever(self):
#         if self.name == 'BM25':
#             retriever = BM25Retriever.from_defaults(nodes=self.knowledge_base,
#                                                     similarity_top_k=self.top_k)
#         else:
#             print("RetrieverModule Error: Select a valid retriever.")
#             retriever = None
#         return retriever

#     def retrieve(self, query: str):
#         """
#         TODO
#         """
#         return self.retriever.retrieve(query.strip())

#     def set_top_k(self, top_k):
#         """
#         TODO
#         """
#         self.top_k = top_k
#         self.retriever = self.set_retriever()


# class Retriever_bm25:
#     """
#     BM25 retriever that can be used with any knowledge base 
#     (both llama index nodes and generic text nodes as defined in chunker)
#     """
#     def __init__(self,
#                  knowledge_base,
#                  name='BM25',
#                  top_k=3):

#         self.name = name
#         self.top_k = top_k
#         self.knowledge_base = knowledge_base.nodes
#         self.retriever = self.set_retriever()
        
#     def set_retriever(self):
#         if self.name == 'BM25':
#             corpus = [doc.text for doc in self.knowledge_base]
#             tokenized_docs = [doc.split(" ") for doc in corpus]
#             retriever = BM25Okapi(tokenized_docs)

#         else:
#             print("RetrieverModule Error: Select a valid retriever.")
#             retriever = None
#         return retriever
    
#     def retrieve(self, query: str):
#         """
#         Retrieve top n most similar chunks to the query
#         """      
#         text_metadata_dict = {node.dict()['text']:node.dict()['metadata'] for node in self.knowledge_base}
#         corpus = list(text_metadata_dict.keys())
#         tokenized_query = query.split(" ")
#         retrieved_docs = self.retriever.get_top_n(tokenized_query, corpus, n=self.top_k)
#         return [TextNode(text = retrieved_doc.strip(), metadata = text_metadata_dict[retrieved_doc]) for retrieved_doc in retrieved_docs]

# class Retriever_BGE_v2_m3:
#     """
#     bge-reranker-v2-m3
#     """
#     def __init__(self,
#                  knowledge_base,
#                  name='BGE',
#                  top_k=3):

#         self.name = name
#         self.top_k = top_k
#         self.knowledge_base = knowledge_base.nodes
#         self.retriever = self.set_retriever()
#         self.tokenizer = self.set_tokenizer()

#     def set_tokenizer(self):
#         tokenizer = tokenizer_bge
#         return tokenizer

#     def set_retriever(self):
#         if self.name == 'BGE':
#             retriever = retriever_bge
#             retriever.eval()
#         else:
#             print("RetrieverModule Error: Select a valid retriever.")
#             retriever = None
#         return retriever

#     def retrieve(self, query: str):
#         """
#         TODO
#         """
#         text_metadata_dict = {node.dict()['text']:node.dict()['metadata'] for node in self.knowledge_base}
#         corpus = list(text_metadata_dict.keys())
#         pairs = [[query, k] for k in text_metadata_dict.keys()]
#         # scores = {k:self.retriever.compute_score([query, k])[0] for k in text_metadata_dict.keys()}

#         with torch.no_grad():
#             inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
#             scores = self.retriever(**inputs, return_dict=True).logits.view(-1, ).float()

#         scores_dict = dict(zip(text_metadata_dict.keys(), scores.tolist()))
#         sorted_values = sorted(scores_dict.values(), reverse=True)
#         retrieved_docs = [next((k for k, v in scores_dict.items() if v == value), None) for value in sorted_values[:self.top_k]]
#         return [TextNode(text = retrieved_doc.strip(), metadata = text_metadata_dict[retrieved_doc]) for retrieved_doc in retrieved_docs]

#     def set_top_k(self, top_k):
#         """
#         TODO
#         """
#         self.top_k = top_k
#         self.retriever = self.set_retriever()