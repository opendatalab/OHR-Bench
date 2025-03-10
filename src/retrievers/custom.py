from abc import ABC
from operator import itemgetter
import os

from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.node_parser import SimpleNodeParser
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.embeddings import LangchainEmbedding
from llama_index import ServiceContext, StorageContext
from langchain.schema.embeddings import Embeddings
from src.retrievers.json_reader import PCJSONReader

def construct_retriever(docs_directory, embed_model=None, chunk_size=128, chunk_overlap=0, similarity_top_k=2):
    documents = SimpleDirectoryReader(docs_directory, file_extractor={".json": PCJSONReader()}, recursive=True).load_data(num_workers=8)
    node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True, num_workers=8)
    
    if embed_model:
        embed_model_instance = LangchainEmbedding(embed_model)
        service_context = ServiceContext.from_defaults(embed_model=embed_model_instance, llm=None)
        doc_store = SimpleDocumentStore()
        doc_store.add_documents(nodes)
        storage_context = StorageContext.from_defaults(docstore=doc_store)
        vector_index = VectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context, show_progress=True)
        return VectorIndexRetriever(index=vector_index, similarity_top_k=similarity_top_k)
    else:
        return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=similarity_top_k)

class CustomBGEM3Retriever(ABC):
    def __init__(self, docs_directory, embed_model, embed_dim=768, chunk_size=128, chunk_overlap=0, similarity_top_k=2):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.retrievers = self.construct_retrievers(docs_directory, embed_model)

    def construct_retrievers(self, docs_directory, embed_model):
        retrievers = {}
        is_subdir_present = False
        for subdir in os.listdir(docs_directory):
            subdir_path = os.path.join(docs_directory, subdir)
            if os.path.isdir(subdir_path):
                is_subdir_present = True
                retrievers[subdir] = construct_retriever(subdir_path, embed_model, self.chunk_size, self.chunk_overlap, self.similarity_top_k)
        if not is_subdir_present:
            retrievers['default'] = construct_retriever(docs_directory, embed_model, self.chunk_size, self.chunk_overlap, self.similarity_top_k)
        
        print("Indexing finished for all directories!")
        return retrievers

    def search_docs(self, query: dict):
        doc_name = query.get("doc_name")
        query_text = query.get("questions")
        subdir = doc_name.split('/')[0]
        retriever = self.retrievers.get(subdir, self.retrievers.get('default'))
        if not retriever:
            raise ValueError(f"No retriever found for directory: {subdir}")

        response_nodes = retriever.retrieve(query_text)
        return [{
            "text": node.get_content(),
            "page_idx": node.metadata.get("page_idx", None),
            "file_name": node.metadata.get("file_name", "").replace(".json", "")
        } for node in response_nodes]

class CustomBM25Retriever(ABC):
    def __init__(self, docs_directory, chunk_size=128, chunk_overlap=0, similarity_top_k=2):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.retrievers = self.construct_retrievers(docs_directory)

    def construct_retrievers(self, docs_directory):
        retrievers = {}
        is_subdir_present = False
        for subdir in os.listdir(docs_directory):
            subdir_path = os.path.join(docs_directory, subdir)
            if os.path.isdir(subdir_path):
                is_subdir_present = True
                retrievers[subdir] = construct_retriever(subdir_path, None, self.chunk_size, self.chunk_overlap, self.similarity_top_k)
        if not is_subdir_present:
            retrievers['default'] = construct_retriever(docs_directory, None, self.chunk_size, self.chunk_overlap, self.similarity_top_k)
        
        print("Indexing finished for all directories!")
        return retrievers

    def search_docs(self, query: dict):
        doc_name = query.get("doc_name")
        query_text = query.get("questions")
        subdir = doc_name.split('/')[0]
        retriever = self.retrievers.get(subdir, self.retrievers.get('default'))
        if not retriever:
            raise ValueError(f"No retriever found for directory: {subdir}")

        response_nodes = retriever.retrieve(query_text)
        return [{
            "text": node.get_content(),
            "page_idx": node.metadata.get("page_idx", None),
            "file_name": node.metadata.get("file_name", "").replace(".json", "")
        } for node in response_nodes]

class CustomHybridRetriever(ABC):
    def __init__(self, docs_directory, embed_model, embed_dim=768, chunk_size=128, chunk_overlap=0, similarity_top_k=2):
        self.weights = [0.5, 0.5]
        self.c: int = 60
        self.top_k = similarity_top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retrievers = self.construct_retrievers(docs_directory, embed_model)

    def construct_retrievers(self, docs_directory, embed_model):
        retrievers = {}
        is_subdir_present = False
        for subdir in os.listdir(docs_directory):
            subdir_path = os.path.join(docs_directory, subdir)
            if os.path.isdir(subdir_path):
                is_subdir_present = True
                retrievers[subdir] = self.construct_index_for_subdir_pair(subdir_path, embed_model)
        if not is_subdir_present:
            retrievers['default'] = self.construct_index_for_subdir_pair(docs_directory, embed_model)
        
        print("Indexing finished for all directories!")
        return retrievers

    def construct_index_for_subdir_pair(self, subdir_path, embed_model):
        documents = SimpleDirectoryReader(subdir_path, file_extractor={".json": PCJSONReader()}, recursive=True).load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        embed_model_instance = LangchainEmbedding(embed_model)
        service_context = ServiceContext.from_defaults(embed_model=embed_model_instance, llm=None)
        doc_store = SimpleDocumentStore()
        doc_store.add_documents(nodes)
        storage_context = StorageContext.from_defaults(docstore=doc_store)
        
        vector_index = VectorStoreIndex(nodes, service_context=service_context, storage_context=storage_context, show_progress=True)
        embedding_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=self.similarity_top_k)
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=self.similarity_top_k)
        return (embedding_retriever, bm25_retriever)

    def search_docs(self, query: dict):
        doc_name = query.get("doc_name")
        query_text = query.get("questions")
        subdir = doc_name.split('/')[0]
        retriever_pair = self.retrievers.get(subdir, self.retrievers.get('default'))
        if not retriever_pair:
            raise ValueError(f"No retrievers found for directory: {subdir}")

        embedding_retriever, bm25_retriever = retriever_pair
        bm25_search_docs = bm25_retriever.retrieve(query_text)
        embedding_search_docs = embedding_retriever.retrieve(query_text)

        doc_lists = [bm25_search_docs, embedding_search_docs]

        all_documents = set()
        for doc_list in doc_lists:
            for doc in doc_list:
                all_documents.add(doc)

        rrf_score_dic = {doc: 0.0 for doc in all_documents}

        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score_dic[doc] += weight * (1 / (rank + self.c))

        sorted_documents = sorted(rrf_score_dic.items(), key=itemgetter(1), reverse=True)
        return [{
            "text": node.get_content(),
            "page_idx": node.metadata.get("page_idx", None),
            "file_name": node.metadata.get("file_name", "").replace(".json", "")
        } for node in sorted_documents[:self.top_k]]

class CustomPageRetriever(ABC):
    def __init__(self, docs_directory: str):
        self.documents = self.construct_index(docs_directory)

    def construct_index(self, docs_directory):
        documents_by_subdir = {}
        is_subdir_present = False
        for subdir in os.listdir(docs_directory):
            subdir_path = os.path.join(docs_directory, subdir)
            if os.path.isdir(subdir_path):
                is_subdir_present = True
                documents = SimpleDirectoryReader(subdir_path, file_extractor={".json": PCJSONReader()}, recursive=True).load_data(num_workers=8)
                documents_by_subdir[subdir] = documents
        if not is_subdir_present:
            documents = SimpleDirectoryReader(docs_directory, file_extractor={".json": PCJSONReader()}, recursive=True).load_data(num_workers=8)
            documents_by_subdir['default'] = documents
        return documents_by_subdir

    def search_docs(self, query: dict):
        doc_name = query.get("doc_name")
        evidence_page_no = query.get("evidence_page_no")
        if not isinstance(evidence_page_no, list):
            evidence_page_no = [evidence_page_no]
        subdir = doc_name.split('/')[0]
        documents = self.documents.get(subdir, self.documents.get('default'))

        def get_doc_name(metadata: dict):
            return os.path.basename(os.path.dirname(metadata.get("file_path", ""))) + "/" + metadata.get("file_name", "").replace(".json", "")

        if not documents:
            raise ValueError(f"No documents found for directory: {subdir}")

        results = [
            {
                "text": doc.text,
                "page_idx": doc.metadata.get("page_idx", None),
                "file_name": doc.metadata.get("file_name", "").replace(".json", "")
            }
            for doc in documents
            if get_doc_name(doc.metadata) == doc_name and doc.metadata.get("page_idx") in evidence_page_no
        ]

        return results
