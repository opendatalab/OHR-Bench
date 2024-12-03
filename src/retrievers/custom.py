from abc import ABC
from operator import itemgetter

from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.retrievers import VectorIndexRetriever, QueryFusionRetriever, BM25Retriever
from llama_index.node_parser import SimpleNodeParser
from llama_index.storage.docstore import SimpleDocumentStore

from llama_index.embeddings import LangchainEmbedding
from llama_index import ServiceContext, StorageContext
from langchain.schema.embeddings import Embeddings

from src.retrievers.json_reader import PCJSONReader

class CustomBGEM3Retriever(ABC):
    def __init__(
            self, 
            docs_directory: str, 
            embed_model: Embeddings,
            embed_dim: int = 768,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            similarity_top_k: int=2,
        ):
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k

        self.construct_index()

        retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=self.similarity_top_k,
        )

        self.retriever = retriever

    def construct_index(self):
        documents = SimpleDirectoryReader(self.docs_directory, file_extractor={".json": PCJSONReader()}).load_data(num_workers=8)
        
        # node_parser = HierarchicalNodeParser.from_defaults()
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True, num_workers=8)

        # leaf_nodes = get_leaf_nodes(nodes)

        self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        doc_store = SimpleDocumentStore()
        doc_store.add_documents(nodes)

        storage_context = StorageContext.from_defaults(docstore=doc_store)
        
        # Process and index nodes in chunks due to Milvus limitations
        self.vector_index = VectorStoreIndex(
            nodes, service_context=service_context, 
            storage_context=storage_context, show_progress=True
        )

        print("Indexing finished!")

    def search_docs(self, query_text: str):
        response_nodes = self.retriever.retrieve(query_text)

        results = [{
            "text": node.get_content(),
            "page_idx": node.metadata.get("page_idx", None),
            "file_name": node.metadata.get("file_name", "").replace(".json", "")
        } for node in response_nodes]
        
        return results


class CustomBM25Retriever(ABC):
    def __init__(
        self, 
        docs_directory: str,
        chunk_size: int = 128,
        chunk_overlap: int = 0,
        similarity_top_k: int=2,
        ):
        self.docs_directory = docs_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k

        self.construct_index()

        # self.query_engine = self.vector_index.as_query_engine()
        retriever = BM25Retriever.from_defaults(
            nodes=self.nodes,
            similarity_top_k=self.similarity_top_k,
        )

        self.retriever = retriever

        # assemble query engine
        # self.query_engine = RetrieverQueryEngine(
        #     retriever=retriever,
        # )

    def construct_index(self):
        documents = SimpleDirectoryReader(self.docs_directory, file_extractor={".json": PCJSONReader()}).load_data()
        
        # node_parser = HierarchicalNodeParser.from_defaults()
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

        # leaf_nodes = get_leaf_nodes(nodes)

        print("Indexing finished!")

    def search_docs(self, query_text: str):
        response_nodes = self.retriever.retrieve(query_text)

        results = [{
            "text": node.get_content(),
            "page_idx": node.metadata.get("page_idx", None),
            "file_name": node.metadata.get("file_name", "").replace(".json", "")
        } for node in response_nodes]
        
        return results


class CustomHybridRetriever(ABC):
    def __init__(
            self, 
            docs_directory: str, 
            embed_model: Embeddings,
            embed_dim: int = 768,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            similarity_top_k: int=2,
        ):
        self.weights = [0.5, 0.5]
        self.c: int = 60
        self.top_k = similarity_top_k
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k

        self.construct_index()

        # self.query_engine = self.vector_index.as_query_engine()
        self.embedding_retriever = VectorIndexRetriever(index=self.vector_index, similarity_top_k=self.similarity_top_k)
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=self.nodes, similarity_top_k=self.similarity_top_k)

    def construct_index(self):
        documents = SimpleDirectoryReader(self.docs_directory, file_extractor={".json": PCJSONReader()}).load_data()

        # node_parser = HierarchicalNodeParser.from_defaults()
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

        # leaf_nodes = get_leaf_nodes(nodes)
        self.nodes = nodes

        self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        doc_store = SimpleDocumentStore()
        doc_store.add_documents(nodes)

        storage_context = StorageContext.from_defaults(docstore=doc_store)
        
        # Process and index nodes in chunks due to Milvus limitations
        self.vector_index = VectorStoreIndex(
            nodes, service_context=service_context, 
            storage_context=storage_context, show_progress=True
        )

        print("Indexing finished!")

    def search_docs(self, query_text: str):
        bm25_search_docs = self.bm25_retriever.retrieve(query_text)
        embedding_search_docs = self.embedding_retriever.retrieve(query_text)

        doc_lists = [bm25_search_docs, embedding_search_docs]

        # Create a union of all unique documents in the input doc_lists
        all_documents = set()
        for doc_list in doc_lists:
            for doc in doc_list:
                all_documents.add(doc)

        # Initialize the RRF score dictionary for each document
        rrf_score_dic = {doc: 0.0 for doc in all_documents}

        # Calculate RRF scores for each document
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score = weight * (1 / (rank + self.c))
                rrf_score_dic[doc] += rrf_score

        # Sort documents by their RRF scores in descending order
        sorted_documents = sorted(rrf_score_dic.items(), key=itemgetter(1), reverse=True)
        results = []
        for node in sorted_documents[:self.top_k]:
            results.append({
                "text": node.get_content(),
                "page_idx": node.metadata.get("page_idx", None),
                "file_name": node.metadata.get("file_name", "").replace(".json", "")
            })

        return results
