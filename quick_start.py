import argparse, os
from loguru import logger
from src.datasets.dataset import get_task_datasets
from evaluator import StageEvaluator
from src.llms import GPT
from src.llms import Qwen2_7B_Instruct, LLaMA31_8B_Instruct, Qwen25_7B_Instruct, Qwen2VL_7B_Instruct, InternVL2_8B_Instruct
from src.llms import Mock

from src.tasks.quest_answer import QuestAnswer, QuestAnswer_image, QuestAnswer_OCR, QuestAnswer_image_OCR
from src.tasks.retrieval import RetrievalTask
from src.retrievers import CustomBM25Retriever, CustomBGEM3Retriever, CustomPageRetriever
from src.embeddings.base import HuggingfaceEmbeddings

parser = argparse.ArgumentParser()

# Model related options
parser.add_argument('--model_name', default='qwen7b', help="Name of the model to use")
parser.add_argument('--temperature', type=float, default=0.1, help="Controls the randomness of the model's text generation")
parser.add_argument('--max_new_tokens', type=int, default=1280, help="Maximum number of new tokens to be generated by the model")

# Dataset related options
parser.add_argument('--data_path', default='data/qas.json', help="Path to the dataset")
parser.add_argument('--shuffle', type=bool, default=True, help="Whether to shuffle the dataset")
parser.add_argument('--ocr_type', type=str, default="gt")
parser.add_argument('--output_path', type=str, default="./output")

# Retriever related options
parser.add_argument('--retriever', type=str, default="bge-m3")
parser.add_argument('--retrieve_top_k', type=int, default=1)
parser.add_argument('--docs_path', type=str, default="data/retrieval_base/gt")

# Evaluation related options
parser.add_argument('--task', default='QA', help="Task to perform")
parser.add_argument('--num_threads', type=int, default=1, help="Number of threads")
parser.add_argument('--show_progress_bar', action='store', default=True, type=bool, help="Whether to show a progress bar")
parser.add_argument('--contain_original_data', action='store_true', help="Whether to contain original data")

parser.add_argument('--evaluation_stage', default='generation', choices=['retrieval', 'generation', 'end2end'], help="Which stage to be evaluated in RAG")

args = parser.parse_args()
logger.info(args)

def setup_seed(seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

if args.evaluation_stage == "retrieval":
    assert args.retriever in ["bge-m3", "bm25"], f"The retriever {args.retriever} is not supported for stage {args.evaluation_stage}."
    args.task = "Retrieval"
    args.model_name = "mock"
elif args.evaluation_stage == "generation":
    assert args.model_name not in ["mock"], f"The model_name {args.model_name} is not supported for stage {args.evaluation_stage}."
    assert args.task in ["QA", "QA_image", "QA_OCR", "QA_image_OCR"], f"The task {args.task} is not supported for stage {args.evaluation_stage}."
    args.retriever = "page"
elif args.evaluation_stage == "end2end":
    assert args.retriever in ["bge-m3", "bm25"], f"The retriever {args.retriever} is not supported for stage {args.evaluation_stage}."
    assert args.task in ["QA", "QA_image", "QA_OCR", "QA_image_OCR"], f"The task {args.task} is not supported for stage {args.evaluation_stage}."


if args.model_name.startswith("gpt"):
    llm = GPT(model_name=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
elif args.model_name == "qwen2_7b":
    llm = Qwen2_7B_Instruct(model_name=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                            top_p=0.8, top_k=10)
elif args.model_name == "qwen25_7b":
    llm = Qwen25_7B_Instruct(model_name=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                            top_p=0.8, top_k=10)
elif args.model_name == "llama3.1_8b":
    llm = LLaMA31_8B_Instruct(model_name=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                            top_p=0.8, top_k=10)
elif args.model_name == "qwen2vl_7b":
    llm = Qwen2VL_7B_Instruct(model_name=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                            top_p=0.8, top_k=10)
elif args.model_name == "internvl2_8b":
    llm = InternVL2_8B_Instruct(model_name=args.model_name, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
                            top_p=0.8, top_k=10)
elif args.model_name == "mock":
    llm = Mock()

if args.retriever == "bge-m3":
    embed_model = HuggingfaceEmbeddings(model_name="BAAI/bge-m3")
    retriever = CustomBGEM3Retriever(
        args.docs_path, embed_model=embed_model, embed_dim=1024,
        chunk_size=1024, chunk_overlap=0, similarity_top_k=args.retrieve_top_k
    )
elif args.retriever == "bm25":
    retriever = CustomBM25Retriever(
        args.docs_path, chunk_size=1024, chunk_overlap=0, similarity_top_k=args.retrieve_top_k
    )
elif args.retriever == "page":
    retriever = CustomPageRetriever(
        args.docs_path
    )
else:
    raise NotImplementedError()

task_mapping = {
    'QA': [QuestAnswer],
    'QA_image': [QuestAnswer_image],
    'QA_OCR': [QuestAnswer_OCR],
    'QA_image_OCR': [QuestAnswer_image_OCR],
    'Retrieval': [RetrievalTask],
}

if args.task not in task_mapping:
    raise ValueError(f"Unknown task: {args.task}")

tasks = [task() for task in task_mapping[args.task]]

datasets = get_task_datasets(args.data_path, args.task)

for task, dataset in zip(tasks, datasets):
    evaluator = StageEvaluator(task, llm, retriever, dataset, 
                               output_dir=os.path.join(args.output_path, args.evaluation_stage,args.ocr_type),
                               output_name="all",
                               num_threads=args.num_threads,
                               stage=args.evaluation_stage)
    results = evaluator.run(show_progress_bar=args.show_progress_bar, contain_original_data=args.contain_original_data)