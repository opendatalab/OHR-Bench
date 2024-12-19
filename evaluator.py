import copy
import json
import os
from abc import ABC
from loguru import logger
from tqdm import tqdm
from threading import Lock
from src.llms.base import BaseLLM
from src.tasks.base import BaseTask
from src.retrievers.base import BaseRetriever
import concurrent.futures

class BaseEvaluator(ABC):
    def __init__(self, task: BaseTask, model: BaseLLM, retriever: BaseRetriever,
        dataset: list[dict], output_dir: str = './output', num_threads: int = 40):
        """
        Args:
            model (BaseLLM): The large language model to be evaluated.
            retriever (BaseRetriever): The retriever to be evaluated.
            task (BaseTask): The task for evaluation.
            dataset (list[dict]): The dataset for evaluation.
            output_dir (str): The directory for result output and caching.
        """
        self.model = model
        self.retriever = retriever
        self.dataset = dataset
        self.task = task
        self.lock = Lock()
        self.num_threads = num_threads

        collection_name = self.retriever.collection_name
        similarity_top_k = self.retriever.similarity_top_k
        output_dir = os.path.join(output_dir, f'{collection_name}_top{similarity_top_k}_{model.__class__.__name__}')
        
        if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
            os.makedirs(output_dir)
        self.output_path = os.path.join(
            output_dir, f'{self.task.__class__.__name__}_{model.params["model_name"]}.json'
        )
        if os.path.exists(self.output_path):
            logger.warning(f'Output file already exists at {self.output_path}. Removing...')
            os.remove(self.output_path)
        self.task.set_model(self.model, self.retriever)

    def task_generation(self, data_point):
        try:
            self.lock.acquire()
            retrieve_context = self.task.retrieve_docs(data_point)
            self.lock.release()
            data_point["retrieve_context"] = retrieve_context

        except Exception as e:
            import traceback
            logger.warning(repr(e))
            logger.warning(traceback.format_exc())
            self.lock.release()
            data_point["retrieve_context"] = ''

        return self.task.model_generation(data_point)

    def multithread_batch_scoring(self, dataset: list[dict], sort=True, show_progress_bar=False, contain_original_data=False) -> list[dict]:
        """Perform batch scoring on the given dataset.

        Args:
            dataset (list[dict]): The dataset for evaluation.
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.

        Returns:
            list[dict]: List of results.
        """

        if os.path.exists(self.output_path):  # Resume evaluation
            results = self.read_output().get('results', [])
            results = self.remove_invalid(results)
            saved_ids = [result['id'] for result in results]
        else:
            results = []
            saved_ids = []

        def process_data_point(data_point):
            if data_point['ID'] in saved_ids:
                return None  # Skip results that have already been evaluated and are valid
            try:
                generated_text = self.task_generation(data_point)
                # TODO fix bugs
                if generated_text == '","msg":"request openai failed"':
                    return None
                
                data_point["generated_text"] = generated_text
                result = {'id': data_point['ID'], **self.task.scoring(data_point)}
                
                if contain_original_data:
                    result['original_data'] = data_point
                result['log']['retrieve_context'] = data_point.get('retrieve_context', '')

                return result
            
            except Exception as e:
                logger.warning(repr(e))
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_results = list(tqdm(executor.map(process_data_point, dataset), total=len(dataset)))
        
        results.extend([result for result in future_results if result is not None])
        
        return sorted(results, key=lambda x: x['id']) if sort else results

    def save_output(self, output: dict) -> None:
        """Save evaluation results."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
    
    def read_output(self) -> dict:
        with open(self.output_path) as f:
            return json.load(f)

    def run(self, sort = True, show_progress_bar = False, contain_original_data = True) -> dict:
        """Run a complete evaluation.

        Args:            
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.
            contain_original_data (bool): Whether to include original data in the results for debugging.

        Returns:
            dict: Output dictionary contains fields such as: info, overall, results, etc.
        """
        info = {
            'task': self.task.__class__.__name__, 
            'llm': str(self.model.params),
        }

        results = self.multithread_batch_scoring(self.dataset, sort, show_progress_bar, contain_original_data)
        valid_results = self.remove_invalid(results)

        try:
            overall = self.task.compute_overall(valid_results) if len(valid_results) > 0 else {}\
        
        except Exception as e:
            logger.warning(repr(e))
            overall = dict()

        self.save_output(output:={'info': info, 'overall': overall, 'results': results})
        print(f'Output saved at {self.output_path}!')
        return output

    @staticmethod
    def remove_invalid(results: list[dict]) -> list[dict]:
        """Remove invalid results from the list and return the cleaned results."""
        return [result for result in results if result['valid']]

    def batch_scoring(self, dataset:list[dict], sort = True, show_progress_bar = False, contain_original_data = False):
        """Perform batch scoring on the given dataset.
        
        Args:
            dataset (list[dict]): The dataset for evaluation.
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.
        
        Returns:
            list[dict]: List of results.
        """
        
        if os.path.exists(self.output_path):  # Resume evaluation
            results = self.read_output().get('results', [])
            results = self.remove_invalid(results)
            saved_ids = [result['id'] for result in results]
        else:
            results = []
            saved_ids = []

        for data_point in (tqdm(dataset, desc=self.model.params['model_name']) if show_progress_bar else dataset):
            if data_point['ID'] in saved_ids:
                continue  # Skip results that have already been evaluated and are valid
            try:
                generated_text = self.task_generation(data_point)
                data_point["generated_text"] = generated_text
                result = {'id': data_point['ID'], **self.task.scoring(data_point)}
                if contain_original_data:
                    result['original_data'] = data_point
                results.append(result)
            except Exception as e:
                logger.warning(repr(e))

        return sorted(results, key=lambda x: x['id']) if sort else results


class StageEvaluator(BaseEvaluator):
    def __init__(self, task: BaseTask, model: BaseLLM, retriever: BaseRetriever,
        dataset: list[dict], output_dir: str = './output/context', output_name = "",
        num_threads: int = 40, stage: str = 'all'):
        self.model = model
        self.retriever = retriever
        self.dataset = dataset
        self.task = task
        self.lock = Lock()
        self.num_threads = num_threads
        self.stage = stage

        if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
            os.makedirs(output_dir)
        self.output_dir = os.path.dirname(output_dir)
        if self.stage == "generation":
            self.output_path = os.path.join(
                output_dir, f'{output_name}_{model.params["model_name"]}.json'
            )
        elif self.stage == "retrieval":
            ret = {
                "CustomBM25Retriever": "bm25",
                "CustomBGEM3Retriever": "bge-m3"
            }[self.retriever.__class__.__name__]
            self.output_path = os.path.join(
                output_dir, f'{output_name}_{ret}_top{self.retriever.similarity_top_k}.json'
            )
        elif self.stage == "end2end":
            ret = {
                "CustomBM25Retriever": "bm25",
                "CustomBGEM3Retriever": "bge-m3"
            }[self.retriever.__class__.__name__]
            self.output_path = os.path.join(
                output_dir, f'{output_name}_{ret}_top{self.retriever.similarity_top_k}_{model.params["model_name"]}.json'
            )
        else:
            raise NotImplementedError()
        self.task.set_model(self.model, self.retriever)

    def multithread_batch_retrieval(self, dataset: list[dict], sort=True, show_progress_bar=False, contain_original_data=False) -> list[dict]:
        """Perform batch retrieval on the given dataset.

        Args:
            dataset (list[dict]): The dataset for evaluation.
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.

        Returns:
            list[dict]: List of results.
        """

        if os.path.exists(self.output_path):  # Resume evaluation
            results = []
            results = self.remove_invalid(results)
            saved_ids = [result['id'] for result in results]
        else:
            results = []
            saved_ids = []

        def process_data_point(data_point):
            if data_point['ID'] in saved_ids:
                return None  # Skip results that have already been evaluated and are valid
            try:
                self.lock.acquire()
                retrieval_results = self.task.retrieve_docs(data_point)
                self.lock.release()
                data_point["retrieval_results"] = retrieval_results
                result = {'id': data_point['ID'], **self.task.scoring(data_point)}
                return result
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.warning(repr(e))
                self.lock.release()
                data_point["retrieval_results"] = []
                result = {'id': data_point['ID'], **self.task.scoring(data_point)}
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_results = list(tqdm(executor.map(process_data_point, dataset), total=len(dataset)))
        
        results.extend([result for result in future_results if result is not None])
        
        return sorted(results, key=lambda x: x['id']) if sort else results

    def run(self, sort = True, show_progress_bar = False, contain_original_data = True) -> dict:
        if self.stage == 'retrieval':
            info = {
                'task': self.task.__class__.__name__, 
                'retriever': str(self.retriever.__class__.__name__),
            }
            results = self.multithread_batch_retrieval(self.dataset, sort, show_progress_bar, contain_original_data)
            valid_results = self.remove_invalid(results)
            try:
                overall = self.task.compute_overall(valid_results) if len(valid_results) > 0 else {}
            except Exception as e:
                logger.warning(repr(e))
                overall = dict()
            # TODO: FIX the path
            # output_context_path = self.output_path.replace("./output/retrieval", "./data/qa_retrieval")
            # os.makedirs(os.path.dirname(output_context_path), exist_ok=True)
            # with open(output_context_path, "w") as f:
            #     lines = []
            #     for data in self.dataset:
            #         line = {k:v for k,v in data.items() if k not in ["retrieval_results", "context"]}
            #         # line["context"] = "\n".join([r["text"] for r in id2retrieval_text[data["ID"]]])
            #         line["context"] = "\n".join([r["text"] for r in data["retrieval_results"]])
            #         lines.append(line)
            #     json.dump(lines, f, indent=2)
            #     print(f'Retrieved context saved at {output_context_path}!')
            self.save_output(output:={'info': info, 'overall': overall, 'results': results})
            print(f'Output saved at {self.output_path}!')
        elif self.stage == 'generation' or self.stage == 'end2end':
            info = {
                'task': self.task.__class__.__name__, 
                'llm': str(self.model.params),
            }
            results = self.multithread_batch_scoring(self.dataset, sort, show_progress_bar, contain_original_data)
            valid_results = self.remove_invalid(results)
            try:
                overall = self.task.compute_overall(valid_results) if len(valid_results) > 0 else {}
            except Exception as e:
                logger.warning(repr(e))
                overall = dict()

            self.save_output(output:={'info': info, 'overall': overall, 'results': results})
            print(f'Output saved at {self.output_path}!')
        else:
            raise NotImplementedError()

        return output
