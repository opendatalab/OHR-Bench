import os
import re
import datetime
from src.tasks.base import BaseTask
from loguru import logger
from src.metric.common import lcs_score


class RetrievalTask(BaseTask):
    def __init__(
            self, 
            output_dir: str = './output',
        ):
        
        if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
            os.makedirs(output_dir)
    
    def set_model(self, model, retriever) -> None:
        self.model = model
        self.retriever = retriever
    
    def retrieve_docs(self, obj:dict) -> str:
        retrieval_results = self.retriever.search_docs(obj)
        return retrieval_results

    def scoring(self, data_point: dict) -> dict:
        hits_results = []
        doc_name = "/".join(data_point["doc_name"].split("/")[1:])
        page_idx = data_point["evidence_page_no"]
        if not isinstance(page_idx, list):
            page_idx = [page_idx]
        ret_context = [r["text"] for r in data_point["retrieval_results"] if r["file_name"] == doc_name and r["page_idx"] in page_idx]
        gt_context = data_point["evidence_context"]
        if isinstance(gt_context, list):
            gt_context = "\n".join(gt_context)
        if len(ret_context) > 0:
            lcs = lcs_score("\n\n".join(ret_context), gt_context)
        else:
            lcs = 0

        return {
            'metrics': {
                'lcs': lcs
            },
            'log': {
                'quest': data_point["questions"],
                'retrieval_context': data_point["retrieval_results"],
                'ground_truth_context': gt_context,
                'evidence_source': data_point["evidence_source"],
                'evaluateDatetime': str(datetime.datetime.now()),
            },
            'valid': len(data_point['retrieval_results']) != 0
        }
    
    def compute_overall(self, results: list[dict]) -> dict:
        overall = {'lcs': 0.0}
        for result in results:
            overall = {key: overall[key] + result['metrics'][key] for key in overall.keys()}
        
        overall_save = {f'avg. {key}': value / len(results) for key, value in overall.items()}
        overall_save['num'] = len(results)
       
        return overall_save
