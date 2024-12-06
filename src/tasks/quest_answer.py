import os
import re
import datetime
from src.tasks.base import BaseTask
from loguru import logger
from src.metric.common import (
    bleu_score, 
    rougeL_score, 
    bert_score,
    exact_match_score,
    f1_score
)


class QuestAnswer(BaseTask):
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
        query_text = obj["questions"]
        retrieve_results = self.retriever.search_docs(obj)
        retrieve_context = "\n\n".join([
            r["text"].split('\nGiven the context information')[0]
            for r in retrieve_results
        ])
        retrieve_context = retrieve_context.split('\nGiven the context information')[0]
        return retrieve_context

    def model_generation(self, obj:dict): 
        template = self._read_prompt_template('QA_prompt.txt')
        query = [
            {
                "role": "system",
                "content": template,
            },
            {
                "role": "user", 
                "content": "Question: {question}\n\nRetrieved Documents: {search_documents}".format(
                    question=f'{obj["questions"]}',
                    search_documents=f'{obj["retrieve_context"]}'
                )
            },
        ]
        res = self.model.safe_request(query)
        real_res = res.split('<response>')[-1].split('</response>')[0]
        if real_res.strip() == "":
            special_match = re.search(r'</?response>(.*?)</?response>', res, re.DOTALL)
            if special_match:
                real_res = special_match.group(1).strip()
        return real_res.strip()

    def _read_prompt_template(self, filename: str):
        path = os.path.join('src/prompts/', filename)
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''

    def scoring(self, data_point: dict) -> dict:
        generated_text = data_point["generated_text"]
        ground_truth_text = data_point["answers"]
        data_point["ground_truth_text"] = ground_truth_text
        
        # if self.use_quest_eval:
        #     QA_avg_F1, QA_recall, quest_eval_save = self.quest_eval.quest_eval(data_point)
        # else:
        QA_avg_F1, QA_recall, quest_eval_save = 0.0, 0.0, {}
        
        # if self.use_bert_score:
        #     bertscore = bert_score(generated_text, ground_truth_text)
        # else:
        bertscore = 0.0
        
        bleu_avg, bleu1, bleu2, bleu3, bleu4 = bleu_score(generated_text, ground_truth_text)

        em = exact_match_score(generated_text, ground_truth_text)
        f1 = f1_score(generated_text, ground_truth_text)

        return {
            'metrics': {
                'bleu-avg': bleu_avg or 0.0,
                'bleu-1': bleu1 or 0.0,
                'bleu-2': bleu2 or 0.0,
                'bleu-3': bleu3 or 0.0,
                'bleu-4': bleu4 or 0.0,
                'rouge-L': rougeL_score(generated_text, ground_truth_text) or 0.0,
                'bertScore': bertscore,
                'F1': f1,
                'EM': em,
                'QA_avg_F1': QA_avg_F1,
                'QA_recall': QA_recall,
                'length': len(generated_text)
            },
            'log': {
                'quest': data_point["questions"],
                'generated_text': generated_text,
                'ground_truth_text': ground_truth_text,
                'quest_eval_save': quest_eval_save,
                'evidence_source': data_point["evidence_source"],
                'evaluateDatetime': str(datetime.datetime.now()),
            },
            'valid': len(generated_text.strip()) != 0
        }

    def compute_overall(self, results: list[dict]) -> dict:
        overall = {'bleu-avg': 0, 'bleu-1': 0, 'bleu-2': 0, 'bleu-3': 0, 
                   'bleu-4': 0, 'rouge-L': 0, 'bertScore': 0, 'F1': 0, 'EM': 0,
                   'QA_avg_F1': 0, 'QA_recall': 0, 'length': 0}
        
        for result in results:
            overall = {key: overall[key] + result['metrics'][key] for key in overall.keys()}
        
        overall_save = {f'avg. {key}': value / len(results) for key, value in overall.items() if key != 'QA_avg_F1' and key != 'QA_recall'}

        overall_save['num'] = len(results)
       
        return overall_save


class QuestAnswer_image(QuestAnswer):
    def __init__(self, output_dir: str = './output'):
        super().__init__(output_dir)

    def model_generation(self, obj: dict):
        template = self._read_prompt_template('QA_image_prompt.txt')
        image_path = "image_data/{doc_name}_page_{page}.jpg".format(
            doc_name=obj["doc_name"],page = obj["evidence_page_no"]
        )

        query = [
            {
                "role": "system",
                "content": template,
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type":"text",
                        "text":"Question: {question}\n".format(question=obj["questions"])
                    },
                    {
                        "type": "image",
                        "image": image_path,
                    }
                ]
            },
        ]
        
        res = self.model.safe_request(query)
        real_res = res.split('<response>')[-1].split('</response>')[0]
        if real_res.strip() == "":
            special_match = re.search(r'</?response>(.*?)</?response>', res, re.DOTALL)
            if special_match:
                real_res = special_match.group(1).strip()
        return real_res.strip()


class QuestAnswer_OCR(QuestAnswer):
    def model_generation(self, obj: dict):
        template = self._read_prompt_template('QA_prompt.txt')
        query = [
            {
                "role": "system",
                "content": template,
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": "Question: {question}\n\nRetrieved Documents: {search_documents}".format(
                                question=f'{obj["questions"]}',
                                search_documents=f'{obj["retrieve_context"]}'
                            )
                    }
                ]
            },
        ]

        res = self.model.safe_request(query)
        real_res = res.split('<response>')[-1].split('</response>')[0]
        if real_res.strip() == "":
            special_match = re.search(r'</?response>(.*?)</?response>', res, re.DOTALL)
            if special_match:
                real_res = special_match.group(1).strip()
        return real_res.strip()


class QuestAnswer_image_OCR(QuestAnswer):
    def model_generation(self, obj: dict):
        template = self._read_prompt_template('QA_image_OCR_prompt.txt')
        image_path = "image_data/{doc_name}_page_{page}.jpg".format(
            doc_name=obj["doc_name"],page = obj["evidence_page_no"]
        )

        query = [
            {
                "role": "system",
                "content": template,
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": "Question: {question}\n\nRetrieved Documents: {search_documents}".format(
                                    question=f'{obj["questions"]}',
                                    search_documents=f'{obj["retrieve_context"]}'
                                )
                    },
                    {
                        "type": "image",
                        "image": image_path,
                    }
                ]
            },
        ]
        
        res = self.model.safe_request(query)
        real_res = res.split('<response>')[-1].split('</response>')[0]
        if real_res.strip() == "":
            special_match = re.search(r'</?response>(.*?)</?response>', res, re.DOTALL)
            if special_match:
                real_res = special_match.group(1).strip()
        return real_res.strip()


