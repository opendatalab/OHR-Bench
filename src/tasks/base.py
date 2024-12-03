import os
from abc import ABC

class BaseTask(ABC):
    def __init__(
            self,
            output_dir: str = './output',
            quest_eval_model: str = "gpt-3.5-turbo"
        ):
        
        if not (os.path.exists(output_dir) and os.path.isdir(output_dir)):
            os.makedirs(output_dir)
        
    
    def set_model(self, model, retriever) -> None:
        
        return 
    
    def retrieve_docs(self, obj:dict) -> str:

        return " "

    def model_generation(self, obj:dict) -> None:
        # use LLM to generate text
        
        return     
        
    def _read_prompt_template(self, filename: str):
        # read template to generate prompt
        
        return

    def scoring(self, data_point: dict) -> dict:
        return {
            'metrics': {
                # Numerical results to be recorded by subclasses, mandatory.
                # Such as accuracy, recall, bleu, rouge, etc.
            },
            'log': {
                # String results to be recorded by subclasses, optional.
                # Such as model output.
            },
            'valid': ...
                # Boolean result to be recorded by subclasses, indicating whether the evaluation is valid, mandatory.
                # True or False.
        }

    def compute_overall(self, results: list[dict]) -> dict:
        return {
                # 'Metric1': Value,
                # 'Metric2': Value,
                # ...
        }

    