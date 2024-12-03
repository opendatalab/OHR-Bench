import openai
import base64
from loguru import logger

from src.llms.base import BaseLLM
from importlib import import_module

try:
    conf = import_module("src.configs.real_config")
except ImportError:
    conf = import_module("src.configs.config")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class GPT(BaseLLM):
    def __init__(self, model_name='gpt-3.5-turbo', temperature=1.0, max_new_tokens=1024, report=False):
        super().__init__(model_name, temperature, max_new_tokens)

    def request(self, query: str) -> str:
        openai.api_key = conf.GPT_api_key
        if conf.GPT_api_base and conf.GPT_api_base.strip():
            openai.base_url = conf.GPT_api_base
        system_q, user_q = query
        user_content = user_q["content"]
        _user_content = []
        for c in user_content:
            if c["type"] == "image":
                base64_image = encode_image(c["image"])
                _user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                    },
                })
            else:
                _user_content.append(c)
        user_q["content"] = _user_content
        query = [system_q, user_q]
        res = openai.chat.completions.create(
            model = self.params['model_name'],
            messages = query if isinstance(query, list) else [{"role": "user","content": query}] ,
            temperature = self.params['temperature'],
            max_tokens = self.params['max_new_tokens'],
            top_p = self.params['top_p'],
            seed = 0
        )
        real_res = res.choices[0].message.content

        token_consumed = res.usage.total_tokens
        logger.info(f'GPT token consumed: {token_consumed}') if self.report else ()
        return real_res
