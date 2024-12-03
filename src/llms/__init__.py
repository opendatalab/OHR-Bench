from importlib import import_module
try:
    conf = import_module("src.configs.real_config")
except ImportError:
    conf = import_module("src.configs.config")

if conf.GPT_api_key != '':
    from .api_model import GPT

from .local_model import Qwen2_7B_Instruct, LLaMA31_8B_Instruct, Mock, Qwen25_7B_Instruct, Qwen2VL_7B_Instruct, InternVL2_8B_Instruct
