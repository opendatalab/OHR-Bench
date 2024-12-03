import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from importlib import import_module
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import math
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

try:
    conf = import_module("src.configs.real_config")
except ImportError:
    conf = import_module("src.configs.config")
from src.llms.base import BaseLLM

Image.MAX_IMAGE_PIXELS = None

class Qwen2_7B_Instruct(BaseLLM):
    def __init__(self, model_name='qwen2_7b', temperature=1.0, max_new_tokens=1024, **more_params):
        super().__init__(model_name, temperature, max_new_tokens, **more_params)
        local_path = conf.Qwen2_7B_local_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, 
                                                          device_map="auto",
                                                          torch_dtype=torch.bfloat16,
                                                          attn_implementation="flash_attention_2",
                                                          trust_remote_code=True).eval()
        self.gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": self.params['temperature'] > 0,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'] if self.params['temperature'] > 0 else 0,
            "top_k": self.params['top_k'] if self.params['temperature'] > 0 else 0,
        }

    def request(self, query: str) -> str:
        if isinstance(query, list):
            input_ids = self.tokenizer.apply_chat_template(query, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        else:
            query = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
            input_ids = self.tokenizer.encode(query, return_tensors="pt").cuda()
        output = self.model.generate(input_ids, **self.gen_kwargs)[0]
        response = self.tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response


class Qwen25_7B_Instruct(BaseLLM):
    def __init__(self, model_name='qwen25_7b', temperature=1.0, max_new_tokens=1024, **more_params):
        super().__init__(model_name, temperature, max_new_tokens, **more_params)
        local_path = conf.Qwen25_7B_local_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, 
                                                          device_map="auto",
                                                          torch_dtype=torch.bfloat16,
                                                          attn_implementation="flash_attention_2",
                                                          trust_remote_code=True).eval()
        self.gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": self.params['temperature'] > 0,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'] if self.params['temperature'] > 0 else 0,
            "top_k": self.params['top_k'] if self.params['temperature'] > 0 else 0,
        }

    def request(self, query: str) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            query, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        output = self.model.generate(input_ids, **self.gen_kwargs)[0]
        response = self.tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response


class LLaMA31_8B_Instruct(BaseLLM):
    def __init__(self, model_name='llama3.1_8b', temperature=1.0, max_new_tokens=1024, **more_params):
        super().__init__(model_name, temperature, max_new_tokens, **more_params)
        local_path = conf.LLaMA31_8B_Instruct
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, 
                                                          device_map="auto",
                                                          torch_dtype=torch.bfloat16,
                                                          attn_implementation="flash_attention_2",
                                                          trust_remote_code=True).eval()
        self.gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": self.params['temperature'] > 0,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'] if self.params['temperature'] > 0 else 0,
            "top_k": self.params['top_k'] if self.params['temperature'] > 0 else 0,
            "eos_token_id": [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        }

    def request(self, query: str) -> str:
        input_ids = self.tokenizer.apply_chat_template(query, tokenize=True, add_generation_prompt=True, return_tensors="pt").cuda()
        output = self.model.generate(input_ids=input_ids, **self.gen_kwargs)[0]
        response = self.tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response

class Mock(BaseLLM):
    def __init__(self):
        super().__init__()

    def request(self, query: str) -> str:
        return ""

class Qwen2VL_7B_Instruct(BaseLLM):
    def __init__(self, model_name='qwen2vl_7b', temperature=1.0, max_new_tokens=1024, **more_params):
        super().__init__(model_name, temperature, max_new_tokens, **more_params)
        local_path = conf.Qwen2VL_7B_path
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            local_path, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation = "flash_attention_2"
        )
        self.tokenizer = AutoProcessor.from_pretrained(local_path)
        self.gen_kwargs = {
            # "temperature": self.params['temperature'],
            "do_sample": self.params['temperature'] > 0,
            "max_new_tokens": self.params['max_new_tokens'],
            # "top_p": self.params['top_p'] if self.params['temperature'] > 0 else 0,
            # "top_k": self.params['top_k'] if self.params['temperature'] > 0 else 0,
        }

    def request(self, query: str) -> str:
        assert isinstance(query, list)
        messages = query
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

class InternVL2_8B_Instruct(BaseLLM):
    def __init__(self, model_name='internvl2_8b', temperature=1.0, max_new_tokens=1024, **more_params):
        super().__init__(model_name, temperature, max_new_tokens, **more_params)
        local_path = conf.InternVL2_8B_path
        # device_map = split_model('InternVL2-8B')

        self.model = AutoModel.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True, use_fast=False)
        self.gen_kwargs = {
            # "temperature": self.params['temperature'],
            "do_sample": self.params['temperature'] > 0,
            "max_new_tokens": self.params['max_new_tokens'],
            # "top_p": self.params['top_p'] if self.params['temperature'] > 0 else 0,
            # "top_k": self.params['top_k'] if self.params['temperature'] > 0 else 0,
        }

    def request(self, query: str) -> str:
        assert isinstance(query, list)
        system_q, user_q = query
        user_content = user_q["content"]
        pixel_values, img_path, user_text = None, None, None
        for c in user_content:
            if c["type"] == "image":
                img_path = c["image"]
            else:
                user_text = c["text"]
        input_seqs = [system_q["content"], user_text]
        if img_path is not None:
            pixel_values = load_image(img_path).to(torch.bfloat16).cuda()
            input_seqs.insert(0, 'Image: <image>')

        question = '\n\n'.join(input_seqs)
        response = self.model.chat(self.tokenizer, pixel_values, question, self.gen_kwargs)
        return response


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map
