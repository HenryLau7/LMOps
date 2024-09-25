from abc import ABC, abstractmethod
from typing import List, Dict, Callable
from liquid import Template
from vllm import LLM, SamplingParams
import os
import utils

class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass

class BinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        response = utils.chatgpt(
            prompt, max_tokens=4, n=1, timeout=2, 
            temperature=self.opt['temperature'])[0]
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred


class Vllm_predictor(ABC):
    
    def __init__(self, model_path=None, max_tokens=9, stop=None, repetition_penalty=1.0, top_p=0.1, temperature=0):
        self.llm = LLM(model=model_path)
        self.max_tokens = max_tokens
        self.stop = stop
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.temperature = temperature

    def inference(self, ex, prompt):
        prompt.replace(':', '\\:')
        prompt = Template(prompt).render(text=ex['text'])

        # utils.log_to_file(os.getenv("LOG_FILE"), f"Prompt:\n{prompt}\n")

        sampling_params = SamplingParams(temperature=self.temperature, repetition_penalty=self.repetition_penalty, top_p=self.top_p, max_tokens=self.max_tokens, stop=self.stop)
        # print(sampling_params)
        utils.log_to_file(os.getenv("LOG_FILE"), f"Sampling Params:\n{sampling_params}\n")
        utils.log_to_file(os.getenv("LOG_FILE"), f"Prompt:\n{prompt}\n")
        
        response = self.llm.generate(prompt, sampling_params)[0].outputs[0].text

        # utils.log_to_file(os.getenv("LOG_FILE"), f"Response:\n{response}\n")

        file_name = os.getenv("GPT_CALL_OUT_FILE_NAME")
        if file_name and not os.path.exists(file_name):
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
        utils.log_to_file(file_name, f"Vllm | Inference\n")

        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred
