"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import time
import os
import requests
import config
import string
from openai import AzureOpenAI

def log_to_file(filename, content, mode='a'):
    """
    :param filename: str
    :param content: str
    :param mode: str
    """
    with open(filename, mode) as outf:
        outf.write(content)
        
def parse_sectioned_prompt(s):

    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
            current_header = current_header.translate(str.maketrans('', '', string.punctuation))
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'

    return result

class AzureGPT4():
    def __init__(self):
        endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', '')
        assert endpoint, "Please set the environment variable AZURE_OPENAI_ENDPOINT"
        api_key = os.environ.get('AZURE_OPENAI_API_KEY', '')
        assert api_key, "Please set the environment variable AZURE_OPENAI_API_KEY"

        self.client = AzureOpenAI(
            azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', ''),
            api_key = os.environ.get('AZURE_OPENAI_API_KEY', ''),
            api_version = "2024-05-01-preview",
        )
    
    def __call__(self, prompt, n = 1,temperature=0.7, max_tokens=1024,top_p=1,stop=None, desc=''):
        # Record Call
        file_name = os.getenv("GPT_CALL_OUT_FILE_NAME")
        if file_name and not os.path.exists(file_name):
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
        log_to_file(file_name, f"GPT4 | {desc} | temperature {temperature}\n")

        messages = [
           {'role': 'user', 'content': prompt}
        ]  
        
        response, timeout = "", 5
        while not response:
            try:
                time.sleep(timeout)
                completion = self.client.chat.completions.create(
                            model="gpt-4",
                            messages=messages,
                            seed=42,
                            temperature = temperature,
                            max_tokens = max_tokens,
                            top_p=top_p,
                            stop=stop,
                            n = n
                )
                response = [choice.message.content for choice in completion.choices]
            except Exception as e:
                pass

            if not response:
                timeout = timeout * 2
                if timeout > 120:
                    timeout=1
                if timeout > 1024:
                    break
                try:
                    print(f"Will retry after {timeout} seconds ...")
                except:
                    pass

        return response

def chatgpt(prompt, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=1024, 
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=10):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias
    }
    retries = 0
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/chat/completions',
                headers = {
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=timeout
            )
            if r.status_code != 200:
                retries += 1
                time.sleep(1)
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(1)
            retries += 1
    r = r.json()
    return [choice['message']['content'] for choice in r['choices']]


def instructGPT_logprobs(prompt, temperature=0.7):
    payload = {
        "prompt": prompt,
        "model": "text-davinci-003",
        "temperature": temperature,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True
    }
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/completions',
                headers = {
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=10
            )  
            if r.status_code != 200:
                time.sleep(2)
                retries += 1
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(5)
    r = r.json()
    return r['choices']


