from models.gpt4o import GPT4o
from models.gpt4v import GPT4v
from models.ollama import Ollama
from models.ollamaopenai import OllamaOpenAI

class ModelFactory:
    @staticmethod
    def create_model(model_name, *args):
        if model_name == 'gpt-4o':
            return GPT4o(model_name, *args)
        elif model_name == 'gpt-4-vision-preview' or model_name == 'gpt-4-turbo':
            return GPT4v(model_name, *args)
        # elif model_name == "x/llama3.2-vision:latest":
        #     return Ollama(model_name, *args)
        elif model_name == "x/llama3.2-vision:latest":
            return OllamaOpenAI(model_name, *args)
        else:
            raise ValueError(f'Unsupported model type {model_name}. Create entry in app/models/')
