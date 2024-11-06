from models.ollamaopenai import OllamaOpenAI

class ModelFactory:
    @staticmethod
    def create_model(model_name, *args):
        return OllamaOpenAI(model_name, *args)

