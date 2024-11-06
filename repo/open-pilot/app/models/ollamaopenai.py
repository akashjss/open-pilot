import json
from typing import Any

from models.model import Model
from utils.screen import Screen


class OllamaOpenAI(Model):
    def __init__(self, model_name='', base_url='http://localhost:11434/v1/', api_key='ollama', context=''):
        # Use the base class constructor to initialize the client and other attributes
        super().__init__(model_name, base_url, api_key, context)
        print(f"OllamaOpenAI model initialized with model_name: {model_name}")

    def get_instructions_for_objective(self, original_user_request: str, step_num: int = 0) -> dict[str, Any]:
        message: list[dict[str, Any]] = self.format_user_request_for_llm(original_user_request, step_num)
        print(f"Sending user request to model (without image): {json.dumps(message[:1], indent=2)}")  # Debug statement to show the request without image
        llm_response = self.send_message_to_llm(message)
        print(f"Received response from model: {llm_response}")  # Debug statement to show the response
        json_instructions: dict[str, Any] = self.convert_llm_response_to_json_instructions(llm_response)
        return json_instructions

    def format_user_request_for_llm(self, original_user_request, step_num) -> list[dict[str, Any]]:
        base64_img: str = Screen().get_screenshot_in_base64()

        request_data: str = json.dumps({
            'original_user_request': original_user_request,
            'step_num': step_num
        })

        # We have to add context every request for now which is expensive because our chosen model doesn't have a
        # stateful/Assistant mode yet.
        message = [
            {'type': 'text', 'text': self.context + request_data},
            {'type': 'image_url',
             'image_url': {
                 'url': f'data:image/jpeg;base64,{base64_img}'
             }
             }
        ]

        return message

    def send_message_to_llm(self, message) -> Any:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    'role': 'user',
                    'content': message,
                }
            ],
            max_tokens=10000,
        )
        return response

    def convert_llm_response_to_json_instructions(self, llm_response: Any) -> dict[str, Any]:
        llm_response_data: str = llm_response.choices[0].message.content.strip()

        # Our current LLM model does not guarantee a JSON response hence we manually parse the JSON part of the response
        # Check for updates here - https://github.com/ollama/ollama/blob/main/docs/api.md
        start_index = llm_response_data.find('{')
        end_index = llm_response_data.rfind('}')

        try:
            json_response = json.loads(llm_response_data[start_index:end_index + 1].strip())
        except Exception as e:
            print(f'Error while parsing JSON response - {e}')
            json_response = {}

        return json_response
