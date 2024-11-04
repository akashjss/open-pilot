# llama3_2_vision.py
import ollama
import json
from models.model import Model
from utils.screens import Screens
from typing import Any
from PIL import Image


class Ollama(Model):
    def __init__(self, model_name, base_url, api_key, context):
        super().__init__(model_name, base_url, api_key, context)
        # Update to correct initialization using ollama
        self.model_name = "x/llama3.2-vision:latest"
        self.screens = Screens()

    def format_user_request_for_llm(self, original_user_request, step_num) -> list[dict[str, Any]]:
        # Use screenshot as an image file directly instead of converting to base64
        screenshot_path = self.screens.save_screenshot_to_file()

        # Update the request prompt to be more specific and clear
        request_data: str = json.dumps({
            'original_user_request': original_user_request,
            'step_num': step_num,
            'screenshot': screenshot_path
        })

        # Create message with prompt and screenshot
        message = [
            {'role': 'user', 'content': request_data},
            {'type': 'image_path', 'image_path': screenshot_path}
        ]
        print(message)
        return message

    def send_message_to_llm(self, message) -> Any:
        # Using ollama to generate the response with image as input
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            "You are assisting with personal, authorized commands on my own machine. "
                            "User requests will be to control my computer locally for routine tasks (e.g., 'Open Sublime Text' or 'Create an Excel sheet with a meal plan'). "
                            "You will respond with JSON steps that help me accomplish these commands, mapping to specific function calls that control the mouse and keyboard using pyautogui. "
                            "This will be used only on my personal device for development and productivity purposes. "
                            "The JSON should be formatted exactly as specified, without extra text. "
                            "Only send back a valid JSON response that can be parsed without errors. "
                            "Here is the expected format:\n\n"
                            "{\n"
                            "    \"steps\": [\n"
                            "        {\n"
                            "            \"function\": \"...\",\n"
                            "            \"parameters\": { ... },\n"
                            "            \"human_readable_justification\": \"...\"\n"
                            "        },\n"
                            "        ...\n"
                            "    ],\n"
                            "    \"done\": ...\n"
                            "}\n\n"
                            "Valid function names and their expected parameters are:\n"
                            "- \"press\": parameters: { \"keys\": [\"key1\", \"key2\", ...], \"presses\": int, \"interval\": float }\n"
                            "- \"write\": parameters: { \"text\": \"string\", \"interval\": float }\n"
                            "- \"hotkey\": parameters: { \"keys\": [\"key1\", \"key2\", ...] }\n"
                            "- \"moveTo\": parameters: { \"x\": int, \"y\": int, \"duration\": float }\n"
                            "- \"click\": parameters: { \"x\": int, \"y\": int, \"button\": \"left\" or \"right\", \"clicks\": int, \"interval\": float }\n"
                            "- \"sleep\": parameters: { \"secs\": float }\n\n"
                            "Valid key names for 'keys' parameters are:\n"
                            "- 'shift', 'ctrl', 'alt', 'command', 'tab', 'space', 'enter', 'left', 'right', 'up', 'down', etc.\n"
                            "Note: Use 'command' instead of 'cmd' for the Command key on macOS.\n"
                            "When responding, ensure you include all necessary steps to complete the user's request fully.\n"
                            "For example, to 'open Google', you might need to:\n"
                            "- Open the browser application.\n"
                            "- Navigate to 'www.google.com' in the browser.\n"
                            "Remember, only output the JSON response in this format without any additional text. Do not use 'pyautogui' as a function name. Use the specific function names listed above."
                        )
                    },
                    {
                        'role': 'user',
                        'content': message[0]['content'],
                        'images': [message[1]['image_path']]
                    }
                ]
            )
            # Log the raw response for debugging purposes
            print(f"Raw LLM Response: {response}")
            return response
        except Exception as e:
            print(f"Error during LLM call: {e}")
            return {}

    def convert_llm_response_to_json_instructions(self, llm_response: Any) -> dict[str, Any]:
        # Assuming the local model generates JSON instructions
        if not llm_response or 'message' not in llm_response or 'content' not in llm_response['message']:
            print("Error: LLM response is empty or missing expected content.")
            return {}

        llm_response_data: str = llm_response['message']['content']

        # Log the raw response for debugging purposes
        print(f"LLM Response: {llm_response_data}")

        # Our current LLM model does not guarantee a JSON response hence we manually parse the JSON part of the response
        start_index = llm_response_data.find('{')
        end_index = llm_response_data.rfind('}')

        if start_index == -1 or end_index == -1:
            print("Error: No JSON object found in the LLM response. Falling back to default message.")
            return {'error': 'No JSON object found', 'message': llm_response_data}

        try:
            json_response = json.loads(llm_response_data[start_index:end_index + 1].strip())
        except Exception as e:
            print(f"Error while parsing JSON response: {e}")
            json_response = {'error': 'JSON parsing failed', 'message': llm_response_data}

        return json_response

    def get_instructions_for_objective(self, original_user_request: str, step_num: int = 0) -> dict[str, Any]:
        # This method generates instructions for a given objective
        message: list[dict[str, Any]] = self.format_user_request_for_llm(original_user_request, step_num)
        llm_response = self.send_message_to_llm(message)
        json_instructions: dict[str, Any] = self.convert_llm_response_to_json_instructions(llm_response)
        return json_instructions

    def cleanup(self):
        # Cleanup resources if necessary, such as deleting temporary files
        self.screens.delete_temp_screenshot_files()
        print("Cleanup completed.")