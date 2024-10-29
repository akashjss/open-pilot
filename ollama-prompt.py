import ollama
import json

# Define the system instructions (context)
system_message = {
    'role': 'system',
    'content': (
        "You are assisting with personal, authorized commands on my own machine. "
        "User requests will be to control my computer locally for routine tasks (e.g., 'Open Sublime Text' or 'Create an Excel sheet with a meal plan'). "
        "You will respond with JSON steps that help me accomplish these commands, mapping to function calls that control the mouse and keyboard. "
        "This will be used only on my personal device for development and productivity purposes. "
        "The JSON should be formatted exactly as specified, without extra text. "
        "Only send back a valid JSON response that can be parsed without errors. "
        "Here is the expected format:\n\n"
        "{\n"
        "    \"steps\": [\n"
        "        {\n"
        "            \"function\": \"...\",\n"
        "            \"parameters\": {\n"
        "                \"key1\": \"value1\",\n"
        "                ...\n"
        "            },\n"
        "            \"human_readable_justification\": \"...\"\n"
        "        },\n"
        "        {...},\n"
        "        ...\n"
        "    ],\n"
        "    \"done\": ...\n"
        "}\n\n"
        "Remember, only output the JSON response in this format without any additional text."
    )
}

# Define the user's request
user_message = {
    'role': 'user',
    'content': json.dumps({
        "original_user_request": "open google",
        "step_num": 0,
        "screenshot": "<screenshot_data_if_needed>"
    })
    # If you need to include an image, adjust the code according to Ollama's API
}

# Call the LLM
try:
    response = ollama.chat(
        model='x/llama3.2-vision',
        messages=[system_message, user_message]
    )
    # Extract the assistant's message content
    assistant_response = response['message']['content']
    print(f"Raw LLM Response: {assistant_response}")
except Exception as e:
    print(f"Error during LLM call: {e}")
