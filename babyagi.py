#!/usr/bin/env python3
from dotenv import load_dotenv

"""
gpt-4o-2024-05-13 Issue: : I attempted to extend the context and enable keyboard control for the snake game, but I was unable to resolve the following error.
No matter how many times I execute it, it results in a build error. It is likely due to version differences, but it did not automatically investigate and resolve this issue.

Error
```
# Command executed most recently
flutter run -d web-server --web-port 8080 --web-hostname 0.0.0.0

# Result of last command executed
The Return Code for the command is 1:
Launching lib/main.dart on Web Server in debug mode...
Waiting for connection from debug service on Web Server...             ⣷⣯⣽⣻⣟⡿⢿⣻⣟⣯⣽⣾⣷⣯⣽⣻
[K[31mlib/main.dart:153:16: Error: The argument type 'void Function(RawKeyEvent)' can't be assigned to the parameter type 'KeyEventResult Function(FocusNode, RawKeyEvent)?'.[39m
Waiting for connection from debug service on Web Server...             ⣟
[K[31m - 'RawKeyEvent' is from 'package:flutter/src/services/raw_keyboard.dart' ('../flutter/packages/flutter/lib/src/services/raw_keyboard.dart').[39m
Waiting for connection from debug service on Web Server...             ⡿
[K[31m - 'KeyEventResult' is from 'package:flutter/src/widgets/focus_manager.dart' ('../flutter/packages/flutter/lib/src/widgets/focus_manager.dart').[39m
Waiting for connection from debug service on Web Server...             ⢿
[K[31m - 'FocusNode' is from 'package:flutter/src/widgets/focus_manager.dart' ('../flutter/packages/flutter/lib/src/widgets/focus_manager.dart').[39m
Waiting for connection from debug service on Web Server...             ⣻
[K[31m        onKey: _onKey,[39m
Waiting for connection from debug service on Web Server...             ⣟
[K[31m               ^[39m
Waiting for connection from debug service on Web Server...             ⣯⣽⣾⣷⣯⣽⣻⣟⡿⢿⣻⣟⣯⣽⣾⣷⣯⣽⣻⣟⡿⢿⣻⣟⣯⣽⣾⣷⣯⣽⣻⣟⡿⢿⣻⣟⣯⣽⣾⣷⣯⣽⣻⣟⡿⢿⣻⣟⣯⣽⣾⣷⣯⣽⣻⣟⡿⢿⣻⣟⣯⣽⣾⣷⣯⣽⣻⣟⡿⢿⣻⣟⣯⣽⣾⣷⣯⣽⣻⣟⡿⢿     9.8s
[31mFailed to compile application.[39m
```
Source
```
  void _onKey(RawKeyEvent event) {
    if (event is RawKeyDownEvent) {
      switch (event.logicalKey.keyLabel) {
        case 'Arrow Up':
          if (direction != 'down') direction = 'up';
          break;
        case 'Arrow Down':
          if (direction != 'up') direction = 'down';
          break;
        case 'Arrow Left':
          if (direction != 'right') direction = 'left';
          break;
        case 'Arrow Right':
          if (direction != 'left') direction = 'right';
          break;
      }
    }
  }
```
"""

# Load default environment variables (.env)
load_dotenv()

import os
import hashlib
import pickle
import subprocess
import select
import pty
import time
import logging
from collections import deque
from typing import Dict, List
import importlib
import anthropic
from anthropic import Anthropic
import openai
from openai import OpenAI
import google.generativeai as genai
import tiktoken as tiktoken
import re
from task_parser import TaskParser
from executed_task_parser import ExecutedTaskParser
import sys
import threading
import base64

#[Test]
#TaskParser().test()
#while True:
#    time.sleep(100)

# Engine configuration
BABY_COMMAND_AGI_FOLDER = "/app"
WORKSPACE_FOLDER = "/workspace"

# Model: GPT, LLAMA, HUMAN, etc.
LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20240620").lower()
LLM_VISION_MODEL = os.getenv("LLM_VISION_MODEL", "claude-3-5-sonnet-20240620").lower()
TOKEN_COUNT_MODEL = os.getenv("TOKEN_COUNT_MODEL", "claude-3-5-sonnet-20240620").lower()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY= os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_AI_STUDIO_API_KEY =  os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


if not (LLM_MODEL.startswith("llama") or LLM_MODEL.startswith("human")):
    assert ANTHROPIC_API_KEY, "\033[91m\033[1m" + "ANTHROPIC_API_KEY environment variable is missing from .env" + "\033[0m\033[0m"

# Table config
RESULTS_STORE_NAME = os.getenv("RESULTS_STORE_NAME", "no-name-table") + "-" + os.getenv("RESULTS_SOTRE_NUMBER", "0")
assert RESULTS_STORE_NAME, "\033[91m\033[1m" + "RESULTS_STORE_NAME environment variable is missing from .env" + "\033[0m\033[0m"

# Run configuration
INSTANCE_NAME = os.getenv("INSTANCE_NAME", "BabyCommandAGI")
COOPERATIVE_MODE = "none"

# If LLM_COMMAND_RESPONSE is set to True, the LLM will automatically respond if there is a confirmation when executing a command, 
# but be aware that this will increase the number of times the LLM is used and increase the cost of the API, etc.
LLM_COMMAND_RESPONSE = True
JOIN_EXISTING_OBJECTIVE = False

MAX_MARGIN_TOKEN = 200 # Allow enough tokens to avoid being on the edge
MAX_MODEL_OUTPUT_TOKEN = 4 * 1024 # default value
MAX_MODEL_INPUT_TOKEN = 128 * 1024 # default value
# Maximum number of tokens is confirmed below
# https://context.ai/compare/gpt-4o/claude-3-5-sonnet
MAX_CLAUDE_3_5_SONNET_OUTPUT_TOKEN = 8 * 1024
MAX_CLAUDE_3_5_SONNET_INPUT_TOKEN = 200 * 1024
# Maximum number of tokens is confirmed below
# https://context.ai/compare/o1-preview/gpt-4o
MAX_O1_PREVIEW_OUTPUT_TOKEN = 32 * 1024
MAX_O1_PREVIEW_INPUT_TOKEN = 128 * 1024
# Maximum number of tokens is confirmed below
# https://platform.openai.com/docs/models/gpt-4o
MAX_CHATGPT_4O_LATEST_OUTPUT_TOKEN = 16 * 1024
MAX_CHATGPT_4O_LATEST_INPUT_TOKEN = 128 * 1024


MAX_COMMAND_RESULT_TOKEN = 8 * 1024
MAX_DUPLICATE_COMMAND_RESULT_TOKEN = 1 * 1024

# Goal configuration
ORIGINAL_OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", "")

# Set Variables
hash_object = hashlib.sha1(ORIGINAL_OBJECTIVE.encode())
hex_dig = hash_object.hexdigest()
objective_table_name = f"{hex_dig[:8]}-{RESULTS_STORE_NAME}"
OBJECTIVE_LIST_FILE = f"{BABY_COMMAND_AGI_FOLDER}/data/{objective_table_name}_objectvie_list.pkl"
TASK_LIST_FILE = f"{BABY_COMMAND_AGI_FOLDER}/data/{objective_table_name}_task_list.pkl"
EXECUTED_TASK_LIST_FILE = f"{BABY_COMMAND_AGI_FOLDER}/data/{RESULTS_STORE_NAME}_executed_task_list.pkl"
PWD_FILE = f"{BABY_COMMAND_AGI_FOLDER}/pwd/{RESULTS_STORE_NAME}"
ENV_DUMP_FILE = f"{BABY_COMMAND_AGI_FOLDER}/env_dump/{RESULTS_STORE_NAME}"

# logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=f"{BABY_COMMAND_AGI_FOLDER}/log/{objective_table_name}.log",
                    filemode='a',
                    level=logging.DEBUG)
def log(message):
    print(message)
    logging.info(message)

# Save and load functions for task_list and executed_task_list
def save_data(data, filename):
  with open(filename, 'wb') as f:
    pickle.dump(data, f)

def load_data(filename):
  if os.path.exists(filename):
    with open(filename, 'rb') as f:
      return pickle.load(f)
  return deque([])

def parse_objective(objective_list: deque) -> str:
    if len(objective_list) == 1:
        return objective_list[0]
    objective = ""
    for idx, objective_item in enumerate(objective_list):
        objective += f"""[Objective {idx + 1}]{objective_item} """
    return objective

objective_list = load_data(OBJECTIVE_LIST_FILE) #deque([])
if len(objective_list) == 0:
    objective_list = deque([ORIGINAL_OBJECTIVE])

OBJECTIVE = parse_objective(objective_list)

# Extensions support begin

def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

DOTENV_EXTENSIONS = os.getenv("DOTENV_EXTENSIONS", "").split(" ")

# Command line arguments extension
# Can override any of the above environment variables
ENABLE_COMMAND_LINE_ARGS = (
        os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
)
if ENABLE_COMMAND_LINE_ARGS:
    if can_import("extensions.argparseext"):
        from extensions.argparseext import parse_arguments

        OBJECTIVE, INITIAL_TASK, LLM_MODEL, DOTENV_EXTENSIONS, INSTANCE_NAME, COOPERATIVE_MODE, JOIN_EXISTING_OBJECTIVE = parse_arguments()

# Human mode extension
# Gives human input to babyagi
if LLM_MODEL.startswith("human"):
    if can_import("extensions.human_mode"):
        from extensions.human_mode import user_input_await

# Load additional environment variables for enabled extensions
# TODO: This might override the following command line arguments as well:
#    OBJECTIVE, INITIAL_TASK, LLM_MODEL, INSTANCE_NAME, COOPERATIVE_MODE, JOIN_EXISTING_OBJECTIVE
if DOTENV_EXTENSIONS:
    if can_import("extensions.dotenvext"):
        from extensions.dotenvext import load_dotenv_extensions

        load_dotenv_extensions(DOTENV_EXTENSIONS)

# TODO: There's still work to be done here to enable people to get
# defaults from dotenv extensions, but also provide command line
# arguments to override them

# Extensions support end

log("\033[95m\033[1m" + "\n*****CONFIGURATION*****\n" + "\033[0m\033[0m")
log(f"Name              : {INSTANCE_NAME}")
log(f"Mode              : {'alone' if COOPERATIVE_MODE in ['n', 'none'] else 'local' if COOPERATIVE_MODE in ['l', 'local'] else 'distributed' if COOPERATIVE_MODE in ['d', 'distributed'] else 'undefined'}")
log(f"LLM_MODEL         : {LLM_MODEL}")
log(f"LLM_VISION_MODEL  : {LLM_VISION_MODEL}")
log(f"TOKEN_COUNT_MODEL : {TOKEN_COUNT_MODEL}")

# Check if we know what we are doing
assert OBJECTIVE, "\033[91m\033[1m" + "OBJECTIVE environment variable is missing from .env" + "\033[0m\033[0m"
assert INITIAL_TASK, "\033[91m\033[1m" + "INITIAL_TASK environment variable is missing from .env" + "\033[0m\033[0m"


LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/llama-13B/ggml-model.bin")
if LLM_MODEL.startswith("llama"):
    if can_import("llama_cpp"):
        from llama_cpp import Llama

        log(f"LLAMA : {LLAMA_MODEL_PATH}" + "\n")
        assert os.path.exists(LLAMA_MODEL_PATH), "\033[91m\033[1m" + f"Model can't be found." + "\033[0m\033[0m"

        CTX_MAX = 1024
        LLAMA_THREADS_NUM = int(os.getenv("LLAMA_THREADS_NUM", 8))
        TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

        log('Initialize model for evaluation')
        llm = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=CTX_MAX,
            n_threads=LLAMA_THREADS_NUM,
            n_batch=512,
            use_mlock=False,
        )

        log(
            "\033[91m\033[1m"
            + "\n*****USING LLAMA.CPP. POTENTIALLY SLOW.*****"
            + "\033[0m\033[0m"
        )
    else:
        log(
            "\033[91m\033[1m"
            + "\nLlama LLM requires package llama-cpp. Falling back to GPT-3.5-turbo."
            + "\033[0m\033[0m"
        )
        LLM_MODEL = "gpt-4o"
        TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

elif LLM_MODEL.startswith("gpt-4"):
    log(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )
    TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

elif LLM_MODEL.startswith("openai/o1-preview"):
    MAX_MODEL_OUTPUT_TOKEN = MAX_O1_PREVIEW_OUTPUT_TOKEN
    MAX_MODEL_INPUT_TOKEN = MAX_O1_PREVIEW_INPUT_TOKEN
    TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

    log(
        "\033[91m\033[1m"
        + "\n*****USING openrouter openai/o1-preview. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

elif LLM_MODEL.startswith("chatgpt-4o-latest"):
    MAX_MODEL_OUTPUT_TOKEN = MAX_CHATGPT_4O_LATEST_OUTPUT_TOKEN
    MAX_MODEL_INPUT_TOKEN = MAX_CHATGPT_4O_LATEST_INPUT_TOKEN
    TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

    log(
        "\033[91m\033[1m"
        + "\n*****USING chatgpt-4o-latest. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

elif LLM_MODEL.startswith("claude-3-5-sonnet"):
    MAX_MODEL_OUTPUT_TOKEN = MAX_CLAUDE_3_5_SONNET_OUTPUT_TOKEN
    MAX_MODEL_INPUT_TOKEN = MAX_CLAUDE_3_5_SONNET_INPUT_TOKEN
    TEMPERATURE = float(os.getenv("ANTHROPIC_TEMPERATURE", "0.0"))

    log(
        "\033[91m\033[1m"
        + "\n*****USING Claude 3.5. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

elif LLM_MODEL.startswith("gemini"):
    TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))

    log(
        "\033[91m\033[1m"
        + "\n*****USING Gemini. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

elif LLM_MODEL.startswith("human"):
    log(
        "\033[91m\033[1m"
        + "\n*****USING HUMAN INPUT*****"
        + "\033[0m\033[0m"
    )
    TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

# Max token initialization
MAX_OUTPUT_TOKEN = MAX_MODEL_OUTPUT_TOKEN
MAX_INPUT_TOKEN = MAX_MODEL_INPUT_TOKEN - MAX_OUTPUT_TOKEN - MAX_MARGIN_TOKEN # default value

log(
    "\033[91m\033[1m"
    + "\n*****POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
    + "\033[0m\033[0m"
)

log("\033[94m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
log(f"{OBJECTIVE}")

# Configure client
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_AI_STUDIO_API_KEY)

# Task storage supporting only a single instance of BabyAGI
class SingleTaskListStorage:
    def __init__(self, task_list: deque):
        self.tasks = task_list

    def append(self, task: Dict):
        self.tasks.append(task)

    def appendleft(self, task: Dict):
        self.tasks.appendleft(task)

    def replace(self, task_list: deque):
        self.tasks = task_list

    def reference(self, index: int):
        return self.tasks[index]

    def pop(self):
        return self.tasks.pop()

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def get_tasks(self):
        return self.tasks

    def remove_target_write_dicts(self, path):
        """
        Remove dictionaries from the list where "target" key matches path and "type" key is "entire_file_after_writing".

        Args:
        - path (str): The target path to match against.
    
        """
        self.tasks = deque([d for d in self.tasks if not (d.get("target") == path and d.get("type") == "entire_file_after_writing")])
        
    def remove_target_command_dicts(self, path, command, result):
        """
        Remove dictionaries from the list where "target" key matches path and "type" key is "command".

        Args:
        - path (str): The target path to match against.
    
        """
        self.tasks = deque([d for d in self.tasks if not (d.get("target") == command and d.get("type") == "command" and "path" in d and d.get("path") == path and d.get("content") == result and self.is_big_command_result(result))])

    def is_big_command_result(self, string) -> bool:

        if TOKEN_COUNT_MODEL.lower().startswith("claude-3"):
            # Claude 3 does not support a way to find out the number of tokens in advance
            # https://github.com/anthropics/anthropic-sdk-python/issues/375#issuecomment-1999982035
            # The count to get OpenAI's tokenizer is often low, as it seems to be about +-15% of OpenAI's tokenizers, so I estimate and calculate at -20% based on gpt-4o.  
            # https://www.reddit.com/r/ClaudeAI/comments/1bgg5v0/comment/kv9fais/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
            # https://www.reddit.com/r/ClaudeAI/comments/1bgg5v0/comment/l0phtj4/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
            try:
                encoding = tiktoken.encoding_for_model('gpt-4o')
            except:
                encoding = tiktoken.encoding_for_model('gpt-4o')  # Fallback for others.
            encoded = encoding.encode(string)

            length = int(len(encoded) * (1.0 - 0.2))

            return MAX_DUPLICATE_COMMAND_RESULT_TOKEN <= length
        else:
            try:
                encoding = tiktoken.encoding_for_model(TOKEN_COUNT_MODEL)
            except:
                encoding = tiktoken.encoding_for_model('gpt-4o')  # Fallback for others.
            encoded = encoding.encode(string)
            return MAX_DUPLICATE_COMMAND_RESULT_TOKEN <= len(encoded)

# Task list
temp_task_list = load_data(TASK_LIST_FILE)
temp_executed_task_list = load_data(EXECUTED_TASK_LIST_FILE)

# Initialize tasks storage
tasks_storage = SingleTaskListStorage(temp_task_list)
executed_tasks_storage = SingleTaskListStorage(temp_executed_task_list)
if COOPERATIVE_MODE in ['l', 'local']:
    if can_import("extensions.ray_tasks"):
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).resolve().parent))
        from extensions.ray_tasks import CooperativeTaskListStorage

        tasks_storage = CooperativeTaskListStorage(OBJECTIVE, temp_task_list)
        log("\nReplacing tasks storage: " + "\033[93m\033[1m" + "Ray" + "\033[0m\033[0m")
        executed_tasks_storage = CooperativeTaskListStorage(OBJECTIVE, temp_executed_task_list)
        log("\nReplacing executed tasks storage: " + "\033[93m\033[1m" + "Ray" + "\033[0m\033[0m")
elif COOPERATIVE_MODE in ['d', 'distributed']:
    pass


if tasks_storage.is_empty() or JOIN_EXISTING_OBJECTIVE:
    log("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {INITIAL_TASK}")
else:
    log("\033[93m\033[1m" + f"\nContinue task" + "\033[0m\033[0m")

log("\n")

def limit_tokens_from_string(string: str, model: str, limit: int) -> str:
    """Limits the string to a number of tokens (estimated)."""

    if model.lower().startswith("claude-3"):
        # Claude 3 does not support a way to find out the number of tokens in advance
        # https://github.com/anthropics/anthropic-sdk-python/issues/375#issuecomment-1999982035
        # The count to get OpenAI's tokenizer is often low, as it seems to be about +-15% of OpenAI's tokenizers, so I estimate and calculate at -20% based on gpt-4o.  
        # https://www.reddit.com/r/ClaudeAI/comments/1bgg5v0/comment/kv9fais/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
        # https://www.reddit.com/r/ClaudeAI/comments/1bgg5v0/comment/l0phtj4/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
        try:
            encoding = tiktoken.encoding_for_model('gpt-4o')
        except:
            encoding = tiktoken.encoding_for_model('gpt-4o')  # Fallback for others.
        limit = int(limit * (1.0 - 0.2))
    else:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except:
            encoding = tiktoken.encoding_for_model('gpt-4o')  # Fallback for others.

    encoded = encoding.encode(string)

    return encoding.decode(encoded[:limit])

def last_tokens_from_string(string: str, model: str, last: int) -> str:
    """Limits the string to a number of tokens (estimated)."""

    if model.lower().startswith("claude-3"):
        # Claude 3 does not support a way to find out the number of tokens in advance
        # https://github.com/anthropics/anthropic-sdk-python/issues/375#issuecomment-1999982035
        # The count to get OpenAI's tokenizer is often low, as it seems to be about +-15% of OpenAI's tokenizers, so I estimate and calculate at -20% based on gpt-4o.  
        # https://www.reddit.com/r/ClaudeAI/comments/1bgg5v0/comment/kv9fais/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
        # https://www.reddit.com/r/ClaudeAI/comments/1bgg5v0/comment/l0phtj4/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
        try:
            encoding = tiktoken.encoding_for_model('gpt-4o')
        except:
            encoding = tiktoken.encoding_for_model('gpt-4o')  # Fallback for others.
        last = int(last * (1.0 - 0.2))
    else:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except:
            encoding = tiktoken.encoding_for_model('gpt-4o')  # Fallback for others.

    encoded = encoding.encode(string)

    return encoding.decode(encoded[-last:])


def separate_markdown(markdown):
    """
    Separates a markdown string into images and text parts.

    Images are identified and categorized as local files or URLs.
    Local files are represented with a dictionary containing path and description.
    URLs are represented with a dictionary containing URL and description.
    Text parts are returned as strings.

    Args:
    markdown (str): The Markdown string to be processed.

    Returns:
    list: A list containing separated parts of the Markdown content.
    """
    # Regex patterns for identifying images and splitting content
    image_pattern = r'!\[(.*?)\]\((.*?)\)'
    split_pattern = r'(!\[.*?\]\(.*?\))'

    # Split the markdown content
    parts = re.split(split_pattern, markdown)

    # Process each part to categorize as image or text
    result = []
    for part in parts:
        # Check if the part is an image
        match = re.match(image_pattern, part)
        if match:
            description, path_or_url = match.groups()
            # Check if it's a local file or a URL
            if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
                result.append({'url': path_or_url, 'description': description})
            else:
                result.append({'path': path_or_url, 'description': description})
        else:
            # If not an image, it's a text part
            if part.strip():
                result.append(part)

    return result

def encode_image(image_path):
    """ Encodes a local image file to base64 """
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def modify_parts_to_new_format_anthropic(parts):
    """
    Modifies the given array of Markdown parts to a new specified format.

    Args:
    parts (list): A list containing separated parts of Markdown content including images and text.

    Returns:
    list: A list of parts in the new specified format.
    """
    new_format_parts = []

    for part in parts:
        if isinstance(part, str):  # Text part
            new_format_parts.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            if 'url' in part: # Image with URL
                raise Exception("Image with URL is not supported in Anthropic API")
            elif 'path' in part:  # Local image file
                # Determine media type based on file extension
                extension = part['path'].split('.')[-1].lower()
                if extension in ['jpg', 'jpeg']:
                    media_type = 'image/jpeg'
                elif extension == 'png':
                    media_type = 'image/png'
                elif extension == 'gif':
                    media_type = 'image/gif'
                elif extension == 'webp':
                    media_type = 'image/webp'
                else:
                    media_type = 'application/octet-stream'  # Default media type
                base64_image = encode_image(part['path'])

                new_format_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_image
                    }
                })
    return new_format_parts

def modify_parts_to_new_format_openai(parts):
    """
    Modifies the given array of Markdown parts to a new specified format.

    Args:
    parts (list): A list containing separated parts of Markdown content including images and text.

    Returns:
    list: A list of parts in the new specified format.
    """
    new_format_parts = []

    for part in parts:
        if isinstance(part, str):  # Text part
            new_format_parts.append({"type": "text", "text": part})
        elif isinstance(part, dict):
            if 'url' in part:  # Image with URL
                new_format_parts.append({"type": "image_url", "image_url": {"url": part['url']}})
            elif 'path' in part:  # Local image file
                extension = part['path'].split('.')[-1].lower()
                if extension in ['jpg', 'jpeg']:
                    media_type = 'image/jpeg'
                elif extension == 'png':
                    media_type = 'image/png'
                elif extension == 'gif':
                    media_type = 'image/gif'
                elif extension == 'webp':
                    media_type = 'image/webp'
                else:
                    media_type = 'application/octet-stream'  # Default media type
                base64_image = encode_image(part['path'])
                new_format_parts.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_image}"}})
    return new_format_parts

def llm_call(
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_OUTPUT_TOKEN,
):
    while True:
        try:
            if model.lower().startswith("llama"):
                result = llm(prompt[:CTX_MAX],
                             stop=["### Human"],
                             echo=False,
                             temperature=0.2,
                             top_k=40,
                             top_p=0.95,
                             repeat_penalty=1.05,
                             max_tokens=200)
                # log('\n*****RESULT JSON DUMP*****\n')
                # log(json.dumps(result))
                # log('\n')
                return result['choices'][0]['text'].strip()
            elif model.lower().startswith("human"):
                return user_input_await(prompt)
            # Comment out for Open Router.
            # elif not model.lower().startswith("gpt-"):
            #     # Use completion API
            #     response = openai.Completion.create(
            #         engine=model,
            #         prompt=prompt,
            #         temperature=temperature,
            #         max_tokens=max_tokens,
            #         top_p=1,
            #         frequency_penalty=0,
            #         presence_penalty=0,
            #     )
            #     return response.choices[0].text.strip()
            elif model.lower().startswith("gemini"):
                
                log(f"【MODEL】:{model}")

                system_prompt = prompt

                generation_config = {
                    "temperature" : temperature,
                    "max_output_tokens": 8192,
                    "response_mime_type": "text/plain",
                }

                model = genai.GenerativeModel(
                    model_name=model,
                    generation_config=generation_config,
                    system_instruction=system_prompt,
                )

                chat_session = model.start_chat(
                    history=[
                    ]
                )
                response = chat_session.send_message(prompt)

                return response.text.strip()
            elif model.lower().startswith("claude"):

                anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

                separated_content = separate_markdown(prompt) # for Vision API
                if len(separated_content) > 1:

                    log(f"【MODEL】:{LLM_VISION_MODEL}")

                    messages = [
                        {
                            "role": "user",
                            "content": modify_parts_to_new_format_anthropic(separated_content)
                        }
                    ]

                    # log("【MESSAGES】")
                    # log(json.dumps(messages))

                    response = anthropic_client.messages.create(
                        model=LLM_VISION_MODEL,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}, # For 8K output https://x.com/alexalbert__/status/1812921642143900036
                    )
                else:

                    log(f"【MODEL】:{model}")

                    messages = [{"role": "user", "content": prompt}]
                    response = anthropic_client.messages.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}, # For 8K output https://x.com/alexalbert__/status/1812921642143900036
                    )

                log(f"【USAGE】input_tokens :{response.usage.input_tokens}")
                log(f"【USAGE】output_tokens :{response.usage.output_tokens}")

                return response.content[0].text.strip()
            elif model.lower().startswith("openai/"): # OpenRouter's OpenAI

                openai_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=OPENROUTER_API_KEY)

                log(f"【MODEL】: OpenRouter {model}")

                messages = [{"role": "system", "content": prompt}]
                response = openai_client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://github.com/saten-private/BabyCommandAGI", # Optional, for including your app on openrouter.ai rankings.
                        "X-Title": "BabyCommandAGI", # Optional. Shows in rankings on openrouter.ai.
                    },
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                return response.choices[0].message.content.strip()

            else:

                openai_client = OpenAI(api_key=OPENAI_API_KEY)

                separated_content = separate_markdown(prompt) # for Vision API
                if len(separated_content) > 1:

                    log(f"【MODEL】:{LLM_VISION_MODEL}")

                    messages = [
                        {
                            "role": "system",
                            "content": modify_parts_to_new_format_openai(separated_content)
                        }
                    ]

                    # log("【MESSAGES】")
                    # log(json.dumps(messages))

                    response = openai_client.chat.completions.create(
                        model=LLM_VISION_MODEL,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:

                    log(f"【MODEL】:{model}")

                    messages = [{"role": "system", "content": prompt}]
                    response = openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                return response.choices[0].message.content.strip()
        # OpenAI
        except openai.RateLimitError as e:
            log(
                f"   *** The OpenAI API rate limit has been exceeded. Waiting 300 seconds and trying again. error: {str(e)} ***"
            )
            time.sleep(300)  # Wait seconds and try again
        except openai.APITimeoutError as e:
            log(
                f"   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. error: {str(e)} ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.APIError as e:
            log(
                f"   *** OpenAI API error occurred. Waiting 300 seconds and trying again. error: {str(e)} ***"
            )
            time.sleep(300)  # Wait seconds and try again
        except openai.APIConnectionError as e:
            log(
                f"   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 300 seconds and trying again. error: {str(e)} ***"
            )
            time.sleep(300)  # Wait seconds and try again
        except openai.BadRequestError as e:
            log(
                f"   *** OpenAI API BadRequestError. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. error: {str(e)} ***"
            )
            raise e
        except openai.InternalServerError as e:
            log(
                f"   *** OpenAI API InternalServerError. error: {str(e)} ***"
            )
            raise e
        # Anthropic
        except anthropic.RateLimitError as e:
            log(
                f"   *** The Anthropic API rate limit has been exceeded. Waiting 300 seconds and trying again. error: {str(e)} ***"
            )
            time.sleep(300)  # Wait seconds and try again
        except anthropic.APIConnectionError as e:
            log(
                f"   *** Anthropic API connection error occurred. error: {str(e)} ***"
            )
            raise e
        except anthropic.APIStatusError as e:
            log(
                f"   *** Anthropic API status error occurred. error: {str(e)} ***"
            )
            raise e
        except Exception as e:
            log(
                f"   *** Other error occurred: {str(e)} ***"
            )
            raise e

# Global variable for flagging input
input_flag = None

def check_input():
    global input_flag
    while True:
        time.sleep(2)
        if input_flag == "f" or input_flag == "a":
            continue
        log("\n" + "\033[33m\033[1m" + 'The system now accepts "f" if you want to send feedback to the AI, or "a" if you want to send the answer to the shell.' + "\033[0m\033[0m" + "\n")
        inp = input()
        if inp == "f" or inp == "a":
            input_flag = inp

# Thread for non-blocking input check
input_thread = threading.Thread(target=check_input, daemon=True)
input_thread.start()

def check_completion_agent(
        objective: str, enriched_result: dict, task_list: deque, executed_task_list: deque, current_dir: str
):
    prompt = f"""You are the best engineer and manage the tasks to achieve the "{objective}". 

Please try to make the tasks you generate as necessary so that they can be executed by writing a single file or in a terminal. If that's difficult, generate "plan" tasks with reduced granularity.

Follow the format in "Example X of tasks output" below to output next tasks. Please never output anything other than a "Example X of tasks output" format that always includes "type:" before ``` blocks. Please never output 'sudo' commands.

Below is the result of the last execution."""

    if enriched_result["type"].startswith("entire_file_after_writing"):
        prompt += f"""
        
# Path where the file was written

{enriched_result["target"]}

# Entire contents of the written file

{enriched_result["result"]}"""
        
    elif enriched_result["type"].startswith("failed_to_save_due_to_invalid_content"):
        prompt += f"""
        
# Pass that I tried to save but failed.

{enriched_result["target"]}

# Invalid content that failed to save

```
{enriched_result["result"]}
```"""
        
    elif enriched_result["type"].startswith("failed_modify_partial_due_to_no_file"):
        prompt += f"""
        
# Failed to make modifications because the file was missing.

path: {enriched_result["target"]}"""
        
    elif enriched_result["type"].startswith("command"):
        prompt += f"""

# Current directory

{current_dir}

# Command executed most recently

{enriched_result["target"]}

# Result of last command executed

{enriched_result["result"]}"""
        
    if len(executed_task_list) > 1:
        after_executed_task_list = executed_task_list.copy()
        after_executed_task_list.popleft()
        prompt += f"""
        
# The list of results executed most recently after that.

{ExecutedTaskParser().encode(after_executed_task_list)}"""

    prompt += f"""

# Uncompleted tasks

{TaskParser().encode(task_list)}"""

    prompt = limit_tokens_from_string(prompt, TOKEN_COUNT_MODEL, MAX_INPUT_TOKEN)
    prompt = TaskParser().close_open_backticks(prompt)
    prompt += """

# Example 1 of tasks output

type: create
path: /workspace/requirements.txt
```
dataclasses
```

type: command
path: /workspace/
```bash
pip install -r requirements.txt
source venv/bin/activate
```

type: plan
```
Designing a Minesweeper.
```

type: create
path: /workspace/minesweeper.py
```python
from board import Board

class Minesweeper:
    def __init__(self):
        self.board = Board()

    def start_game(self):
        while not game_over:
            row, col, action = self.play_turn()
            if action == "r":
                game_over = self.board.reveal_cell(row, col)
            elif action == "f":
                self.board.flag_cell(row, col)

            if self.board.is_game_over():
                break

        self.display_board()
        print("Game Over!")
```

type: modify_partial
path: /workspace/minesweeper.py
```python
from board import Board

class Minesweeper:
    def __init__(self, rows: int, cols: int, mines: int):
        self.board = Board(rows, cols, mines)

    def start_game(self):
        game_over = False

        # ... Rest of the code remains unchanged ...
```

type: modify_partial
path: /workspace/minesweeper.py
```python
- if action == "R":
-     game_over = self.board.reveal_cell(row, col)
- elif action == "F":
-     self.board.flag_cell(row, col)
+# if action == "r":
+#     game_over = self.board.reveal_cell(row, col)
+# elif action == "f":
+#     self.board.flag_cell(row, col)
```

# Example 2 of tasks output

type: command
path: /workspace/
```
pip install curl
```

type: command
path: /workspace/
```
curl -s https://wttr.in/Tokyo
```

# Example 3 of tasks output

type: command
path: /workspace/
```
apt-get update
apt-get install -y git
apt-get install -y npm
git clone https://github.com/samuelcust/flappy-bird-assets.git
```

type: command
path: /workspace/flappy-bird-assets/
```
npm init -y
npm install express
```

type: create
path: /workspace/flappy-bird-assets/server.js
```
const express = require('express');
const app = express();
const PORT = 8080;

// Serve static files from the root directory
app.use(express.static(__dirname));

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Flappy Bird game running at http://0.0.0.0:${PORT}/`);
});
```

type: create
path: /workspace/flappy-bird-assets/index.html
```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flappy Bird</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/p5.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/addons/p5.sound.min.js"></script>
    <script src="flappy-bird.js"></script>
</head>
<body>
</body>
</html>
```

type: create
path: /workspace/flappy-bird-assets/flappy-bird.js
```
let bird;
let pipes = [];
let score = 0;
let birdImg = [];
let pipeImg;
let bgImg;
let scoreSound;
let hitSound;
let baseImg;
let baseX = 0;

function preload() {
  // Load bird images
  birdImg[0] = loadImage('sprites/yellowbird-downflap.png');
  birdImg[1] = loadImage('sprites/yellowbird-midflap.png');
  birdImg[2] = loadImage('sprites/yellowbird-upflap.png');

  // Load other images
  pipeImg = loadImage('sprites/pipe-green.png');
  bgImg = loadImage('sprites/background-day.png');
  baseImg = loadImage('sprites/base.png');

  // Load sounds
  scoreSound = loadSound('audio/point.ogg');
  hitSound = loadSound('audio/hit.ogg');
}

function setup() {
  createCanvas(400, 600);
  bird = new Bird();
  pipes.push(new Pipe());
}

function draw() {
  image(bgImg, 0, 0, width, height);

  // Draw and update bird
  bird.update();

  if (bird.touchesGround()) {
    console.log('GAME OVER - TOUCHED GROUND');
    hitSound.play();
    noLoop();
  }

  bird.show();

  // Check if a pipe is off-screen and generate new pipes
  if (pipes[0].x + pipes[0].width < 0) {
    score++;
    scoreSound.play();
    pipes.shift();
    pipes.push(new Pipe());
  }

  // Draw and update pipes
  for (let pipe of pipes) {
    pipe.update();
    pipe.show();

    if (pipe.hits(bird)) {
      console.log('GAME OVER');
      hitSound.play();
      noLoop();
    }
  }

  // Display moving base
  image(baseImg, baseX, height - baseImg.height);
  image(baseImg, baseX + baseImg.width, height - baseImg.height);
  baseX -= 2;
  if (baseX <= -baseImg.width) {
    baseX = 0;
  }

  // Show score
  fill(255);
  textSize(32);
  text(score, 10, 32);
}

function mouseClicked() {
  bird.flap();
}

class Bird {
  constructor() {
    this.y = height / 2;
    this.x = 50;
    this.size = 32;
    this.gravity = 0.15;  // Make the bird fall more slowly
    this.lift = -4;      // Make the bird rise more slowly
    this.velocity = 0;
    this.animation = birdImg;
    this.index = 0;
  }

  touchesGround() {
    return this.y + (this.size / 2) >= height - baseImg.height;
  }

  update() {
    this.y += this.velocity;
    this.velocity += this.gravity;
    this.y = constrain(this.y, 0, height - baseImg.height);

    // Cycle through bird images for animation
    this.index += 0.2;
    if (this.index >= this.animation.length) {
      this.index = 0;
    }
  }

  show() {
    let img = this.animation[floor(this.index)];
    image(img, this.x, this.y);
  }

  flap() {
    this.velocity += this.lift;
  }
}

class Pipe {
  constructor() {
    this.spacing = 150;
    this.top = random(height / 6, (2 / 3) * height);
    this.bottom = height - (this.top + this.spacing) - baseImg.height;
    this.x = width;
    this.width = pipeImg.width;
    this.speed = 2;
  }

  hits(bird) {
    if (bird.y < this.top || bird.y > height - this.bottom - baseImg.height) {
      if (bird.x > this.x && bird.x < this.x + this.width) {
        return true;
      }
    }
    return false;
  }

  update() {
    this.x -= this.speed;
  }

  show() {
    // Display top pipe
    image(pipeImg, this.x, 0, this.width, this.top);
    // Display bottom pipe (flipped)
    image(pipeImg, this.x, height - this.bottom, this.width, this.bottom);
  }
}
```

type: command
path: /workspace/flappy-bird-assets/
```
node server.js
```

# Next tasks output

"""

    log("\033[34m\033[1m" + "[[Prompt]]" + "\033[0m\033[0m" + "\n\n" + prompt +
        "\n\n")
    responseString = llm_call(prompt)
    log("\033[31m\033[1m" + "[[Response]]" + "\033[0m\033[0m" + "\n\n" +
        responseString + "\n\n")
    if responseString.startswith("Complete"):
        return responseString
    try:
        return TaskParser().decode(responseString)
    except Exception as error:
        log("task parse error:")
        log(error)
        log("\nRetry\n\n")
        return check_completion_agent(objective, enriched_result, task_list, executed_task_list, current_dir)

def plan_agent(objective: str, task: str,
               executed_task_list: deque, current_dir: str):
  #context = context_agent(index=YOUR_TABLE_NAME, query=objective, n=5)
    prompt = f"""You are a best engineer.
To achieve the "{objective}" from the following executed result states, before you begin the following single task, please make your own assumptions, clarify them, and then execute, and absolutely output next tasks in the format of "Example X of tasks output" that always includes "type:" before ``` blocks. Please never output 'sudo' commands.

# Task to be performed.

{task}

# Current directory

{current_dir}

# List of most recently executed results

{ExecutedTaskParser().encode(executed_task_list)}"""
    
    prompt = limit_tokens_from_string(prompt, TOKEN_COUNT_MODEL, MAX_INPUT_TOKEN)
    prompt = TaskParser().close_open_backticks(prompt)
    prompt += """

# Example 1 of tasks output

type: create
path: /workspace/requirements.txt
```
dataclasses
```

type: command
path: /workspace/
```bash
pip install -r requirements.txt
source venv/bin/activate
```

type: plan
```
Designing a Minesweeper.
```

type: create
path: /workspace/minesweeper.py
```python
from board import Board

class Minesweeper:
    def __init__(self):
        self.board = Board()

    def start_game(self):
        while not game_over:
            row, col, action = self.play_turn()
            if action == "r":
                game_over = self.board.reveal_cell(row, col)
            elif action == "f":
                self.board.flag_cell(row, col)

            if self.board.is_game_over():
                break

        self.display_board()
        print("Game Over!")
```

type: modify_partial
path: /workspace/minesweeper.py
```python
from board import Board

class Minesweeper:
    def __init__(self, rows: int, cols: int, mines: int):
        self.board = Board(rows, cols, mines)

    def start_game(self):
        game_over = False

        # ... Rest of the code remains unchanged ...
```

type: modify_partial
path: /workspace/minesweeper.py
```python
- if action == "R":
-     game_over = self.board.reveal_cell(row, col)
- elif action == "F":
-     self.board.flag_cell(row, col)
+# if action == "r":
+#     game_over = self.board.reveal_cell(row, col)
+# elif action == "f":
+#     self.board.flag_cell(row, col)
```

# Example 2 of tasks output

type: command
path: /workspace/
```
pip install curl
```

type: command
path: /workspace/
```
curl -s https://wttr.in/Tokyo
```

# Example 3 of tasks output

type: command
path: /workspace/
```
apt-get update
apt-get install -y git
apt-get install -y npm
git clone https://github.com/samuelcust/flappy-bird-assets.git
```

type: command
path: /workspace/flappy-bird-assets/
```
npm init -y
npm install express
```

type: create
path: /workspace/flappy-bird-assets/server.js
```
const express = require('express');
const app = express();
const PORT = 8080;

// Serve static files from the root directory
app.use(express.static(__dirname));

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Flappy Bird game running at http://0.0.0.0:${PORT}/`);
});
```

type: create
path: /workspace/flappy-bird-assets/index.html
```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flappy Bird</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/p5.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/addons/p5.sound.min.js"></script>
    <script src="flappy-bird.js"></script>
</head>
<body>
</body>
</html>
```

type: create
path: /workspace/flappy-bird-assets/flappy-bird.js
```
let bird;
let pipes = [];
let score = 0;
let birdImg = [];
let pipeImg;
let bgImg;
let scoreSound;
let hitSound;
let baseImg;
let baseX = 0;

function preload() {
  // Load bird images
  birdImg[0] = loadImage('sprites/yellowbird-downflap.png');
  birdImg[1] = loadImage('sprites/yellowbird-midflap.png');
  birdImg[2] = loadImage('sprites/yellowbird-upflap.png');

  // Load other images
  pipeImg = loadImage('sprites/pipe-green.png');
  bgImg = loadImage('sprites/background-day.png');
  baseImg = loadImage('sprites/base.png');

  // Load sounds
  scoreSound = loadSound('audio/point.ogg');
  hitSound = loadSound('audio/hit.ogg');
}

function setup() {
  createCanvas(400, 600);
  bird = new Bird();
  pipes.push(new Pipe());
}

function draw() {
  image(bgImg, 0, 0, width, height);

  // Draw and update bird
  bird.update();

  if (bird.touchesGround()) {
    console.log('GAME OVER - TOUCHED GROUND');
    hitSound.play();
    noLoop();
  }

  bird.show();

  // Check if a pipe is off-screen and generate new pipes
  if (pipes[0].x + pipes[0].width < 0) {
    score++;
    scoreSound.play();
    pipes.shift();
    pipes.push(new Pipe());
  }

  // Draw and update pipes
  for (let pipe of pipes) {
    pipe.update();
    pipe.show();

    if (pipe.hits(bird)) {
      console.log('GAME OVER');
      hitSound.play();
      noLoop();
    }
  }

  // Display moving base
  image(baseImg, baseX, height - baseImg.height);
  image(baseImg, baseX + baseImg.width, height - baseImg.height);
  baseX -= 2;
  if (baseX <= -baseImg.width) {
    baseX = 0;
  }

  // Show score
  fill(255);
  textSize(32);
  text(score, 10, 32);
}

function mouseClicked() {
  bird.flap();
}

class Bird {
  constructor() {
    this.y = height / 2;
    this.x = 50;
    this.size = 32;
    this.gravity = 0.15;  // Make the bird fall more slowly
    this.lift = -4;      // Make the bird rise more slowly
    this.velocity = 0;
    this.animation = birdImg;
    this.index = 0;
  }

  touchesGround() {
    return this.y + (this.size / 2) >= height - baseImg.height;
  }

  update() {
    this.y += this.velocity;
    this.velocity += this.gravity;
    this.y = constrain(this.y, 0, height - baseImg.height);

    // Cycle through bird images for animation
    this.index += 0.2;
    if (this.index >= this.animation.length) {
      this.index = 0;
    }
  }

  show() {
    let img = this.animation[floor(this.index)];
    image(img, this.x, this.y);
  }

  flap() {
    this.velocity += this.lift;
  }
}

class Pipe {
  constructor() {
    this.spacing = 150;
    this.top = random(height / 6, (2 / 3) * height);
    this.bottom = height - (this.top + this.spacing) - baseImg.height;
    this.x = width;
    this.width = pipeImg.width;
    this.speed = 2;
  }

  hits(bird) {
    if (bird.y < this.top || bird.y > height - this.bottom - baseImg.height) {
      if (bird.x > this.x && bird.x < this.x + this.width) {
        return true;
      }
    }
    return false;
  }

  update() {
    this.x -= this.speed;
  }

  show() {
    // Display top pipe
    image(pipeImg, this.x, 0, this.width, this.top);
    // Display bottom pipe (flipped)
    image(pipeImg, this.x, height - this.bottom, this.width, this.bottom);
  }
}
```

type: command
path: /workspace/flappy-bird-assets/
```
node server.js
```

# Next tasks output

"""

    log("\033[34m\033[1m" + "[[Prompt]]" + "\033[0m\033[0m" + "\n\n" + prompt +
        "\n\n")
    responseString = llm_call(prompt)
    log("\033[31m\033[1m" + "[[Response]]" + "\033[0m\033[0m" + "\n\n" +
        responseString + "\n\n")
    try:
        return TaskParser().decode(responseString)
    except Exception as error:
        log("task parse error:")
        log(error)
        log("\nRetry\n\n")
        return plan_agent(objective, task, executed_task_list, current_dir)

def list_std_blocks(target_list: list) -> list[str]:
    std_blocks = []
    if target_list:
        is_target_list_break = False
        for read in target_list:
            buffer = bytes()
            while True:
                try:
                    chunk = os.read(read, 1024)
                except:
                    is_target_list_break = True
                    break

                if not chunk:
                    # Nothing more to read
                    break

                buffer += chunk
                is_read_break = False
                while buffer:
                    try:
                        # Try to decode the entire buffer
                        text = buffer.decode('utf8')
                        buffer = bytes()
                    except UnicodeDecodeError as e:
                        if e.reason == 'unexpected end of data':
                            # Need more data
                            break
                        else:
                            # Something else happened. Depending on your needs, you might want
                            # to break and re-raise the exception, or skip this character,
                            # or replace it, or something else.
                            log(f"\nUnicodeDecodeError:\n {e}\n\n")
                            raise e
                    else:
                        output_block = text
                        if output_block:
                            print(output_block, end="")
                            std_blocks.append(output_block)
                        is_read_break = True
                        break
                
                if is_read_break == True:
                    break
            
            if is_target_list_break == True:
                break

    return std_blocks

# Execute a command
def execution_command(objective: str, command: str, task_list: deque,
                      executed_task_list: deque, current_dir: str) -> str:
    global pty_master
    global input_flag
    if pty_master is not None:
        os.close(pty_master)
        pty_master = None
        time.sleep(1)

    #[Test]
    #command = "export PATH=$PATH:$PWD/flutter/bin"

    #log("saburo:")
    # output environment variables
    #for key, value in os.environ.items():
    #    log(f"{key}: {value}")

    # Read the dumped environment variables
    if os.path.isfile(ENV_DUMP_FILE):
        with open(ENV_DUMP_FILE, "r") as env_file:
            for line in env_file:
                # Skip lines with null bytes
                if '\0' in line:
                    continue
                name, _, value = line.partition("=")
                # Remove any null bytes from the value
                value = value.replace('\0', '')
                #log(f"new environment:{value.strip()}")
                os.environ[name] = value.strip()  # Set the environment variable in the parent process
                #log(f"set environment:{os.environ[name]}")

    log(f"current_dir:\n{current_dir}\n")

    log("\033[33m\033[1m" + "[[Input]]" + "\033[0m\033[0m" + "\n\n" + command +
        "\n")

    # Add an extra command to dump environment variables to a file
    command_to_execute = f"cd {current_dir}; {command}; echo $? > /tmp/cmd_exit_status; pwd > {PWD_FILE}; env > {ENV_DUMP_FILE}"

    pty_master, slave = pty.openpty()

    try:
        process = subprocess.Popen(command_to_execute,
                                stdin=slave,
                                stdout=slave,
                                stderr=slave,
                                shell=True,
                                text=True,
                                bufsize=1,
                                env=os.environ)
        os.close(slave)

        std_blocks = []

        start_time = time.time()
        notification_time = time.time()
        print("\n" + "\033[33m\033[1m" + '"f": go to "feedback"' + "\033[0m\033[0m" + "\n")

        result = ""
        end_prcessing = False
        wait_input = False

        while process.poll() is None:

            if notification_time + 30 < time.time():
                notification_time = time.time()
                if end_prcessing:
                    end_prcessing = False
                    wait_input = False
                else:
                    if wait_input:
                        print("\n" + "\033[33m\033[1m" + 'Enter "f" if you want to send feedback to the AI or "a" if you want to send the answer to the shell.' + "\033[0m\033[0m" + "\n")
                    else:
                        print("\n" + "\033[33m\033[1m" + 'Enter "f" if you want to send feedback to the AI or "a" if you want to send the answer to the shell.' + "\033[0m\033[0m" + "\n")
            
            # Check for output with a timeout of some minutes
            rlist, wlist, xlist = select.select([pty_master], [], [], 2)
            if rlist or wlist or xlist:
                std_blocks.extend(list_std_blocks(rlist))
                std_blocks.extend(list_std_blocks(wlist))
                std_blocks.extend(list_std_blocks(xlist))

            else:
                if LLM_COMMAND_RESPONSE:
                    if time.time() - start_time > 300 and result == "":
                        start_time = time.time()

                        # Concatenate the output and split it by lines
                        stdout_lines = "".join(std_blocks).splitlines()

                        # No output received within 5 miniutes, call the check_wating_for_response function with the last 3 lines or the entire content
                        lastlines = stdout_lines[-3:] if len(stdout_lines) >= 3 else stdout_lines
                        lastlines = "\n".join(lastlines)
                        llm_command_response = llm_command_response_for_waiting(objective, lastlines, command,
                                                "".join(std_blocks), task_list,
                                                executed_task_list, current_dir)
                        if llm_command_response.startswith('BabyCommandAGI: Complete'):
                            result = 'Waiting for user feedback'
                        elif llm_command_response.startswith('BabyCommandAGI: Interruption'):
                            result = 'The command was interrupted by BabyCommandAGI'
                        elif llm_command_response.startswith('BabyCommandAGI: Continue'):
                            pass
                        else:
                            llm_command_response += '\n'
                            os.write(pty_master, llm_command_response.encode())

            if input_flag == "f":
                result = 'Waiting for user feedback'
            
            if result != "":
                if input_flag is None:
                    if wait_input == False:
                        wait_input = True
                        log("\n" + "\033[33m\033[1m" + 'Enter "f" if you want to send feedback to the AI or "a" if you want to send the answer to the shell.' + "\033[0m\033[0m" + "\n")
                    continue

                if end_prcessing:
                    continue
                log("\n" + "\033[33m\033[1m" + 'Enter your answer to the shell to end the process before sending feedback to the AI. If you do not need the answer to the shell, leave it empty and type [Enter].' + "\033[0m\033[0m" + "\n")
                response = input()
                if response == "":
                    break
                response += '\n'
                os.write(pty_master, response.encode())
                end_prcessing = True
                notification_time = time.time()
                start_time = time.time()

                if input_flag == "a":
                    input_flag = None

            elif input_flag == "a":
                log("\n" + "\033[33m\033[1m" + 'Enter your answer to the shell.' + "\033[0m\033[0m" + "\n")
                response = input()
                response += '\n'
                os.write(pty_master, response.encode())
                notification_time = time.time()
                start_time = time.time()
                input_flag = None


        out = "".join(std_blocks)

        if input_flag == "f":
            result = 'Waiting for user feedback'

        if result == "":
            with open("/tmp/cmd_exit_status", "r") as status_file:
                cmd_exit_status = int(status_file.read().strip())

            result = f"The Return Code for the command is {cmd_exit_status}"

        all_result = f"{result}:\n{out}"
        result = all_result

        log("\n" + "\033[33m\033[1m" + "[[Output]]" + "\033[0m\033[0m" + "\n\n" +
            result + "\n\n")
    finally:
        os.close(pty_master)
        pty_master = None
        process.terminate()
        process.wait()
        if input_flag == "a":
            input_flag = None

    
    return result

def llm_command_response_for_waiting(objective: str, lastlines: str, command: str,
                           all_output_for_command: str, task_list: deque,
                           executed_task_list: deque, current_dir: str) -> str:
    prompt = f"""You are an expert in shell commands to achieve the "{objective}".
Based on the information below, if the objective has been achieved, please output only 'BabyCommandAGI: Complete'.
Based on the information below, if the objective cannot be achieved and it seems that the objective can be achieved by inputting while waiting for the user's input, please output only the input content for the waiting input content to achieve the objective.
Based on the information below, if the objective cannot be achieved and it seems better to interrupt the execution of the command to achieve the objective, please output only 'BabyCommandAGI: Interruption'.
Otherwise, please output only 'BabyCommandAGI: Continue'.

# All output content so far for the command being executed
{all_output_for_command}

# List of most recently executed results
{ExecutedTaskParser().encode(executed_task_list)}

# Uncompleted tasks
{TaskParser().encode(task_list)}"""

    prompt = limit_tokens_from_string(prompt, TOKEN_COUNT_MODEL, MAX_INPUT_TOKEN)
    prompt = TaskParser().close_open_backticks(prompt)
    prompt += f"""

# Current directory
{current_dir}

# Command being executed
{command}

# The last 3 lines of the terminal
{lastlines}

# Absolute rule
Please output only the following relevant content. Never output anything else.

If the objective has been achieved: 'BabyCommandAGI: Complete'
If the objective cannot be achieved and it seems that the objective can be achieved by inputting while waiting for the user's input: Input content for the waiting input content to achieve the objective
If the objective cannot be achieved and it seems better to interrupt the execution of the command to achieve the objective: 'BabyCommandAGI: Interruption'
In cases other than the above: 'BabyCommandAGI: Continue'"""

    log("\n\n")
    log("\033[34m\033[1m" + "[[Prompt]]" + "\033[0m\033[0m" + "\n\n" + prompt +
        "\n\n")
    result = llm_call(prompt)
    log("\033[31m\033[1m" + "[[Response]]" + "\033[0m\033[0m" + "\n\n" +
        result + "\n\n")
    return result

def analyze_command_result(result: str) -> str:
    lastString = last_tokens_from_string(result, TOKEN_COUNT_MODEL, MAX_COMMAND_RESULT_TOKEN)
    result_lines = lastString.split('\n')[-100:]  # Extract the last many lines
    for idx, line in enumerate(result_lines):
        if "fail" in line.lower() or "error" in line.lower():
            start_idx = max(0, idx - 10)  # Start from 10 lines before the "failure" line
            return '\n'.join(result_lines[start_idx:])  # Return all lines from the first match
    return '\n'.join(result_lines)  # If no match, return the last many lines

def write_file(file_path: str, content: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            file.write(content)
    except IsADirectoryError:
        return
    except Exception as e:
        raise e

def merge_file(base_content: str, modify_content: str) -> str:
    
    prompt = f"""You are the best engineer.
Please merge the following code modifications into the base code.

Base code:
```
{base_content}
```

Code modifications:
```
{modify_content}
```

The merged code:
"""

    log("\n\n")
    log("\033[34m\033[1m" + "[[Prompt]]" + "\033[0m\033[0m" + "\n\n" + prompt +
        "\n\n")
    result = llm_call(prompt).strip()
    log("\033[31m\033[1m" + "[[Response]]" + "\033[0m\033[0m" + "\n\n" +
        result + "\n\n")

    if "```" in result:
        result = result.split("```")[1]
        if "\n" in result:
            result_lines = result.split("\n")[1:] # Delete up to the first line break.
            result = "\n".join(result_lines).strip()
        else:
            log("merge parse error")
            log("\nRetry\n\n")
            return merge_file(base_content, modify_content)

    return result

def user_feedback() -> str:

    log("\033[33m\033[1m" + "*****USER FEEDBACK*****\n\n" + "\033[0m\033[0m")

    # Ask the user in English
    log('Please enter feedback to the AI on how the OBJECTIVE can be achieved. The AI will continue to execute based on feedback: \n')
    response = input()
    log('\n')

    log("\033[33m\033[1m" + "[[Feedback]]" + "\n\n" + response + "\033[0m\033[0m" + "\n")
    return response

def check_first_two_lines_same_comment_style(text):
    """
    To detect if it starts with a comment because there are many differential updates, etc.
    
    Extracts the first two lines from the given string and checks if both start with either '# ' or both start with '// '.
    If there's only one line, it checks if that line starts with '# ' or '// '.

    Args:
    text (str): The string from which the lines are to be extracted.

    Returns:
    bool: True if the first two lines (or the only line if one exists) start with the same comment style ('# ' or '// '), False otherwise.
    """
    # Split the text into lines
    lines = text.split('\n')

    # Extract the first two lines
    lines_to_check = lines[:2]

    # Check for the case where there is only one line
    if len(lines_to_check) == 1:
        return lines_to_check[0].strip().startswith(('# ', '// '))

    # Check if both lines start with the same comment style
    return (lines_to_check[0].strip().startswith('# ') and lines_to_check[1].strip().startswith('# ')) or \
           (lines_to_check[0].strip().startswith('// ') and lines_to_check[1].strip().startswith('// '))

# Add the initial task if starting new objective
if tasks_storage.is_empty() or JOIN_EXISTING_OBJECTIVE:
    first_task = {"type": "command", "content": f"cd {WORKSPACE_FOLDER}"}
    tasks_storage.append(first_task)
    initial_task = {"type": "plan", "content": INITIAL_TASK}
    tasks_storage.append(initial_task)

pty_master = None

def execute_objective():
    global OBJECTIVE
    global input_flag

    current_dir = WORKSPACE_FOLDER
    if os.path.isfile(PWD_FILE):
        with open(PWD_FILE, "r") as pwd_file:
            current_dir = pwd_file.read().strip()

    new_tasks_list = []
    while True:
        # As long as there are tasks in the storage...
        if len(tasks_storage.get_tasks()) == 0:
            break
        else:
            # Pull the first task
            task = tasks_storage.popleft()
            log("\033[92m\033[1m" + "*****NEXT TASK*****\n\n" + "\033[0m\033[0m")
            log(str(task['type']) + ": " + task['content'] + "\n\n")

            # Check executable content
            if task['type'].startswith("create") or task['type'].startswith("modify") or task['type'].startswith("modify_partial") or task['type'].startswith("command"):

                enriched_result = {}
                is_check_result = False
                is_next_plan = False
                is_complete = False
                while True:

                    if task['type'].startswith("create") or task['type'].startswith("modify") or task['type'].startswith("modify_partial"):
                        path = task['path']
                        content = task['content']

                        # Ensure that results are not ignored.
                        if path.endswith(".sh"):
                            content = content.replace(" || true", "")

                        log("task['type']: " + task['type'] + "\n\n")
                        log("path: " + path + "\n\n")
                        log(content + "\n\n")

                        has_rest_code = False
                        content_lines = content.split("\n")
                        for line in content_lines:
                            line = line.strip()
                            if line.startswith("//"):
                                if " rest " in line.lower() or " existing " in line.lower():
                                    has_rest_code = True
                        if task['type'].startswith("modify_partial") or has_rest_code or check_first_two_lines_same_comment_style(content):
                            log("\033[33m\033[1m" + "*****MODIFY TASK*****\n\n" + "\033[0m\033[0m")

                            try:
                                with open(path, 'r', encoding='utf-8') as file:
                                    base_content = file.read()

                                    new_content = merge_file(base_content, content)

                                    write_file(path, new_content)

                                    # Enrich result and store
                                    save_data(tasks_storage.get_tasks(), TASK_LIST_FILE)

                                    enriched_result = {
                                        "type": "entire_file_after_writing",
                                        "target": path,
                                        "result": new_content
                                        }
                                    # TODO: Verification of execution results required. 
                                    # It is possible that the deletion of duplicate writes may have caused post-write execution errors to incorrectly recognize the context and because the API now supports long contexts,
                                    # try to disable the process of removing duplicate write execution results
                                    # executed_tasks_storage.remove_target_write_dicts(path)

                                    executed_tasks_storage.appendleft(enriched_result)
                                    save_data(executed_tasks_storage.get_tasks(), EXECUTED_TASK_LIST_FILE)
                                            
                                    # API now supports long contexts, disable processing of the maximum value in the execution history
                                    # if len(executed_tasks_storage.get_tasks()) > 30:
                                    #     executed_tasks_storage.pop()
                                        
                                    if tasks_storage.is_empty():
                                        break
                                    else:
                                        next_task = tasks_storage.reference(0)
                                        if next_task['type'].startswith("create") or next_task['type'].startswith("modify") or next_task['type'].startswith("modify_partial") or next_task['type'].startswith("command"):
                                            task = tasks_storage.popleft()
                                        else:
                                            is_next_plan = True
                                            break                            
                            except FileNotFoundError:
                                log("*MODIFY PATH FILE NOTHING*")

                                # Enrich result and store
                                save_data(tasks_storage.get_tasks(), TASK_LIST_FILE)

                                enriched_result = {
                                    "type": "failed_modify_partial_due_to_no_file",
                                    "target": path,
                                    "result": content
                                    }
                                executed_tasks_storage.appendleft(enriched_result)
                                save_data(executed_tasks_storage.get_tasks(), EXECUTED_TASK_LIST_FILE)
                                break
                            except Exception as e:
                                log(
                                    f"   *** Other error occurred: {str(e)} ***"
                                )
                                raise e

                        else:
                            log("\033[33m\033[1m" + "*****CREATE TASK*****\n\n" + "\033[0m\033[0m")

                            try:
                                write_file(path, content)
                            except Exception as e:
                                log(
                                    f"   *** Other error occurred: {str(e)} ***"
                                )
                                raise e

                            # Step 2: Enrich result and store
                            save_data(tasks_storage.get_tasks(), TASK_LIST_FILE)

                            enriched_result = {
                                "type": "entire_file_after_writing",
                                "target": path,
                                "result": content
                                }
                            
                            # TODO: Verification of execution results required. 
                            # It is possible that the deletion of duplicate writes may have caused post-write execution errors to incorrectly recognize the context and because the API now supports long contexts,
                            # try to disable the process of removing duplicate write execution results
                            # executed_tasks_storage.remove_target_write_dicts(path)

                            executed_tasks_storage.appendleft(enriched_result)
                            save_data(executed_tasks_storage.get_tasks(), EXECUTED_TASK_LIST_FILE)
                                    
                            # API now supports long contexts, disable processing of the maximum value in the execution history
                            # if len(executed_tasks_storage.get_tasks()) > 30:
                            #     executed_tasks_storage.pop()
                                
                            if tasks_storage.is_empty():
                                break
                            else:
                                next_task = tasks_storage.reference(0)
                                if next_task['type'].startswith("create") or next_task['type'].startswith("modify") or next_task['type'].startswith("modify_partial") or next_task['type'].startswith("command"):
                                    task = tasks_storage.popleft()
                                else:
                                    is_next_plan = True
                                    break

                    elif task['type'].startswith("command"):

                        log("\033[33m\033[1m" + "*****EXCUTE COMMAND TASK*****\n\n" + "\033[0m\033[0m")

                        if 'path' in task:
                            if current_dir.strip().rstrip('/') != task['path'].strip().rstrip('/'):
                                current_dir = task['path'].strip().rstrip('/')
                                command = f"cd {current_dir}"
                                all_result = execution_command(OBJECTIVE, command, tasks_storage.get_tasks(),
                                                executed_tasks_storage.get_tasks(), current_dir)
                                enriched_result = { "type": "command", "target": command}
                                if all_result.startswith("The Return Code for the command is 0:"):
                                    enriched_result['result'] = all_result #"Success"
                                else:
                                    enriched_result['result'] = all_result
                                if os.path.isfile(PWD_FILE):
                                    with open(PWD_FILE, "r") as pwd_file:
                                        current_dir = pwd_file.read().strip()

                                executed_tasks_storage.appendleft(enriched_result)
                                save_data(executed_tasks_storage.get_tasks(), EXECUTED_TASK_LIST_FILE)

                                # API now supports long contexts, disable processing of the maximum value in the execution history
                                # if len(executed_tasks_storage.get_tasks()) > 30:
                                #     executed_tasks_storage.pop()

                        while True:
                            content = task['content'].strip()
                            if content == "":
                                break
                            commands = deque(content.split("\n"))
                            command = commands.popleft()
                            if command.strip().startswith("#"):
                                task['content'] = "\n".join(list(commands))
                                continue
                            # Ensure that results are not ignored.
                            command = command.replace(" || true", "")
                            # Remove "sudo" because "docker attach" probably does not read output executed by "sudo".
                            command = command.replace("sudo ", "")
                            all_result = execution_command(OBJECTIVE, command, tasks_storage.get_tasks(),
                                            executed_tasks_storage.get_tasks(), current_dir)
                            
                            # TODO: Verification of execution results required. 
                            # It is possible that the deletion of duplicate writes may have caused post-write execution errors to incorrectly recognize the context and because the API now supports long contexts,
                            # stop partial extraction of command execution results.
                            result = all_result # analyze_command_result(all_result)

                            if os.path.isfile(PWD_FILE):
                                with open(PWD_FILE, "r") as pwd_file:
                                    current_dir = pwd_file.read().strip()

                            # Enrich result and store
                            task['content'] = "\n".join(list(commands))
                            tasks_storage.appendleft(task)
                            save_data(tasks_storage.get_tasks(), TASK_LIST_FILE)

                            enriched_result = { "type": "command", "target": command, "path": current_dir}

                            if all_result.startswith("Waiting for user feedback"):
                                enriched_result['result'] = result #"Success"
                                executed_tasks_storage.appendleft(enriched_result)
                                save_data(executed_tasks_storage.get_tasks(), EXECUTED_TASK_LIST_FILE)

                                # API now supports long contexts, disable processing of the maximum value in the execution history
                                # if len(executed_tasks_storage.get_tasks()) > 30:
                                #     executed_tasks_storage.pop()

                                is_complete = True
                                break

                            if all_result.startswith("The Return Code for the command is 0:") is False:
                                enriched_result['result'] = result

                                # TODO: Verification of execution results required. 
                                # By deleting duplicate command execution results, there is no history of execution errors and the possibility of not learning about errors in the field, and because the API context has become longer,
                                # try to disable the process of removing duplicate command execution results
                                # executed_tasks_storage.remove_target_command_dicts(current_dir, command, result)

                                executed_tasks_storage.appendleft(enriched_result)
                                save_data(executed_tasks_storage.get_tasks(), EXECUTED_TASK_LIST_FILE)

                                # API now supports long contexts, disable processing of the maximum value in the execution history
                                # if len(executed_tasks_storage.get_tasks()) > 30:
                                #     executed_tasks_storage.pop()

                                is_check_result = True
                                break

                            enriched_result['result'] = result #"Success"
                            executed_tasks_storage.appendleft(enriched_result)
                            save_data(executed_tasks_storage.get_tasks(), EXECUTED_TASK_LIST_FILE)

                            # API now supports long contexts, disable processing of the maximum value in the execution history
                            # if len(executed_tasks_storage.get_tasks()) > 30:
                            #     executed_tasks_storage.pop()

                            task = tasks_storage.popleft()

                        if is_complete:
                            break
                        if is_check_result:
                            break

                        if tasks_storage.is_empty():
                            break
                        else:
                            next_task = tasks_storage.reference(0)
                            if next_task['type'].startswith("create") or next_task['type'].startswith("modify") or next_task['type'].startswith("modify_partial") or next_task['type'].startswith("command"):
                                task = tasks_storage.popleft()
                            else:
                                is_next_plan = True
                                break

                log("\033[32m\033[1m" + "*****TASK RESULT*****\n\n" + "\033[0m\033[0m")

                if is_complete:
                    break
                if is_next_plan:
                    continue
                if not "type" in enriched_result:
                    continue


                # Create new tasks and reprioritize task list
                new_tasks_list = check_completion_agent(OBJECTIVE, enriched_result, tasks_storage.get_tasks(),
                                                        executed_tasks_storage.get_tasks(), current_dir)
                    
                if isinstance(new_tasks_list, str) and new_tasks_list.startswith("Complete"):
                    break

            else:
                log("\033[33m\033[1m" + "*****PLAN TASK*****\n\n" + "\033[0m\033[0m")
                new_tasks_list = plan_agent(OBJECTIVE, task['content'], executed_tasks_storage.get_tasks(), current_dir)

                log("\033[32m\033[1m" + "*****TASK RESULT*****\n\n" + "\033[0m\033[0m")

        tasks_storage.replace(deque(new_tasks_list))
        save_data(tasks_storage.get_tasks(), TASK_LIST_FILE)
        
        time.sleep(1)

    log("\033[92m\033[1m" + "*****COMPLETE*****\n\n" + "\033[0m\033[0m")

    if input_flag != "f":
        input_flag = None
        log("\n" + "\033[33m\033[1m" + 'If the OBJECTIVE has not been achieved, please input "f". The AI will continue to execute based on the feedback.' + "\033[0m\033[0m" + "\n")
        while True:
            time.sleep(1)
            if input_flag == "f":
                break
            elif input_flag is not None:
                input_flag = None
    feedback = user_feedback()
    enriched_result = {
        "type": "feedback",
        "target": "user",
        "result": feedback
    }
    executed_tasks_storage.appendleft(enriched_result)
    save_data(executed_tasks_storage.get_tasks(), EXECUTED_TASK_LIST_FILE)

    objective_list = deque([feedback, ORIGINAL_OBJECTIVE])
    save_data(objective_list, OBJECTIVE_LIST_FILE)
    OBJECTIVE = parse_objective(objective_list)
    tasks_storage.appendleft({"type": "plan", "content": feedback})
    save_data(tasks_storage.get_tasks(), TASK_LIST_FILE)
    input_flag = None

def main():
    while True:
        execute_objective()

if __name__ == "__main__":
    main()

