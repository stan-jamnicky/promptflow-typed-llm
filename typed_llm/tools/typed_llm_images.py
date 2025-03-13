from typing import List, Dict
import importlib.util
import sys
from pathlib import Path
from promptflow.core import tool

from promptflow.core._prompty_utils import build_messages
from promptflow.tools.common import handle_openai_error, \
    preprocess_template_string, find_referenced_image_set, convert_to_chat_list, init_azure_openai_client, \
    post_process_chat_api_response, list_deployment_connections, build_deployment_dict \

from promptflow._internal import ToolProvider, tool
from promptflow.connections import AzureOpenAIConnection
from promptflow.contracts.types import PromptTemplate, FilePath

API_VERSION = "2024-08-01-preview"

def _import_module(module_path: str):
    module_name = Path(module_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def list_deployment_names(
    subscription_id=None,
    resource_group_name=None,
    workspace_name=None,
    connection=""
) -> List[Dict[str, str]]:
    res = []
    deployment_collection = list_deployment_connections(subscription_id, resource_group_name, workspace_name,
                                                        connection)
    if not deployment_collection:
        return res

    for item in deployment_collection:
        deployment = build_deployment_dict(item)
        if deployment.version == API_VERSION:
            cur_item = {
                "value": deployment.name,
                "display_value": deployment.name,
            }
            res.append(cur_item)

    return res

@tool(streaming_option_parameter="stream")
@handle_openai_error()
def typed_llm_images(
    connection: AzureOpenAIConnection,
    prompt: PromptTemplate,
    response_type: str,
    module_path: FilePath,
    deployment_name: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    # stream is a hidden to the end user, it is only supposed to be set by the executor.
    stream: bool = False,
    stop: list = None,
    max_tokens: int = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    seed: int = None,
    detail: str = 'auto',
    **kwargs,
) -> str:
    client = init_azure_openai_client(connection)
    prompt = preprocess_template_string(prompt)
    referenced_images = find_referenced_image_set(kwargs)

    # convert list type into ChatInputList type
    converted_kwargs = convert_to_chat_list(kwargs)
    messages = build_messages(prompt=prompt, images=list(referenced_images), detail=detail, **converted_kwargs)

    headers = {
        "Content-Type": "application/json",
        "ms-azure-ai-promptflow-called-from": "aoai-gpt4v-tool"
    }

    module = _import_module(module_path)
    if response_type not in module.__dict__:
        raise ValueError(f"response_type {response_type} not found in {module_path}")
    response_format = module.__dict__[response_type]

    params = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
        "stream": stream,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "extra_headers": headers,
        "model": deployment_name,
        "response_format": response_format,
    }

    if stop:
        params["stop"] = stop
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if seed is not None:
        params["seed"] = seed

    completion = client.chat.completions.create(**params)
    return post_process_chat_api_response(completion, stream)