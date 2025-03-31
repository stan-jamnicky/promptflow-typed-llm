import asyncio
from typing import List, Dict
import importlib.util
import sys
from pathlib import Path
from promptflow.core import tool

from promptflow.core._prompty_utils import build_messages
from promptflow.tools.common import handle_openai_error, \
    preprocess_template_string, find_referenced_image_set, convert_to_chat_list, \
    list_deployment_connections, build_deployment_dict \

from openai import AsyncAzureOpenAI
from promptflow._internal import tool
from promptflow.contracts.types import PromptTemplate, FilePath

# Has to be hardcoded because only the new API supports structured JSON API
API_VERSION = "2024-11-20"

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
    
async def stream_responses(response):
    """Generator to yield streaming responses."""
    for chunk in response.choices:
        if chunk and hasattr(chunk.message, 'content') and chunk.message.content is not None:
            yield chunk.message.content  # Yield the content for streaming

@tool
@handle_openai_error()
async def typed_llm_images(
    connection: AsyncAzureOpenAI,
    prompt: PromptTemplate,
    response_type: str,
    module_path: FilePath,
    deployment_name: str,
    n: int = 1,
    stream: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop: list = None,
    max_tokens: int = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    seed: int = None,
    detail: str = 'auto',
    **kwargs,
) -> list[str]:
    prompt = preprocess_template_string(prompt)
    referenced_images = find_referenced_image_set(kwargs)

    # convert list type into ChatInputList type
    converted_kwargs = convert_to_chat_list(kwargs)
    
    if referenced_images:
        messages = build_messages(prompt=prompt, images=list(referenced_images), detail=detail, **converted_kwargs)
    else:
        messages = build_messages(prompt=prompt)

    headers = {
        "Content-Type": "application/json",
        "ms-azure-ai-promptflow-called-from": "typed_llm_images"
    }

    module = _import_module(module_path)
    if response_type not in module.__dict__:
        raise ValueError(f"response_type {response_type} not found in {module_path}")
    response_format = module.__dict__[response_type]

    params = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "extra_headers": headers,
        "model": deployment_name,
        "response_format": response_format
    }

    if stop:
        params["stop"] = stop
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if seed is not None:
        params["seed"] = seed

    if connection.api_key:
        client = AsyncAzureOpenAI(api_key=connection.api_key, azure_endpoint=connection.api_base, api_version=API_VERSION)
    else:
        client = AsyncAzureOpenAI(azure_ad_token_provider=connection.get_token, azure_endpoint=connection.api_base, api_version=API_VERSION)
    
    response = await client.beta.chat.completions.parse(**params)

    if stream:
        return stream_responses(response)
    else:
        # Return all choices' content
        return [choice.message.content for choice in response.choices if hasattr(choice.message, 'content') and choice.message.content is not None]