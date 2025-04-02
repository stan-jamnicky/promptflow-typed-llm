import asyncio
from typing import List, Dict
import importlib.util
import sys
from pathlib import Path
from promptflow.core import tool
import asyncio
from promptflow.core._prompty_utils import build_messages
from promptflow.tools.common import handle_openai_error, \
    preprocess_template_string, find_referenced_image_set, convert_to_chat_list, \
    render_jinja_template, parse_chat, list_deployment_connections, build_deployment_dict \

from openai import AsyncAzureOpenAI
from promptflow._internal import tool
from promptflow.contracts.types import PromptTemplate, FilePath

# Has to be hardcoded because only the new API supports structured JSON API
API_VERSION = "2025-01-01-preview"
MAX_CONCURRENT_REQUESTS = 4

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

@handle_openai_error()
async def _async_do_openai_request(
    client: AsyncAzureOpenAI, 
    semaphore: asyncio.Semaphore,
    **kwargs):

    async with semaphore:
        while True:
            completion = await client.beta.chat.completions.parse(**kwargs)

            if not completion.choices:
                return []

            contents = [
                choice.message.content
                for choice in completion.choices
                if not choice.message.refusal and getattr(choice.message, "content", None)
            ]

            refusals = [
                choice.message.refusal
                for choice in completion.choices
                if choice.message.refusal
            ]

            if refusals:
                print(f"Completion refused: {', '.join(refusals)}. Retrying...")
                continue  # Retry the request if there are refusals

            return contents
    
@tool
def typed_llm_images(
    connection: AsyncAzureOpenAI,
    prompt: PromptTemplate,
    response_type: str,
    module_path: FilePath,
    deployment_name: str,
    n: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    detail: str = 'auto',
    number_of_requests: int = 1,
    **kwargs,
) -> list[str]:
    referenced_images = find_referenced_image_set(kwargs)
    prompt = preprocess_template_string(prompt)

    # convert list type into ChatInputList type
    converted_kwargs = convert_to_chat_list(kwargs)
    
    if referenced_images:
        messages = build_messages(prompt=prompt, images=list(referenced_images), detail=detail, **converted_kwargs)
    else:
        # Text based prompt renderer
        chat_str = render_jinja_template(prompt, **converted_kwargs)
        messages = parse_chat(chat_str)

    module = _import_module(module_path)
    if response_type not in module.__dict__:
        raise ValueError(f"response_type {response_type} not found in {module_path}")
    response_format = module.__dict__[response_type]

    if deployment_name == 'o3-mini':
        params = {
            "model": deployment_name,
            "messages": messages,
            "response_format": response_format,
            "n": n,
        }
    else:
        params = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "model": deployment_name,
            "response_format": response_format,
        }
    
    async def do_requests():
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        if connection.api_key:
            client = AsyncAzureOpenAI(api_key=connection.api_key, azure_endpoint=connection.api_base, api_version=API_VERSION)
        else:
            client = AsyncAzureOpenAI(azure_ad_token_provider=connection.get_token, azure_endpoint=connection.api_base, api_version=API_VERSION)

        tasks = (_async_do_openai_request(client, semaphore, **params) for _ in range(number_of_requests))
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If the event loop is already running (e.g., in a Jupyter Notebook or other async context)
        return asyncio.run(do_requests())
    else:
        return loop.run_until_complete(do_requests())