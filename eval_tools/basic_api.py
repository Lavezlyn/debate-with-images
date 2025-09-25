from __future__ import annotations
import os
import logging
import time
from typing import Any, Callable, List, Union
import json
import PIL
import hashlib

# Remove global API_KEY

import ray

from tqdm import tqdm

from utils import prompt_to_conversation, prompt_to_reason_conversation

@ray.remote(num_cpus=1)
def bean_gpt_api(
    system_content: str,
    user_content: str,
    images: Union[List[PIL.Image], List[str]] = [],
    post_process: Callable = lambda x: x,
    model: str = 'gpt-4o',
    infer_cfgs: dict = None,
    api_key: str = None,
) -> Any:
    from urllib3.util.retry import Retry 
    import urllib3
    import base64
    from io import BytesIO
    import signal
    from functools import partial

    # Set up timeout handler
    def timeout_handler(signum, frame, timeout_msg="API call timed out"):
        raise TimeoutError(timeout_msg)

    # Set timeout to 60 seconds
    signal.signal(signal.SIGALRM, partial(timeout_handler, timeout_msg="API call timed out after 60 seconds"))
    signal.alarm(60)

    retry_strategy = Retry(
        total=10,  # Maximum retry count
        backoff_factor=1.0,  # Wait factor between retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to force a retry on
        allowed_methods=['POST'],  # Retry only for POST request
        raise_on_redirect=False,  # Don't raise exception
        raise_on_status=False,  # Don't raise exception
    )
    http = urllib3.PoolManager(
        retries=retry_strategy,
        timeout=urllib3.Timeout(connect=30.0, read=60.0),  # 30 seconds for connection, 60 seconds for reading
        maxsize=10,  # Maximum number of connections in the pool
        block=True,  # Block when pool is full
    )   
    
    messages = prompt_to_conversation(user_content, system_content, images)

    openai_api = 'https://api.openai.com'

    # Build API parameters with defaults and overrides from infer_cfgs
    params_gpt = {
        'model': model,
        'messages': messages,
        'temperature': 0.0,
        'top_p': 1.0,
        'max_tokens': 512,
    }
    
    # Override with infer_cfgs if provided
    if infer_cfgs:
        # Common OpenAI API parameters
        if 'temperature' in infer_cfgs:
            params_gpt['temperature'] = infer_cfgs['temperature']
        if 'max_tokens' in infer_cfgs:
            params_gpt['max_tokens'] = infer_cfgs['max_tokens']
        if 'top_p' in infer_cfgs:
            params_gpt['top_p'] = infer_cfgs['top_p']
        if 'top_k' in infer_cfgs:
            params_gpt['top_k'] = infer_cfgs['top_k']
        if 'frequency_penalty' in infer_cfgs:
            params_gpt['frequency_penalty'] = infer_cfgs['frequency_penalty']
        if 'presence_penalty' in infer_cfgs:
            params_gpt['presence_penalty'] = infer_cfgs['presence_penalty']
        if 'stop' in infer_cfgs:
            params_gpt['stop'] = infer_cfgs['stop']
        if 'stream' in infer_cfgs:
            params_gpt['stream'] = infer_cfgs['stream']
    
    url = openai_api + '/v1/chat/completions'

    # Use the provided api_key, or raise if not set
    if not api_key:
        raise ValueError("API key must be provided for API engine. Set 'api_key' in your YAML config.")
   
    headers = {
        'Content-Type': 'application/json',
        'Authorization': api_key,
        'Connection':'close',
    }

    encoded_data = json.dumps(params_gpt).encode('utf-8')
    max_try = 50
    while max_try > 0:
        try:
            response = http.request('POST', url, body=encoded_data, headers=headers)
            if response.status == 200:
                response = json.loads(response.data.decode('utf-8'))['choices'][0]['message']['content']
                logging.info(response)
                break
            else:
                err_msg = f'Access openai error, status code: {response.status} response: {response.data.decode("utf-8")}'
                logging.error(err_msg)
                time.sleep(3)
                max_try -= 1
                continue
        except TimeoutError as e:
            logging.error(f"Timeout error: {str(e)}")
            return "API call timed out"
        except Exception as e:
            logging.error(f"Error in API call: {str(e)}")
            time.sleep(3)
            max_try -= 1
            continue
    else:
        print('Bean Proxy API Failed...')
        response = 'Bean Proxy API Failed...'

    # Disable the alarm
    signal.alarm(0)
    return post_process(response)

def generate_hash_uid(to_hash: dict | tuple | list | str):
    """Generates a unique hash for a given model and arguments."""
    # Convert the dictionary to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()

    return hash_uid

def api(
    system_contents: list[str],
    user_contents: list[str],
    image_urls: list[PIL.Image.Image] | None = None,
    num_workers: int = 50,
    post_process: Callable = lambda x: x,
    hash_checker: Callable = lambda x: True,
    cache_dir: str = './cache',
    model: str = 'gpt-4o',
    use_cache: bool = True,
    infer_cfgs: dict = None,
    api_key: str = None,
):
    """API"""
    if len(system_contents) != len(user_contents):
        raise ValueError('Length of system_contents and user_contents should be equal.')
    server = bean_gpt_api

    api_interaction_count = 0
    ray.init()
    
    if image_urls is None:
        contents = list(enumerate(zip(system_contents, user_contents)))
        uids = [generate_hash_uid((system_content, user_content)) for system_content, user_content in zip(system_contents, user_contents)]
    else:
        contents = list(enumerate(zip(system_contents, user_contents, image_urls)))
        uids = [generate_hash_uid((system_content, user_content, tuple(images) if isinstance(images, list) else images)) for system_content, user_content, images in zip(system_contents, user_contents, image_urls)]
    bar = tqdm(total=len(system_contents))
    results = [None] * len(system_contents)
    not_finished = []
    while True:

        if len(not_finished) == 0 and len(contents) == 0:
            break

        while len(not_finished) < num_workers and len(contents) > 0:
            index, content = contents.pop()
            uid = uids[index]
            cache_path = os.path.join(cache_dir, f'{uid}.json')
            if use_cache and os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    try:
                        result = json.load(f)
                    except:
                        os.remove(cache_path)
                        continue
                results[index] = result
                bar.update(1)
                continue

            if image_urls is not None:
                future = server.remote(content[0], content[1], content[2], post_process, model, infer_cfgs, api_key)
            else:
                future = server.remote(content[0], content[1], post_process, model, infer_cfgs, api_key)
            not_finished.append([index, future])
            api_interaction_count += 1

        if len(not_finished) == 0:
            continue

        indices, futures = zip(*not_finished)
        finished, not_finished_futures = ray.wait(list(futures), timeout=1.0)
        finished_indices = [indices[futures.index(task)] for task in finished]

        for i, task in enumerate(finished):
            results[finished_indices[i]] = ray.get(task)
            uid = uids[finished_indices[i]]
            cache_path = os.path.join(cache_dir, f'{uid}.json')
            if use_cache:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(results[finished_indices[i]], f, ensure_ascii=False, indent=4)

        not_finished = [(index, future) for index, future in not_finished if future not in finished]
        bar.update(len(finished))
    bar.close()

    assert all(result is not None for result in results)

    ray.shutdown()
    print(f'API interaction count: {api_interaction_count}')

    return results
