"""
API Utils
"""

import os
import logging
import time
from typing import Any, Callable, List, Union, Dict, Tuple
import json
import hashlib

import ray

from tqdm import tqdm

# Additional imports for async functionality
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not available, falling back to synchronous concurrent processing")

@ray.remote(num_cpus=1)
def bean_gpt_api(
    messages: List[Dict[str, Any]],
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

    openai_api = 'https://api.openai.com'

    # Build API parameters with defaults and overrides from infer_cfgs
    params_gpt = {
        'model': model,
        'messages': messages,
        'temperature': 0.0,  # Default temperature
        'max_tokens': 512,   # Default max_tokens
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
        print('API Failed...')
        response = 'API Failed...'

    # Disable the alarm
    signal.alarm(0)
    return response

def generate_hash_uid(to_hash: dict | tuple | list | str):
    """Generates a unique hash for a given model and arguments."""
    # Convert the dictionary to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()

    return hash_uid

def api(
    messages_list: List[List[Dict[str, Any]]],
    config: dict
):
    """API"""
    server = bean_gpt_api

    num_workers = config['num_workers']
    cache_dir = config['cache_dir']
    model = config['model']
    use_cache = config['use_cache']
    infer_cfgs = config['infer_cfgs']
    api_key = config['api_key']

    api_interaction_count = 0
    ray.init()
    
    contents = list(enumerate(messages_list))
    bar = tqdm(total=len(messages_list))
    results = [None] * len(messages_list)
    uids = [generate_hash_uid(messages) for messages in messages_list]
    not_finished = []
    while True:

        if len(not_finished) == 0 and len(contents) == 0:
            break

        while len(not_finished) < num_workers and len(contents) > 0:
            index, messages = contents.pop()
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

            future = server.remote(messages, model, infer_cfgs, api_key)
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

async def vllm_api_async(messages_list, config):
    """
    Async version for VLLM API with proper concurrent processing.
    """
    import asyncio
    import aiohttp
    import time
    
    openai_api_base = "http://localhost:8000/v1" # default vllm server port
    model = config['model']
    infer_cfgs = config['infer_cfgs']
    max_concurrent_requests = config.get('max_concurrent_requests', 64)
    request_timeout = config.get('request_timeout', 300)
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def make_single_request(session, messages, index):
        """Make a single async request to vLLM server"""
        async with semaphore:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": infer_cfgs.get('temperature', 0.0),
                "max_tokens": infer_cfgs.get('max_tokens', 512),
                "top_p": infer_cfgs.get('top_p', 1.0),
            }
            
            try:
                async with session.post(
                    f"{openai_api_base}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=request_timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return index, result['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        logging.error(f"Request {index} failed with status {response.status}: {error_text}")
                        return index, f"Error: HTTP {response.status}"
            except asyncio.TimeoutError:
                logging.error(f"Request {index} timed out")
                return index, "Error: Request timed out"
            except Exception as e:
                logging.error(f"Request {index} failed: {str(e)}")
                return index, f"Error: {str(e)}"
    
    # Create session and run all requests concurrently
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            make_single_request(session, messages, idx)
            for idx, messages in enumerate(messages_list)
        ]
        
        # Execute with progress tracking
        results = [None] * len(messages_list)
        completed = 0
        
        with tqdm(total=len(messages_list), desc="Processing VLLM Requests (Async)") as pbar:
            for coro in asyncio.as_completed(tasks):
                index, result = await coro
                results[index] = result
                completed += 1
                pbar.update(1)
                
                # Log progress
                if completed % max(1, len(messages_list) // 10) == 0:
                    logging.info(f"VLLM API async progress: {completed}/{len(messages_list)} completed")
    
    return results

def vllm_api(messages_list, config):
    """
    Support concurrent batch inference with vLLM using both sync and async approaches.
    
    Args:
        messages_list: List of messages lists for batch processing
        config: Configuration dictionary containing model and inference settings
    
    Returns:
        List of responses corresponding to each input
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    
    # Determine which approach to use based on config and availability
    use_async = config.get('use_async', True) and AIOHTTP_AVAILABLE
    
    if use_async:
        # Use async approach for better performance
        try:
            return asyncio.run(vllm_api_async(messages_list, config))
        except Exception as e:
            logging.warning(f"Async method failed, falling back to sync method: {str(e)}")
            use_async = False
    
    if not use_async:
        # Fallback to sync approach with ThreadPoolExecutor
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"

        model = config['model']
        infer_cfgs = config['infer_cfgs']
        
        # Get concurrency settings from config, with sensible defaults
        max_concurrent_requests = config.get('max_concurrent_requests', 64)
        request_timeout = config.get('request_timeout', 300)
        
        def create_single_request(messages):
            """Create a single OpenAI API request"""
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
                timeout=request_timeout
            )
            
            try:
                chat_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=infer_cfgs.get('temperature', 0.0),
                    max_tokens=infer_cfgs.get('max_tokens', 512),
                    top_p=infer_cfgs.get('top_p', 1.0),
                )
                return chat_response.choices[0].message.content
            except Exception as e:
                logging.error(f"Error in VLLM API request: {str(e)}")
                return f"Error: {str(e)}"

        # Use ThreadPoolExecutor for concurrent requests
        results = [None] * len(messages_list)
        
        with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            # Submit all requests
            future_to_index = {
                executor.submit(create_single_request, messages): idx 
                for idx, messages in enumerate(messages_list)
            }
            
            # Collect results with progress bar
            completed_count = 0
            with tqdm(total=len(messages_list), desc="Processing VLLM Requests (Sync)") as pbar:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        logging.error(f"Request {idx} failed: {str(e)}")
                        results[idx] = f"Error: {str(e)}"
                    
                    completed_count += 1
                    pbar.update(1)
                    
                    # Log progress every 10% completion
                    if completed_count % max(1, len(messages_list) // 10) == 0:
                        logging.info(f"VLLM API sync progress: {completed_count}/{len(messages_list)} completed")

        return results
