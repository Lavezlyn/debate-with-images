"""
Evaluate models' responses to MM-Deceptionbench by MLLM-as-a-judge
"""

from eval_tools.basic_api import api
from eval_tools.vllm_infer import vllm_api
from eval_tools.utils import encode_image_to_base64
from prompt import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT
import os, re, json

def define_image_path(image_path: str) -> str:
    if image_path.startswith('http'):
        return image_path
    else:
        return os.path.join('./mm-deceptionbench', image_path)

def build_eval_payload(ds:list[dict], system_prompt, user_prompt_format):
    system_prompts = [system_prompt] * len(ds)
    user_prompts = [user_prompt_format.format(
        scenario=item['scenario'],
        assistant_profile=item['assistant_profile'],
        user_profile=item['user_profile'],
        user_prompt=item['prompt'],
        assistant_reasoning=item['result']['reasoning'],
        assistant_output=item['result']['output']
    ) for item in ds]
    images_list = []
    for item in ds:
        item_images = []
        images = item['images']
        for image_path in images:
            image_path = define_image_path(image_path)
            item_images.append(image_path)
        images_list.append(item_images)
    return system_prompts, user_prompts, images_list

def build_message_list(system_prompts, user_prompts, images_list):
    message_list = []
    for system_prompt, user_prompt, image_list in zip(system_prompts, user_prompts, images_list):
        user_content = [
            {"type": "text", "text": user_prompt},
        ]
        for image in image_list:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image;base64,{encode_image_to_base64(image, auto_compress=False)}"}})
        message = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content
            }
        ]
        message_list.append(message)
    return message_list

def evaluate(infer_api:callable, message_list, config:dict):
    responses = infer_api(message_list, config)
    return responses

def post_process_eval_results(responses:list[str], ds:list[dict]):
    eval_results = []
    for response, item in zip(responses, ds):
        # First try to find JSON in code blocks
        match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if match:
            structured_response = match.group(1).strip()
            try:
                structured_response = json.loads(structured_response)
            except json.JSONDecodeError:
                structured_response = {}
        else:
            # If no code blocks, try to parse the entire response as JSON
            try:
                structured_response = json.loads(response)
            except json.JSONDecodeError:
                # If that fails, try to find JSON-like content with curly braces
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                    try:
                        structured_response = json.loads(json_content)
                    except json.JSONDecodeError:
                        structured_response = response
                else:
                    structured_response = response
        
        eval_results.append(
            {
                **item, 
                "evaluation": structured_response
            }
        )
    return eval_results

def run_evaluation(ds:list[dict], system_prompt: str, user_prompt_format: str, infer_api:callable, config:dict):
    system_prompts, user_prompts, images = build_eval_payload(ds, system_prompt, user_prompt_format)
    message_list = build_message_list(system_prompts, user_prompts, images)
    responses = evaluate(infer_api, message_list, config)
    eval_results = post_process_eval_results(responses, ds)
    return eval_results

def main():
    ds_path = "./data/human_labeled_data.json"
    output_dir = "./results/human_agreement"

    import os
    os.makedirs(output_dir, exist_ok=True)

    with open(ds_path, 'r', encoding='utf-8') as f:
        ds = json.load(f)
    config = {
        "model": "Qwen2.5-VL-72B-Instruct",
        "backend": "vllm",
        "use_async": True,
        "max_concurrent_requests": 64,
        "infer_cfgs": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 512
        }
    }
    eval_results = run_evaluation(ds, JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT, vllm_api, config)
    ds_name = ds_path.split('/')[-1].replace('.json', '')
    os.makedirs(output_dir, exist_ok=True) 
    with open(os.path.join(output_dir, f'{ds_name}_eval_by_qwen72b.json'), 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()