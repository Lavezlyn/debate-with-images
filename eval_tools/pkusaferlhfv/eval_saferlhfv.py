"""
Evaluate PKU-SafeRLHF-V preference judgement by MLLM-as-a-judge
"""

from eval_tools.vllm_infer import vllm_api
from eval_tools.utils import encode_image_to_base64

from prompt import JUDGE_PROMPT, JUDGE_COT_PROMPT, definition_mapping
import os, re, json
from typing import List, Dict, Any

def load_data(dataset_path: str) -> List[Dict[str, Any]]:
    """Load results data from JSON file"""
   
    data = []
    from datasets import load_dataset
    for category in definition_mapping.keys():
    ds = load_dataset(dataset_path, name=category, split="evaluation")
    for item in ds:
        data.append({
            "question": item['question'],
            "image": item['image'],
            "category": item['category'],
            "response_1": item['response_1'],
            "response_2": item['response_2'],
            "safer_response_id": item['safer_response_id'],
        })
    print(f"Loaded {len(data)} samples from {dataset_path}")
    return data

def build_eval_payload(ds:list[dict], system_prompt: str, user_prompt: str):
    system_prompts = [system_prompt] * len(ds)
    user_prompts = [user_prompt.format(
        category=item['category'],
        definition=definition_mapping[item['category']],
        question=item['question'],
        response_1=item['response_1'],
        response_2=item['response_2']
    ) for item in ds]
    images_list = []
    for item in ds:
        image = item['image']
        images_list.append([image])
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
    ds_path = "PKU-Alignment/PKU-SafeRLHF-V"
    output_dir = "./results/saferlhfv"

    import os
    os.makedirs(output_dir, exist_ok=True)

    ds = load_data(ds_path)
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
    eval_results = run_evaluation(ds, system_prompt="", user_prompt=JUDGE_PROMPT, infer_api=vllm_api, config=config)
    ds_name = ds_path.split('/')[-1]
    os.makedirs(output_dir, exist_ok=True) 
    with open(os.path.join(output_dir, f'{ds_name}_eval_by_qwen72b.json'), 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()