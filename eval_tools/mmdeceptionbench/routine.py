"""
Routine for inferring models' responses to MM-Deceptionbench
"""
import json
import os
from tqdm import tqdm
import argparse
import glob
from pathlib import Path
from basic_api import api
from openai_reasoning_api import think_api
from anthropic_reasoning_api import claude_think_api
from vllm_infer import vllm_api
from utils import parse_yaml_config, post_process_reason_response, post_process_claude_reason_response, post_process_response
from prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_REASON, USER_PROMPT
import re

def define_image_path(image_path: str) -> str:
    if image_path.startswith('http'):
        return image_path
    else:
        return os.path.join('./mm_deceptionbench', image_path)


def prepare_dataset(ds: list[dict], is_reason: bool = False) -> list[dict]:
    system_prompts = []
    user_prompts = []
    images_list = []
    
    for item in tqdm(ds, desc="Preparing dataset"):
        if is_reason:
            system_prompt = SYSTEM_PROMPT_REASON.format(
                assistant_profile=item['assistant_profile']
            )
            user_prompt = USER_PROMPT.format(
                scenario=item['scenario'],
                user_profile=item['user_profile'],
                prompt=item['prompt']
            )
        else:
            system_prompt = SYSTEM_PROMPT.format(
                assistant_profile=item['assistant_profile']
            )
            user_prompt = USER_PROMPT.format(
                scenario=item['scenario'],
                user_profile=item['user_profile'],
                prompt=item['prompt']
            )
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt)
        
        item_images = []
        for image in item['images']:
            image_path = define_image_path(image)
            item_images.append(image_path)
        images_list.append(item_images)
        
    return system_prompts, user_prompts, images_list
        
def generate_responses(ds: list[dict], engine: str, model_name: str, infer_cfgs: dict, model_cfgs: dict, use_cache: bool, api_key: str = None) -> list[dict]:
    
    if engine == 'vllm':
        system_prompts, user_prompts, images = prepare_dataset(ds, is_reason=False)
        results = vllm_api(system_prompts, user_prompts, images, model_cfgs, infer_cfgs)
    elif engine == 'api':
        system_prompts, user_prompts, images = prepare_dataset(ds, is_reason=False)
        cache_dir = f'./cache/{model_name}'
        api_model = model_cfgs.get('model', model_name)
        results = api(system_prompts, user_prompts, images, num_workers=50, 
                     cache_dir=cache_dir, model=api_model, use_cache=use_cache, infer_cfgs=infer_cfgs, api_key=api_key)
    elif engine == 'reason_api':
        system_prompts, user_prompts, images = prepare_dataset(ds, is_reason=True)
        cache_dir = f'./cache/{model_name}'
        api_model = model_cfgs.get('model', model_name)
        results = think_api(system_prompts, user_prompts, images, num_workers=50, 
                     cache_dir=cache_dir, model=api_model, use_cache=use_cache, infer_cfgs=infer_cfgs, api_key=api_key)
    elif engine == 'claude_reason_api':
        system_prompts, user_prompts, images = prepare_dataset(ds, is_reason=True)
        cache_dir = f'./cache/{model_name}'
        api_model = model_cfgs.get('model', model_name)
        print("system_prompts_item_1", system_prompts[0])
        print("cache_dir", cache_dir)
        print("api_model", api_model)
        results = claude_think_api(system_prompts, user_prompts, images, num_workers=50, 
                     cache_dir=cache_dir, model=api_model, use_cache=use_cache, infer_cfgs=infer_cfgs, api_key=api_key)
    else:
        raise ValueError(f"Unsupported engine: {engine}")
    
    final_results = []
    for result, item in tqdm(zip(results, ds), total=len(ds), desc="Post-processing"):
        new_item = item.copy()
        if engine == 'reason_api':
            new_item['result'] = post_process_reason_response(result)
        elif engine == 'claude_reason_api':
            new_item['result'] = post_process_claude_reason_response(result)
        else:
            new_item['result'] = post_process_response(result)

        final_results.append(new_item)
    return final_results

def process_single_dataset(dataset_path: str, engine: str, model_name: str, infer_cfgs: dict, model_cfgs: dict, use_cache: bool, output_dir: str, api_key: str = None) -> str:
    """
    Process a single dataset file and return the output path
    """
    print(f"\n=== Processing dataset: {dataset_path} ===")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        ds = json.load(f)

    final_results = generate_responses(ds, engine, model_name, infer_cfgs, model_cfgs, use_cache, api_key=api_key)
    
    # Generate output filename
    import time
    ds_name = Path(dataset_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"responses_{ds_name}_{timestamp}.json")
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to: {save_path}")
    return save_path


def get_dataset_files(input_path: str) -> list[str]:
    """
    Get list of dataset files from input path
    If input_path is a file, return [input_path]
    If input_path is a directory, return all .json files in it
    """
    path = Path(input_path)
    
    if path.is_file():
        if path.suffix.lower() == '.json':
            return [str(path)]
        else:
            raise ValueError(f"Input file must be a JSON file: {input_path}")
    elif path.is_dir():
        json_files = list(path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in directory: {input_path}")
        return [str(f) for f in sorted(json_files)]
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Process dataset(s) for AI model evaluation")
    parser.add_argument("--dataset", type=str, 
                       default="./mm_deceptionbench/dataset/sycophancy.json",
                       help="Path to dataset file or directory containing dataset files")
    parser.add_argument("--output_dir", type=str, 
                       default="./results", 
                       help="Base output directory (model name will be appended)")
    parser.add_argument("--config", type=str, default=None, required=True,
                       help="Path to YAML configuration file")
    args = parser.parse_args()

    # Parse configuration
    engine, model_name, infer_cfgs, model_cfgs, use_cache, api_key = parse_yaml_config(args.config)
    
    # Set up output directory
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of dataset files to process
    dataset_files = get_dataset_files(args.dataset)
    
    print(f"Found {len(dataset_files)} dataset file(s) to process:")
    for i, file_path in enumerate(dataset_files, 1):
        print(f"  {i}. {file_path}")
    
    # Process each dataset file
    output_files = []
    for dataset_file in dataset_files:
        try:
            output_path = process_single_dataset(
                dataset_file, engine, model_name, infer_cfgs, 
                model_cfgs, use_cache, output_dir, api_key
            )
            output_files.append(output_path)
        except Exception as e:
            print(f"Error processing {dataset_file}: {str(e)}")
            continue
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed {len(output_files)} out of {len(dataset_files)} datasets")
    print("Output files:")
    for output_file in output_files:
        print(f"  - {output_file}")
    
if __name__ == "__main__":
    main()




