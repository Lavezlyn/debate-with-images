from eval_tools.basic_api import api   
from eval_tools.utils import encode_image_to_base64

from datasets import load_dataset
import json
from tqdm import tqdm
import os

user_format = """
{question}

Answer yes or no
Please think step by step before answering.
output format:

<think>...</think>
<output>yes | no</output> 
"""

answer_mapping = {
    "0":"no",
    "1": "yes"
}

def extract_answer(text):
    import re
    pattern = r'<output>(.*?)</output>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else ""


def prepare_inputs(ds):
    input_list = []
    for item in tqdm(ds, desc="processing inputs"):
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(os.path.join('./HallusionBench/data', item['filename']), auto_compress=False)}"}
                    },
                    {
                        "type": "text",
                        "text": user_format.format(question=item['question'])
                    }
                ]
            }
        ]
        input_list.append(message)
    return input_list

def evaluate_answer(ds, output):
    count = 0
    for ds_item, output_item in tqdm(zip(ds, output),desc="evaluating"):
        if ds_item['answer'].lower() == output_item.strip('.').lower():
            count += 1
    print("acc", {count/len(output)})

def post_process(ds, output):
    result = []
    for ds_item, output in tqdm(zip(ds, output), desc="Post-processing"):
        result_item = ds_item.copy()
        result_item["model_prediction"] = output
        result_item['is_correct'] = answer_mapping[ds_item['gt_answer']].lower() == extract_answer(output).strip('.').lower()
        result.append(result_item)
    return result


def main():
    config = {
        "model": "gpt-4o",
        "number_workers": 50,
        "use_cache": True,
        "cache_dir": "./cache",
        "api_key": "sk-xxx",
        "infer_cfgs": {
            "temperature": 1.0,
            'top_p': 0.9,
            "max_tokens": 512
        },
    }
    ds_path="./HallusionBench/HallusionBench.json"
    with open(ds_path, 'r', encoding='utf-8') as f:
        ds = json.load(f)
    ds_VD = [item for item in ds if item['category']=='VD']
    input_list = prepare_inputs(ds_VD)
    output = api(input_list, config)
    result = post_process(ds_VD, output)
    with open("./results/hallusionbench.json", 'w', encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()

