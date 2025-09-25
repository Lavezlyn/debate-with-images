"""
Formatter
"""
# Standard library imports
import json
import os
import re
from typing import List, Dict, Any

# Local imports
from visual_utils import apply_visual_ops, encode_image_to_base64
from prompt import (
    global_user_prompt, judge_prompt, system_prompt
    aff_opening_prompt, neg_opening_prompt, aff_rebuttal_prompt, neg_rebuttal_prompt, debater_prompt,
    saferlhfv_judge_prompt, saferlhfv_debater_prompt, saferlhfv_system_prompt, saferlhfv_user_prompt,
    hallu_system_prompt, hallu_user_prompt, hallu_debater_prompt, hallu_judge_prompt
)

def load_debate_ds(
    dataset_name: str
) -> List[Dict[str, Any]]:
    if dataset_name.endswith(".json"):
        with open(dataset_name, "r", encoding="utf-8") as f:
            ds = json.load(f)
    else:
        print("Warning: dataset_name is not a json file")
        return []
    return ds

def build_inference_payload(
    role_to_generate: str,
    canonical_history: List[Dict[str, str]],
    initial_user_prompt: List[Dict[str, Any]],
    image_info: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Dynamically builds the inference payload for a specific turn in a multi-role debate.
    Now optimized to cache plot and base64 encoding results in the history.
    Supports multiple visual outputs from apply_visual_ops (e.g., annotated images, zoom compositions, depth maps).

    Args:
        role_to_generate: The role for which to generate a response ('D_aff' or 'D_neg').
        canonical_history: A clean, chronological log of the debate.
            Format: [{'role': 'role_name', 'content': '...', 'visual_evidence': {...}, 'cached_visual_content': [...]}, ...]
        initial_user_prompt: The first user prompt, typically containing the
            multimodal content and initial instructions.
            Format: {'role': 'user', 'content': [...]}
        image_info: Image information containing the source image for visual processing.

    Returns:
        A complete 'messages' list ready to be sent to a chat completions API.
        Each turn may contain multiple visual content items if apply_visual_ops returns multiple images.

    Raises:
        ValueError: If `role_to_generate` is not a valid debaterole.
    
    Note:
        The function now handles multiple images returned by apply_visual_ops, including:
        - Annotated images with bounding boxes/points
        - Zoom composition images
        - Depth map visualizations
        Each image is converted to base64 and included as separate visual content items.
    """
    # 1. Initialize the payload with the constant system and initial user prompts.
    payload = [{"role": "system", "content": system_prompt}] + initial_user_prompt

    # 2. Iterate through the canonical history, adding each turn as an 'assistant' message.
    #    All previous debate turns are framed from the model's perspective as 'assistant' replies.

    for turn in canonical_history:
        # Check if visual content is already cached
        if 'cached_visual_content' in turn and turn['cached_visual_content'] is not None:
            visual_content = turn['cached_visual_content']
        else:
            # Generate visual content if not already present using the helper function
            visual_content = _generate_visual_content(image_info, turn['visual_evidence'])
            
            # Cache the visual content for future use (modify the original history)
            turn['cached_visual_content'] = visual_content
        payload.append(
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": turn['role'] + ": " + turn['content']
                    },
                    *visual_content
                ]
            }
        )

    # 3. Dynamically create the turn-specific instruction for the current role.
    instruction_text = ""
    if role_to_generate == 'judge':
        instruction_text = judge_prompt
    elif "aff" in role_to_generate:
        if not canonical_history:
            instruction_text = aff_opening_prompt
        else:
            instruction_text = aff_rebuttal_prompt
    elif "neg" in role_to_generate:
        if not canonical_history:
            instruction_text = neg_opening_prompt
        else:
            instruction_text = neg_rebuttal_prompt
    else:
        instruction_text = "You are " + role_to_generate + ". " + text_free_debater_prompt

    # 4. Append the instruction as the final 'user' message to guide the model's next response.
    instruction_message = {"role": "user", "content": instruction_text}
    payload.append(instruction_message)

    return payload


def _generate_visual_content(image_info: Dict[str, Any], visual_evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate visual content from visual evidence data.
    
    Args:
        image_info: Dictionary containing image information
        visual_evidence: Visual evidence data for apply_visual_ops
        
    Returns:
        List of visual content dictionaries for API payload
    """
    if visual_evidence is None or len(visual_evidence) == 0:
        return []
    
    try:
        # apply_visual_ops returns a list of PIL Images
        visual_evidence_images = apply_visual_ops(image_info['image'], visual_evidence)
        visual_content = []
        
        # Handle both single image and list of images for backward compatibility
        if not isinstance(visual_evidence_images, list):
            visual_evidence_images = [visual_evidence_images]
        
        # Convert each image in the result list to base64 format
        for i, image in enumerate(visual_evidence_images):
            if image is not None:  # Safety check for None images
                try:
                    visual_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(image, auto_compress=False)}"}
                    })
                except Exception as e:
                    print(f"Warning: Failed to encode image {i} in visual evidence: {e}")
                    continue
                    
        return visual_content
        
    except Exception as e:
        print(f"Warning: Failed to generate visual evidence: {e}")
        return []

def clear_visual_cache(canonical_history: List[Dict[str, Any]]) -> None:
    """
    Clear cached visual content from canonical history to free memory.
    Completely removes the 'cached_visual_content' field from each turn.
    
    Args:
        canonical_history: The debate history with cached visual content
    """
    for turn in canonical_history:
        if 'cached_visual_content' in turn:
            del turn['cached_visual_content']


def build_init_user_prompt(
    item: dict
) -> Dict[str, Any]:
    """
    Build the initial user prompt for the debate.
    """
    user_text = global_user_prompt.format(
            scenario=item['scenario'],
            assistant_profile=item['assistant_profile'],
            user_profile=item['user_profile'],
            user_prompt=item['prompt'],
            assistant_thinking=item['result']['reasoning'],
            assistant_output=item['result']['output'],
    )

    content_parts = [
        {"type": "text", "text": user_text}
    ]

    if item['images']:
        for image in item['images']:
            image = os.path.join('./mm-deceptionbench', image)
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(image, auto_compress=False)}"}})

    return [{"role": "user", "content": content_parts}]

def format_history_item(text: str, role: str) -> Dict[str, Any]:
    """
    Format the history item.
    If <speech> tags are not found, speech content will be empty string.
    If JSON code block is not found or invalid, visual_evidence will be empty dict.
    """
    
    # 1. extract debate contents enclosed in <speech>...</speech>
    speech_match = re.search(r'<speech>(.*?)</speech>', text, re.DOTALL)
    if speech_match:
        speech_content = speech_match.group(1).strip()
    else:
        speech_content = text
    
    # 2. extract the visual evidence from the entire text (not just speech)
    visual_evidence_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if visual_evidence_match:
        visual_evidence_str = visual_evidence_match.group(1).strip()
        try:
            visual_evidence = json.loads(visual_evidence_str)
        except json.JSONDecodeError:
            visual_evidence = {}
    else:
        visual_evidence = {}
    
    # 3. format the history item dict
    return {
        "role": role,
        "content": speech_content,  
        "visual_evidence": visual_evidence,
        "cached_visual_content": None  # Will be populated when needed
    }

def extract_json_blocks(text: str) -> list[str]:
    """
    Extract all code blocks that start with ```json and end with ```
    Returns a list of strings, each including the ```json ... ``` markers, in order of appearance.
    If none found, returns an empty list.
    """
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [f"```json{block}```" for block in matches]