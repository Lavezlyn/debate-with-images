import re
from typing import List, Dict, Union, Tuple, Any
import PIL
from PIL import Image
import base64
from io import BytesIO
import yaml
import os

def compress_image(image: Union[str, "PIL.Image"], max_size: Tuple[int, int] = (512,512), quality: int = 80) -> "PIL.Image":
    """
    Compress image to reduce file size and token count
    
    Args:
        image: PIL Image object or image file path
        max_size: Maximum dimensions (width, height) for the image
        quality: JPEG quality (1-100, higher is better quality but larger file)
    
    Returns:
        Compressed PIL Image object
    """
    try:
        if isinstance(image, str):
            image_input = Image.open(image)
        else:
            image_input = image.copy()
        
        # Resize image if it's too large
        if image_input.size[0] > max_size[0] or image_input.size[1] > max_size[1]:
            print(f"Compressing image {image_input.size} to {max_size}")
            image_input.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary (JPEG doesn't support transparency)
        if image_input.mode != 'RGB':
            image_input = image_input.convert('RGB')
        
        return image_input
        
    except Exception as e:
        raise ValueError(f"Failed to compress image: {str(e)}")

def encode_image_to_base64(image: Union[str, "PIL.Image"], max_size: Tuple[int, int] = (512,512), quality: int = 80, auto_compress: bool = False) -> str:
        """
        Convert image to base64 string with optional automatic compression
        
        Args:
            image: PIL Image object or image file path
            max_size: Maximum dimensions for compression (width, height)
            quality: JPEG quality for compression (1-100)
            auto_compress: Whether to automatically compress large images
        
        Returns:
            Base64 encoded string
        """
        try:
            if isinstance(image, str):
                image_input = Image.open(image)
            else:
                image_input = image
            
            # Auto-compress if enabled and image is large
            if auto_compress and (image_input.size[0] > max_size[0] or image_input.size[1] > max_size[1]):
                image_input = compress_image(image_input, max_size, quality)
            
            # Determine if image needs transparency (only for uncompressed images)
            if not auto_compress and image_input.mode == 'RGBA' and _has_transparency(image_input):
                buffer = BytesIO()
                image_input.save(buffer, format="PNG")  # Use PNG for images with transparency
            else:
                # Convert to RGB if necessary and save as JPEG
                if image_input.mode != 'RGB':
                    image_input = image_input.convert("RGB")
                buffer = BytesIO()
                image_input.save(buffer, format="JPEG", quality=quality, optimize=True)
            
            img_bytes = buffer.getvalue()
            base64_data = base64.b64encode(img_bytes).decode("utf-8")
            return base64_data
            
        except Exception as e:
            raise ValueError(f"Failed to encode image to base64: {str(e)}")


def _has_transparency(image: Image.Image) -> bool:
    """Check if image has any transparent pixels"""
    if image.mode == 'RGBA':
        extrema = image.getextrema()
        if len(extrema) >= 4:  # Make sure we have alpha channel
            alpha_min, alpha_max = extrema[3]
            return alpha_min < 255
    return False

def prompt_to_conversation(  
            user_prompt: str, 
            system_prompt: Union[str, None] = None, 
            images: Union[List[PIL.Image], List[str]] = []
        ) -> List[Dict]:
        """
        Convert input prompt string to the specified conversation format
        
        Args:
            user_prompt (str): Input user_prompt with image placeholders such as <image>, <imagen>, or <image n>. If no placeholder is present, images will be automatically prepended to the beginning of the text
            system_prompt (str): Input system_prompt (if exists)
            images (list): List of PIL.Image objects to be encoded and inserted into the conversation
            
        Returns:
            list: Conversation object in the specified format
        """
        content_parts = []
        matches = list(re.finditer(r'<image\s*(\d*)>', user_prompt))
        
        if matches:
            assert len(images) == len(matches), f"Number of images ({len(images)}) does not match number of placeholders ({len(matches)}), input user_prompt: {user_prompt}"
            
            last_end = 0
            for i, match in enumerate(matches):
                if match.start() > last_end:
                    content_parts.append({"type": "text", "text": user_prompt[last_end:match.start()]})
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(images[i], auto_compress=True)}"}})
                last_end = match.end()
                
            if last_end < len(user_prompt):
                content_parts.append({"type": "text", "text": user_prompt[last_end:]})
        else:
            content_parts.extend([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(img, auto_compress=True)}"}} for img in images])
            if user_prompt:
                content_parts.append({"type": "text", "text": user_prompt})

        conversation = [{"role": "user", "content": content_parts}]
        if system_prompt is not None:
            conversation.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        
        return conversation

def prompt_to_claude_conversation(  
            user_prompt: str, 
            system_prompt: Union[str, None] = None, 
            images: Union[List[PIL.Image], List[str]] = []
        ) -> List[Dict]:
        """
        Convert input prompt string to the specified conversation format
        
        Args:
            user_prompt (str): Input user_prompt with image placeholders such as <image>, <imagen>, or <image n>. If no placeholder is present, images will be automatically prepended to the beginning of the text
            system_prompt (str): Input system_prompt (if exists)
            images (list): List of PIL.Image objects to be encoded and inserted into the conversation
            
        Returns:
            list: Conversation object in the specified format
        """
        content_parts = []
        matches = list(re.finditer(r'<image\s*(\d*)>', user_prompt))
        
        if matches:
            assert len(images) == len(matches), f"Number of images ({len(images)}) does not match number of placeholders ({len(matches)}), input user_prompt: {user_prompt}"
            
            last_end = 0
            for i, match in enumerate(matches):
                if match.start() > last_end:
                    content_parts.append({"type": "text", "text": user_prompt[last_end:match.start()]})
                content_parts.append(
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": f"{encode_image_to_base64(images[i], auto_compress=False)}"}}
                )
                last_end = match.end()
                
            if last_end < len(user_prompt):
                content_parts.append({"type": "text", "text": user_prompt[last_end:]})
        else:
            content_parts.extend([{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": f"{encode_image_to_base64(img, auto_compress=False)}"}} for img in images])
            if user_prompt:
                content_parts.append({"type": "text", "text": user_prompt})

        conversation = [{"role": "user", "content": content_parts}]
        if system_prompt is not None:
            conversation.insert(0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        
        return conversation

def prompt_to_reason_conversation(  
            user_prompt: str, 
            system_prompt: Union[str, None] = None, 
            images: Union[List[PIL.Image], List[str]] = []
        ) -> List[Dict]:
        """
        Convert input prompt string to the specified conversation format
        
        Args:
            user_prompt (str): Input user_prompt with image placeholders such as <image>, <imagen>, or <image n>. If no placeholder is present, images will be automatically prepended to the beginning of the text
            system_prompt (str): Input system_prompt (if exists)
            images (list): List of PIL.Image objects to be encoded and inserted into the conversation
            
        Returns:
            list: Conversation object in the specified format
        """
        content_parts = []
        matches = list(re.finditer(r'<image\s*(\d*)>', user_prompt))
        
        if matches:
            assert len(images) == len(matches), f"Number of images ({len(images)}) does not match number of placeholders ({len(matches)}), input user_prompt: {user_prompt}"
            
            last_end = 0
            for i, match in enumerate(matches):
                if match.start() > last_end:
                    content_parts.append({"type": "input_text", "text": user_prompt[last_end:match.start()]})
                content_parts.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{encode_image_to_base64(images[i], auto_compress=True)}"})
                last_end = match.end()
                
            if last_end < len(user_prompt):
                content_parts.append({"type": "input_text", "text": user_prompt[last_end:]})
        else:
            content_parts.extend([{"type": "input_image", "image_url": f"data:image/jpeg;base64,{encode_image_to_base64(img, auto_compress=True)}"} for img in images])
            if user_prompt:
                content_parts.append({"type": "input_text", "text": user_prompt})

        conversation = [{"role": "user", "content": content_parts}]
        if system_prompt is not None:
            conversation.insert(0, {"role": "developer", "content": [{"type": "input_text", "text": system_prompt}]})
        
        return conversation

def parse_yaml_config(yaml_path: str) -> Tuple[str, str, dict, dict, bool, str]:
    """
    Parse YAML configuration file and return engine type, model name, inference configs, model configs, cache setting, and API key.
    
    Args:
        yaml_path (str): Path to the YAML configuration file
        
    Returns:
        Tuple[str, str, Dict[str, Any], Dict[str, Any], bool, str]: (engine, model_name, infer_cfgs, model_cfgs, use_cache, api_key)
        
    Expected YAML structure:
    engine: "api" or "vllm"
    model_name: "your-model-name"
    use_cache: true/false
    api_key: "sk-..."  # Only for API engine
    # For API engine:
    infer_cfgs:
        temperature: 0.7
        max_tokens: 1000
        # ... other inference configs
    # For VLLM engine:
    model_cfgs:
        model_path: "/path/to/model"
        tensor_parallel_size: 1
        # ... other model configs
    infer_cfgs:
        temperature: 0.7
        max_tokens: 1000
        # ... other inference configs
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {yaml_path}: {e}")
    
    # Extract engine type
    engine = config.get('engine', 'api')
    if engine not in ['api', 'vllm', 'reason_api', 'claude_reason_api']:
        raise ValueError(f"Invalid engine '{engine}' in YAML config. Must be 'api' or 'vllm' or 'reason_api' or 'claude_reason_api': {yaml_path}")
    
    # Extract model name
    model_name = config.get('model_name')
    if not model_name:
        raise ValueError(f"Missing 'model_name' in YAML config: {yaml_path}")
    
    # Extract cache setting
    use_cache = config.get('use_cache', True)
    if not isinstance(use_cache, bool):
        raise ValueError(f"'use_cache' must be a boolean in YAML config: {yaml_path}")
    
    # Extract API key (optional, only for API engine)
    api_key = config.get('api_key', None)
    
    # Extract inference configs
    infer_cfgs = config.get('infer_cfgs', {})
    if not isinstance(infer_cfgs, dict):
        raise ValueError(f"'infer_cfgs' must be a dictionary in YAML config: {yaml_path}")
    
    # Extract model configs (only required for VLLM)
    model_cfgs = config.get('model_cfgs', {})
    if not isinstance(model_cfgs, dict):
        raise ValueError(f"'model_cfgs' must be a dictionary in YAML config: {yaml_path}")
    
    # Validate engine-specific requirements
    if engine == 'vllm' and not model_cfgs.get('model_path'):
        raise ValueError(f"'model_path' is required in model_cfgs for VLLM engine: {yaml_path}")
    
    return engine, model_name, infer_cfgs, model_cfgs, use_cache, api_key

def post_process_reason_response(response: list) -> dict:
    """
    Post-process the response to the reason engine.
    """
    reasoning = ""
    output = ""
    if len(response) == 2:
        reasoning = response[0]["summary"]
        output = response[1]["content"][0]["text"].strip()
    elif len(response) == 1:
        output = response[0]["content"][0]["text"].strip()
    else:
        reasoning = "Error"
        output = "Error"

    return {
        "reasoning": reasoning,
        "output": output
    }

def post_process_response(response: str) -> dict:
    # Extract thinking content
    thinking = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if thinking:
        thinking_content = thinking.group(1).strip()
        thinking_end_pos = thinking.end()
    else:
        # Check if <think> tag exists but is incomplete (no closing tag)
        incomplete_thinking = re.search(r'<think>(.*?)(?=<output>|$)', response, re.DOTALL)
        if incomplete_thinking:
            thinking_content = incomplete_thinking.group(1).strip()
            thinking_end_pos = incomplete_thinking.end()
        else:
            thinking_content = ""
            thinking_end_pos = 0
    
    # Extract output content
    output = re.search(r'<output>(.*?)</output>', response, re.DOTALL)
    if output:
        output_content = output.group(1).strip()
        output_start_pos = output.start()
    else:
        # Check if <output> tag exists but is incomplete (no closing tag)
        incomplete_output = re.search(r'<output>(.*?)$', response, re.DOTALL)
        if incomplete_output:
            output_content = incomplete_output.group(1).strip()
            output_start_pos = incomplete_output.start()
        elif thinking_content and thinking_end_pos > 0:
            # If <think> is complete but <output> is missing/incomplete,
            # extract everything after </think> as output
            remaining_text = response[thinking_end_pos:].strip()
            # Remove any incomplete <output> tag at the beginning
            remaining_text = re.sub(r'^<output>\s*', '', remaining_text)
            output_content = remaining_text
            output_start_pos = thinking_end_pos
        else:
            output_content = ""
            output_start_pos = len(response)
    
    # If <think> is incomplete but <output> is complete, 
    # extract everything before <output> as thinking content
    if not thinking_content and output_content and output_start_pos > 0:
        text_before_output = response[:output_start_pos].strip()
        # Remove any incomplete <think> tag at the beginning
        text_before_output = re.sub(r'^<think>\s*', '', text_before_output)
        if text_before_output:
            thinking_content = text_before_output

    return {
        "reasoning": thinking_content,
        "output": output_content,
        "original_response": response
    }

def post_process_claude_reason_response(response: list) -> dict:
    """
    Post-process the response to the reason engine.
    """
    response = response["content"]
    reasoning = ""
    output = ""
    if len(response) == 2:
        reasoning = response[0]["thinking"]
        output = response[1]["text"].strip()
    elif len(response) == 1:
        output = response[0]["text"].strip()
    else:
        reasoning = "Error"
        output = "Error"

    return {
        "reasoning": reasoning,
        "output": output,
    }