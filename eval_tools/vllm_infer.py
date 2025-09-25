"""
VLLM inference module for multimodal deception evaluation.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
import PIL
from PIL import Image
import base64
from io import BytesIO
import time
from transformers import AutoProcessor
from tqdm import tqdm

from vllm import LLM, SamplingParams

from utils import prompt_to_conversation, encode_image_to_base64

class VLLMInferenceEngine:
    """VLLM inference engine for multimodal models."""
    
    def __init__(self, model_cfgs: Dict[str, Any]):
        """
        Initialize VLLM model with the given configurations.
        
        Args:
            model_cfgs (Dict[str, Any]): Model configuration dictionary
        """
        self.model_cfgs = model_cfgs
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the VLLM model with the provided configurations."""
        try:
            # Extract model path and other configurations
            model_path = self.model_cfgs.get('model_path')
            if not model_path:
                raise ValueError("'model_path' is required in model_cfgs")
            
            # Set up VLLM model parameters
            vllm_params = {
                'model': model_path,
                'tensor_parallel_size': self.model_cfgs.get('tensor_parallel_size', 1),
                'gpu_memory_utilization': self.model_cfgs.get('gpu_memory_utilization', 0.9),
                'max_model_len': self.model_cfgs.get('max_model_len', 4096),
                'trust_remote_code': self.model_cfgs.get('trust_remote_code', True),
                'dtype': self.model_cfgs.get('dtype', 'auto'),
                'limit_mm_per_prompt': {'image': 2}
            }
            
            # Add optional parameters if specified
            if 'swap_space' in self.model_cfgs:
                vllm_params['swap_space'] = self.model_cfgs['swap_space']
            if 'enforce_eager' in self.model_cfgs:
                vllm_params['enforce_eager'] = self.model_cfgs['enforce_eager']

            self.model = LLM(**vllm_params)
            self.processor = AutoProcessor.from_pretrained(model_path)
            
            
        except Exception as e:
            logging.error(f"Failed to initialize VLLM model: {e}")
            raise
        
    def _prepare_vllm_inputs(self, system_prompts: List[str], user_prompts: List[str], 
                             images: List[List[Union[str, PIL.Image]]]) -> List[Dict[str, Any]]:
        """
        Prepare VLLM inputs for inference.
        """
        vllm_inputs = []
        for user_prompt, system_prompt, image_list in tqdm(zip(user_prompts, system_prompts, images), total=len(user_prompts), desc="Preparing VLLM inputs"):
            conversation = prompt_to_conversation(user_prompt, system_prompt, image_list)
            prompts = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            mm_data = []
            for image in image_list:
                if isinstance(image, str):
                    image = Image.open(image)
                elif isinstance(image, PIL.Image.Image):
                    image = image
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")
                mm_data.append(image)
            vllm_inputs.append(
                {
                    "prompt": prompts,  
                    "multi_modal_data": {"image": mm_data},
                }
            )
        return vllm_inputs
    
    def infer(self, system_prompts: List[str], user_prompts: List[str], 
              images: List[List[Union[str, PIL.Image]]], 
              infer_cfgs: Dict[str, Any]) -> List[str]:
        """
        Perform inference using VLLM.
        
        Args:
            system_prompts: List of system prompts
            user_prompts: List of user prompts  
            images: List of image lists for each prompt
            infer_cfgs: Inference configuration dictionary
            
        Returns:
            List of generated responses
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("VLLM model not initialized")
        
        vllm_inputs = self._prepare_vllm_inputs(system_prompts, user_prompts, images)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=infer_cfgs.get('temperature', 1.0),
            max_tokens=infer_cfgs.get('max_tokens', 1024),
            top_p=infer_cfgs.get('top_p', 1.0),
            top_k=infer_cfgs.get('top_k', -1),
            repetition_penalty=infer_cfgs.get('repetition_penalty', 1.0),
        )
        
        try:
            # Perform inference
            outputs = self.model.generate(vllm_inputs, sampling_params)
            
            # Extract generated text
            responses = []
            for output in outputs:
                if output.outputs:
                    generated_text = output.outputs[0].text
                    responses.append(generated_text)
                else:
                    responses.append("")
            
            return responses
            
        except Exception as e:
            logging.error(f"Error during VLLM inference: {e}")
            raise

def vllm_api(system_contents: List[str], user_contents: List[str], 
             images: Optional[List[List[Union[str, PIL.Image]]]] = None,
             model_cfgs: Dict[str, Any] = None,
             infer_cfgs: Dict[str, Any] = None) -> List[str]:
    """
    VLLM API function compatible with the existing API interface.
    
    Args:
        system_contents: List of system prompts
        user_contents: List of user prompts
        images: List of image lists (optional)
        model_cfgs: Model configuration dictionary
        infer_cfgs: Inference configuration dictionary
        
    Returns:
        List of generated responses
    """
    if model_cfgs is None:
        raise ValueError("model_cfgs is required for VLLM inference")
    if infer_cfgs is None:
        infer_cfgs = {}
    
    # Initialize VLLM engine
    engine = VLLMInferenceEngine(model_cfgs)
    
    # Perform inference
    responses = engine.infer(system_contents, user_contents, images, infer_cfgs)
    
    return responses
    