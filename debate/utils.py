"""
General Utility Functions Module
Import all functions from each specialized module, providing a unified interface
"""

# Import all functions from each specialized module
from api_utils import (
    bean_gpt_api,
    generate_hash_uid,
    api,
    vllm_api
)

from format_utils import (
    load_debate_ds,
    build_inference_payload,
    clear_visual_cache,
    build_init_user_prompt,
    format_history_item,
    extract_json_blocks
)

from visual_utils import (
    zoom_in_region,
    create_zoom_composition,
    convert_normalized_bbox_to_pixels,
    find_non_overlapping_position,
    draw_text_with_font,
    draw_point_annotation,
    draw_bbox_annotation,
    draw_line_annotation,
    process_zoom_operations,
    create_annotated_image,
    generate_depth_map,
    process_depth_operations,
    apply_visual_ops,
    combine_images,
    compress_image,
    encode_image_to_base64
)

__all__ = [
    # API related
    'bean_gpt_api',
    'generate_hash_uid', 
    'api',
    'vllm_api',
    
    # Formatting related
    'load_debate_ds',
    'build_inference_payload',
    'clear_visual_cache',
    'build_init_user_prompt',
    'format_history_item',
    'extract_json_blocks',
    'parse_structured_response',
    'normalize_text',
    'truncate_text',
    'extract_key_value_pairs',
    
    # Visualization related
    'zoom_in_region',
    'create_zoom_composition',
    'convert_normalized_bbox_to_pixels',
    'find_non_overlapping_position',
    'draw_text_with_font',
    'draw_point_annotation',
    'draw_bbox_annotation',
    'draw_line_annotation',
    'process_zoom_operations',
    'create_annotated_image',
    'generate_depth_map',
    'process_depth_operations',
    'apply_visual_ops',
    'combine_images',
    'compress_image',
    'encode_image_to_base64',
    'apply_visual_operations',
    'extract_visual_operations',
    'process_visual_evidence'
]
