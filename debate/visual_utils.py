"""
Visual operation tools
Include all functions related to image processing, annotation, and visualization
"""

import base64
import json
import re
import warnings
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def zoom_in_region(image: Image.Image, 
                   bbox: List[float], 
                   output_size: Optional[Tuple[int, int]] = None,
                   padding: float = 0.0) -> Image.Image:
    """
    Crop and scale image region based on normalized coordinates
    
    Args:
        image: Input PIL Image object
        bbox: Normalized bounding box [x, y, w, h], all values in [0, 1]
        output_size: Output image size (width, height), None keeps original crop size
        padding: Bounding box expansion ratio (0.0-1.0), 0 means no expansion
    
    Returns:
        Cropped and possibly scaled PIL Image object
    """
    if len(bbox) != 4:
        raise ValueError("bbox must contain 4 values: [x, y, w, h]")
    
    x_norm, y_norm, w_norm, h_norm = bbox
    
    if not all(0 <= val <= 1 for val in bbox):
        raise ValueError("All bbox values must be in [0, 1] range")
    
    if w_norm <= 0 or h_norm <= 0:
        raise ValueError("Width and height must be greater than 0")
    
    img_width, img_height = image.size
    
    x_pixel = x_norm * img_width
    y_pixel = y_norm * img_height
    w_pixel = w_norm * img_width
    h_pixel = h_norm * img_height
    
    if padding > 0:
        padding_x = w_pixel * padding
        padding_y = h_pixel * padding
        
        x_pixel = max(0, x_pixel - padding_x)
        y_pixel = max(0, y_pixel - padding_y)
        w_pixel = min(img_width - x_pixel, w_pixel + 2 * padding_x)
        h_pixel = min(img_height - y_pixel, h_pixel + 2 * padding_y)
    
    x_pixel = max(0, min(x_pixel, img_width))
    y_pixel = max(0, min(y_pixel, img_height))
    x2_pixel = min(img_width, x_pixel + w_pixel)
    y2_pixel = min(img_height, y_pixel + h_pixel)
    
    actual_w = x2_pixel - x_pixel
    actual_h = y2_pixel - y_pixel
    
    if actual_w <= 0 or actual_h <= 0:
        raise ValueError("Invalid crop region: calculated region is empty")
    
    crop_box = (int(x_pixel), int(y_pixel), int(x2_pixel), int(y2_pixel))
    cropped_image = image.crop(crop_box)
    
    if output_size is not None:
        output_width, output_height = output_size
        if output_width <= 0 or output_height <= 0:
            raise ValueError("Output size must be greater than 0")
        cropped_image = cropped_image.resize((output_width, output_height), 
                                           Image.Resampling.LANCZOS)
    
    return cropped_image

def create_zoom_composition(zoom_regions: List[Image.Image], zoom_labels: List[str]) -> Image.Image:
    """
    Arrange multiple zoom regions into a blank canvas
    
    Args:
        zoom_regions: List of zoom region images (up to 3)
        zoom_labels: List of corresponding labels
    
    Returns:
        Arranged composed image
    """
    if not zoom_regions:
        return None
    
    regions = zoom_regions[:3]
    labels = zoom_labels[:len(regions)]
    
    canvas_width = 1200
    canvas_height = 800
    background_color = 'white'
    margin = 30
    label_height = 50
    
    canvas = Image.new('RGB', (canvas_width, canvas_height), background_color)
    draw = ImageDraw.Draw(canvas)
    
    try:
        title_font = ImageFont.truetype("Arial.ttf", 24)
        label_font = ImageFont.truetype("Arial.ttf", 16)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    title = f"Zoom Regions ({len(regions)} regions arranged)"
    try:
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
    except:
        title_width = len(title) * 12
    
    title_x = (canvas_width - title_width) // 2
    draw.text((title_x, 15), title, fill='black', font=title_font)
    
    available_height = canvas_height - 60 - margin
    available_width = canvas_width - 2 * margin
    
    if len(regions) == 1:
        # 单个区域：居中放置
        region = regions[0]
        label = labels[0]
        
        max_width = available_width
        max_height = available_height - label_height
        
        scale_w = max_width / region.size[0]
        scale_h = max_height / region.size[1]
        scale = min(scale_w, scale_h, 1.0)
        
        new_width = int(region.size[0] * scale)
        new_height = int(region.size[1] * scale)
        
        if scale < 1.0:
            region = region.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        x = (canvas_width - new_width) // 2
        y = 60 + (available_height - label_height - new_height) // 2
        
        canvas.paste(region, (x, y))
        
        try:
            label_bbox = draw.textbbox((0, 0), label, font=label_font)
            label_width = label_bbox[2] - label_bbox[0]
        except:
            label_width = len(label) * 8
        
        label_x = (canvas_width - label_width) // 2
        label_y = y + new_height + 10
        draw.text((label_x, label_y), label, fill='black', font=label_font)
        
    elif len(regions) == 2:
        # 两个区域：左右排列
        region_width = (available_width - margin) // 2
        region_height = available_height - label_height
        
        for i, (region, label) in enumerate(zip(regions, labels)):
            scale_w = region_width / region.size[0]
            scale_h = region_height / region.size[1]
            scale = min(scale_w, scale_h, 1.0)
            
            new_width = int(region.size[0] * scale)
            new_height = int(region.size[1] * scale)
            
            if scale < 1.0:
                region = region.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            x = margin + i * (region_width + margin) + (region_width - new_width) // 2
            y = 60 + (region_height - new_height) // 2
            
            canvas.paste(region, (x, y))
            
            try:
                label_bbox = draw.textbbox((0, 0), label, font=label_font)
                label_width = label_bbox[2] - label_bbox[0]
            except:
                label_width = len(label) * 8
            
            label_x = margin + i * (region_width + margin) + (region_width - label_width) // 2
            label_y = y + new_height + 10
            draw.text((label_x, label_y), label, fill='black', font=label_font)
    
    else:  
        # Three regions: 1 on top, 2 on bottom
        top_region = regions[0]
        top_label = labels[0]
        
        top_height = (available_height - label_height * 2 - margin) // 2
        
        scale_w = available_width / top_region.size[0]
        scale_h = top_height / top_region.size[1]
        scale = min(scale_w, scale_h, 1.0)
        
        top_width = int(top_region.size[0] * scale)
        top_height_actual = int(top_region.size[1] * scale)
        
        if scale < 1.0:
            top_region = top_region.resize((top_width, top_height_actual), Image.Resampling.LANCZOS)
        
        top_x = (canvas_width - top_width) // 2
        top_y = 60
        
        canvas.paste(top_region, (top_x, top_y))
        
        try:
            label_bbox = draw.textbbox((0, 0), top_label, font=label_font)
            label_width = label_bbox[2] - label_bbox[0]
        except:
            label_width = len(top_label) * 8
        
        label_x = (canvas_width - label_width) // 2
        label_y = top_y + top_height_actual + 5
        draw.text((label_x, label_y), top_label, fill='black', font=label_font)
        
        # 下半部分：两个区域
        bottom_start_y = top_y + top_height_actual + label_height + margin
        bottom_height = available_height - (bottom_start_y - 60) - label_height
        bottom_width = (available_width - margin) // 2
        
        for i, (region, label) in enumerate(zip(regions[1:], labels[1:])):
            scale_w = bottom_width / region.size[0]
            scale_h = bottom_height / region.size[1]
            scale = min(scale_w, scale_h, 1.0)
            
            new_width = int(region.size[0] * scale)
            new_height = int(region.size[1] * scale)
            
            if scale < 1.0:
                region = region.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            x = margin + i * (bottom_width + margin) + (bottom_width - new_width) // 2
            y = bottom_start_y + (bottom_height - new_height) // 2
            
            canvas.paste(region, (x, y))
            
            try:
                label_bbox = draw.textbbox((0, 0), label, font=label_font)
                label_width = label_bbox[2] - label_bbox[0]
            except:
                label_width = len(label) * 8
            
            label_x = margin + i * (bottom_width + margin) + (bottom_width - label_width) // 2
            label_y = y + new_height + 5
            draw.text((label_x, label_y), label, fill='black', font=label_font)
    
    return canvas

def convert_normalized_bbox_to_pixels(bbox, img_width, img_height):
    """Convert normalized bbox [x, y, w, h] to pixel coordinates [x1, y1, x2, y2]."""
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {len(bbox)}")
    
    x, y, w, h = bbox
    
    x1 = x * img_width
    y1 = y * img_height
    x2 = (x + w) * img_width
    y2 = (y + h) * img_height
    
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    
    if x2 <= x1:
        x2 = min(x1 + 1, img_width)
    if y2 <= y1:
        y2 = min(y1 + 1, img_height)
        
    return [x1, y1, x2, y2]

def find_non_overlapping_position(bbox, label, text_positions, width, height, font_size=12, is_point=False):
    """Find a position for text that doesn't overlap with existing text or go outside image bounds."""
    text_width = len(label) * font_size * 0.6
    text_height = font_size + 8
    
    if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        if is_point:
            center_x = width // 2
            center_y = height // 2
            bbox = [center_x - 50, center_y - 50, center_x + 50, center_y + 50]
        else:
            bbox = [0, 0, min(100, width), min(100, height)]
    
    candidate_positions = [
        (bbox[0], bbox[1] - 20),
        (bbox[0], bbox[3] + 10),
        (bbox[2] + 10, bbox[1]),
        (bbox[2] + 10, bbox[3] - text_height),
        (bbox[0] - text_width - 10, bbox[1]),
        (bbox[0] - text_width - 10, bbox[3] - text_height),
        (bbox[0] + (bbox[2] - bbox[0] - text_width) / 2, bbox[1] - 20),
        (bbox[0] + (bbox[2] - bbox[0] - text_width) / 2, bbox[3] + 10),
        (bbox[0] - text_width - 20, bbox[1] + (bbox[3] - bbox[1]) / 2),
        (bbox[2] + 20, bbox[1] + (bbox[3] - bbox[1]) / 2),
    ]
    
    for i, (x, y) in enumerate(candidate_positions):
        if (x >= 0 and y >= text_height and 
            x + text_width <= width and y + text_height <= height):
            
            overlap = False
            for existing_x, existing_y, existing_w, existing_h in text_positions:
                if not (x + text_width + 5 < existing_x or x > existing_x + existing_w + 5 or
                        y + text_height + 5 < existing_y or y > existing_y + existing_h + 5):
                    overlap = True
                    break
            
            if not overlap:
                return x, y, text_width, text_height
    
    base_x, base_y = bbox[0], bbox[1] - 20
    offset_y = len(text_positions) * 25
    offset_x = (len(text_positions) % 3) * 100
    
    final_x = max(10, min(base_x + offset_x, width - text_width - 10))
    final_y = max(text_height + 5, min(base_y - offset_y, height - text_height - 5))
    
    return final_x, final_y, text_width, text_height

def draw_text_with_font(ax, text_x, text_y, label, color, chinese_font, font_size=12):
    """Draw text with Chinese font support."""
    if chinese_font:
        plt.text(text_x, text_y, label, color=color, fontsize=font_size, fontweight='bold',
               fontproperties=chinese_font,
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor=color, linewidth=2))
    else:
        plt.text(text_x, text_y, label, color=color, fontsize=font_size, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor=color, linewidth=2))

def draw_point_annotation(ax, item, i, color, width, height, text_positions, shown_labels, chinese_font):
    """Draw point annotation on the plot."""
    point = item['point_2d']
    label = item['label']
    
    if not point or len(point) != 2:
        return False
    
    try:
        pixel_x = point[0] * width
        pixel_y = point[1] * height
        
        pixel_x = max(0, min(pixel_x, width))
        pixel_y = max(0, min(pixel_y, height))
    except Exception as e:
        return False
    
    circle = patches.Circle((pixel_x, pixel_y), radius=8, 
                          linewidth=3, edgecolor=color, facecolor=color, alpha=0.7)
    ax.add_patch(circle)
    
    ax.text(pixel_x, pixel_y, str(i+1), 
           ha='center', va='center', fontsize=12, fontweight='bold',
           color='white', bbox=dict(boxstyle="circle,pad=0.2", facecolor=color, edgecolor=color))
    
    if label not in shown_labels:
        shown_labels.add(label)
        
        point_bbox = [pixel_x-10, pixel_y-10, pixel_x+10, pixel_y+10]
        text_x, text_y, text_w, text_h = find_non_overlapping_position(point_bbox, label, text_positions, width, height, font_size=12, is_point=True)
        text_positions.append((text_x, text_y, text_w, text_h))
        
        line_end_x = text_x + text_w / 2
        line_end_y = text_y + text_h / 2
        
        ax.plot([pixel_x, line_end_x], [pixel_y, line_end_y], 
               color=color, linestyle='--', alpha=0.6, linewidth=2)
        
        draw_text_with_font(ax, text_x, text_y, label, color, chinese_font)
    
    return True

def draw_bbox_annotation(ax, item, i, color, width, height, text_positions, shown_labels, chinese_font):
    """Draw bounding box annotation on the plot."""
    bbox = item['bbox_2d']
    label = item['label']
    
    if not bbox or len(bbox) != 4:
        return False
    
    try:
        pixel_bbox = convert_normalized_bbox_to_pixels(bbox, width, height)
    except Exception as e:
        return False
    
    rect = patches.Rectangle((pixel_bbox[0], pixel_bbox[1]), 
                           pixel_bbox[2] - pixel_bbox[0], pixel_bbox[3] - pixel_bbox[1], 
                           linewidth=3, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    
    box_center_x = pixel_bbox[0] + (pixel_bbox[2] - pixel_bbox[0]) / 2
    box_center_y = pixel_bbox[1] + (pixel_bbox[3] - pixel_bbox[1]) / 2
    ax.text(box_center_x, box_center_y, str(i+1), 
           ha='center', va='center', fontsize=14, fontweight='bold',
           bbox=dict(boxstyle="circle,pad=0.3", facecolor='white', edgecolor=color))
    
    if label not in shown_labels:
        shown_labels.add(label)
        
        text_x, text_y, text_w, text_h = find_non_overlapping_position(pixel_bbox, label, text_positions, width, height, font_size=12)
        text_positions.append((text_x, text_y, text_w, text_h))
        
        line_start_x = pixel_bbox[0] + (pixel_bbox[2] - pixel_bbox[0]) / 2
        line_start_y = pixel_bbox[1] + (pixel_bbox[3] - pixel_bbox[1]) / 2
        line_end_x = text_x + text_w / 2
        line_end_y = text_y + text_h / 2
        
        ax.plot([line_start_x, line_end_x], [line_start_y, line_end_y], 
               color=color, linestyle='--', alpha=0.6, linewidth=2)
        
        draw_text_with_font(ax, text_x, text_y, label, color, chinese_font)
    
    return True

def draw_line_annotation(ax, item, i, color, width, height, text_positions, shown_labels, chinese_font):
    """Draw line annotation on the plot."""
    line = item['line_2d']
    label = item['label']
    
    if not line or len(line) != 4:
        return False
    
    try:
        x1, y1, x2, y2 = line
        
        pixel_x1 = x1 * width
        pixel_y1 = y1 * height
        pixel_x2 = x2 * width
        pixel_y2 = y2 * height
        
        pixel_x1 = max(0, min(pixel_x1, width))
        pixel_y1 = max(0, min(pixel_y1, height))
        pixel_x2 = max(0, min(pixel_x2, width))
        pixel_y2 = max(0, min(pixel_y2, height))
    except Exception as e:
        return False
    
    ax.plot([pixel_x1, pixel_x2], [pixel_y1, pixel_y2], 
           color=color, linewidth=4, alpha=0.8)
    
    ax.annotate('', xy=(pixel_x2, pixel_y2), xytext=(pixel_x1, pixel_y1),
               arrowprops=dict(arrowstyle='->', color=color, lw=3))
    
    mid_x = (pixel_x1 + pixel_x2) / 2
    mid_y = (pixel_y1 + pixel_y2) / 2
    ax.text(mid_x, mid_y, str(i+1), 
           ha='center', va='center', fontsize=12, fontweight='bold',
           color='white', bbox=dict(boxstyle="circle,pad=0.2", facecolor=color, edgecolor=color))
    
    if label not in shown_labels:
        shown_labels.add(label)
        
        line_bbox = [mid_x-20, mid_y-20, mid_x+20, mid_y+20]
        text_x, text_y, text_w, text_h = find_non_overlapping_position(line_bbox, label, text_positions, width, height, font_size=12)
        text_positions.append((text_x, text_y, text_w, text_h))
        
        line_end_x = text_x + text_w / 2
        line_end_y = text_y + text_h / 2
        
        ax.plot([mid_x, line_end_x], [mid_y, line_end_y], 
               color=color, linestyle='--', alpha=0.6, linewidth=2)
        
        draw_text_with_font(ax, text_x, text_y, label, color, chinese_font)
    
    return True

def process_zoom_operations(image, bbox_data):
    """Process zoom operations and return zoom regions and labels."""
    zoom_regions = []
    zoom_labels = []
    zoom_count = 0
    max_zoom = 3
    
    for i, item in enumerate(bbox_data):
        if isinstance(item, dict) and 'zoom_in_2d' in item and zoom_count < max_zoom:
            zoom_bbox = item['zoom_in_2d']
            label = item.get('label', f'Zoom Region {zoom_count+1}')
            
            if not zoom_bbox or len(zoom_bbox) != 4:
                continue
            
            output_size = item.get('output_size', None)
            padding = item.get('padding', 0.0)
            
            if output_size and isinstance(output_size, list) and len(output_size) == 2:
                output_size = tuple(output_size)
            
            try:
                zoomed_region = zoom_in_region(image, zoom_bbox, output_size, padding)
                zoom_regions.append(zoomed_region)
                zoom_labels.append(label)
                zoom_count += 1
            except Exception as e:
                print(f"Warning: Failed to create zoom region {zoom_count+1}: {e}")
                continue
    
    return zoom_regions, zoom_labels

def create_annotated_image(image, bbox_data):
    """Create the annotated image with all non-zoom annotations."""
    try:
        width, height = image.size
    except Exception as e:
        return image
    
    chinese_font = None
    try:
        macos_chinese_fonts = [
            'Hiragino Sans GB',
            'STHeiti',
            'Arial Unicode MS',
            'Helvetica',
        ]
        
        for font_name in macos_chinese_fonts:
            try:
                chinese_font = fm.FontProperties(family=font_name)
                break
            except:
                continue
    except:
        chinese_font = None
    
    # Only filter font related warnings when drawing text, other operations preserve all warnings
    fig, ax = plt.subplots(1, figsize=(15, 10))
    ax.imshow(image)
    ax.axis('off')
    
    shown_labels = set()
    text_positions = []
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, item in enumerate(bbox_data):
        if not isinstance(item, dict) or 'label' not in item or 'zoom_in_2d' in item or 'depth' in item:
            continue
        
        color = colors[i % len(colors)]
        
        # Only filter font related warnings when calling functions that may use fonts
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*missing from font.*")
            if 'point_2d' in item:
                draw_point_annotation(ax, item, i, color, width, height, text_positions, shown_labels, chinese_font)
            elif 'bbox_2d' in item:
                draw_bbox_annotation(ax, item, i, color, width, height, text_positions, shown_labels, chinese_font)
            elif 'line_2d' in item:
                draw_line_annotation(ax, item, i, color, width, height, text_positions, shown_labels, chinese_font)

    legend_elements = []
    for i, item in enumerate(bbox_data):
        if 'zoom_in_2d' not in item:
            color = colors[i % len(colors)]
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=3, label=f'Evidence {i+1}'))
    
    # Legend drawing may also involve fonts, so filter font warnings
    if len(legend_elements) <= 10:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*missing from font.*")
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2, dpi=150)
    plt.close(fig)
    buf.seek(0)
    plotted_image = Image.open(buf).convert("RGB")
    buf.close()
    
    return plotted_image

def generate_depth_map(image: Image.Image, method: str = "simple") -> Image.Image:
    """
    Generate depth map of the image
    
    Args:
        image: Input PIL Image object
        method: Depth estimation method ("simple", "gradient", "edge")
    
    Returns:
        Depth map PIL Image object
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale image
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    if method == "simple":
        # Simple depth estimation based on brightness   
        depth = 255 - gray
        
    elif method == "gradient":
        # Depth estimation based on gradient
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-255 range
        gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
        depth = gradient_magnitude.astype(np.uint8)
        
    elif method == "edge":
        # Depth estimation based on edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Use distance transform
        dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        # Normalize to 0-255 range
        depth = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    else:
        # Default use simple method
        depth = 255 - gray
    
    # Apply color mapping, make depth map more intuitive
    # Use JET color mapping: blue=far, red=near
    depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    
    # Convert back to RGB format
    depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    depth_image = Image.fromarray(depth_rgb)
    
    # Add title
    draw = ImageDraw.Draw(depth_image)
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    title = "Depth Map"
    try:
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
    except:
        text_width = len(title) * 12
    
    # Add title with white background on top of the image
    title_height = 30
    title_bg = Image.new('RGB', (depth_image.width, title_height), 'white')
    title_draw = ImageDraw.Draw(title_bg)
    
    title_x = (depth_image.width - text_width) // 2
    title_draw.text((title_x, 5), title, fill='black', font=font)
    
    # Merge title and depth map
    final_image = Image.new('RGB', (depth_image.width, depth_image.height + title_height))
    final_image.paste(title_bg, (0, 0))
    final_image.paste(depth_image, (0, title_height))
    
    return final_image

def process_depth_operations(image: Image.Image, bbox_data: List[Dict]) -> List[Image.Image]:
    """
    Process depth operations, generate depth map
    - Only allow one depth request
    - Default is full image operation
    - Use edge method
    
    Args:
        image: Source image
        bbox_data: List of annotation data
    
    Returns:
        Depth map list (up to 1 depth map)
    """
    # Check if there is a depth operation request (only process the first one)
    has_depth_request = any(
        isinstance(item, dict) and 'depth' in item 
        for item in bbox_data
    )
    
    if has_depth_request:
        try:
            # Use edge method to generate single depth map
            depth_img = generate_depth_map(image, method="edge")
            return [depth_img]
        except Exception as e:
            print(f"Warning: Failed to generate depth map: {e}")
            return []
    
    return []

def apply_visual_ops(image, bbox_data):
    """
    Refactored main function: support depth map functionality
    
    Args:
        image: PIL Image object
        bbox_data: List of dicts with annotation data
    
    Returns:
        List of PIL Images:
            - Always contains: [annotated_image]
            - If there is zoom operation: [..., zoom_composition]
            - If there is depth operation: [..., depth_map(s)]
    """
    if (bbox_data is None or len(bbox_data) == 0 or image is None):
        return []
        
    result_images = []
    
    # 1. Create annotated image
    annotated_image = create_annotated_image(image, bbox_data)
    result_images.append(annotated_image)
    
    # 2. Process zoom operation
    zoom_regions, zoom_labels = process_zoom_operations(image, bbox_data)
    if zoom_regions:
        composed_zoom = create_zoom_composition(zoom_regions, zoom_labels)
        result_images.append(composed_zoom)
    
    # 3. Process depth operation
    depth_images = process_depth_operations(image, bbox_data)
    if depth_images:
        result_images.extend(depth_images)
    
    return result_images

def combine_images(images: List[Image.Image], layout: str = "horizontal", spacing: int = 10, background_color: str = "white") -> Image.Image:
    """
    Combine multiple PIL images into a single image.
    
    Args:
        images: List of PIL Image objects
        layout: "horizontal" or "vertical" arrangement
        spacing: Pixels of spacing between images
        background_color: Background color for spacing
    
    Returns:
        Combined PIL Image
    """
    if not images:
        raise ValueError("No images provided")
    
    if len(images) == 1:
        return images[0].copy()
    
    rgb_images = []
    for img in images:
        if img.mode != 'RGB':
            rgb_images.append(img.convert('RGB'))
        else:
            rgb_images.append(img.copy())
    
    if layout == "horizontal":
        total_width = sum(img.width for img in rgb_images) + spacing * (len(rgb_images) - 1)
        max_height = max(img.height for img in rgb_images)
        
        combined = Image.new('RGB', (total_width, max_height), background_color)
        
        x_offset = 0
        for img in rgb_images:
            y_offset = (max_height - img.height) // 2
            combined.paste(img, (x_offset, y_offset))
            x_offset += img.width + spacing
            
    elif layout == "vertical":
        max_width = max(img.width for img in rgb_images)
        total_height = sum(img.height for img in rgb_images) + spacing * (len(rgb_images) - 1)
        
        combined = Image.new('RGB', (max_width, total_height), background_color)
        
        y_offset = 0
        for img in rgb_images:
            x_offset = (max_width - img.width) // 2
            combined.paste(img, (x_offset, y_offset))
            y_offset += img.height + spacing
    else:
        raise ValueError(f"Invalid layout '{layout}'. Must be 'horizontal' or 'vertical'")
    
    return combined

def compress_image(image: Union[str, Image.Image], max_size: Tuple[int, int] = (512,512), quality: int = 80) -> Image.Image:
    """
    Compress image to reduce file size
    
    Args:
        image: PIL Image object or path to image file
        max_size: Maximum size (width, height)
        quality: JPEG quality (1-100)
    
    Returns:
        Compressed PIL Image
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    # Convert to RGB if necessary
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    
    # Resize if larger than max_size
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Compress by saving to BytesIO with specified quality
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=quality, optimize=True)
    buffer.seek(0)
    
    return Image.open(buffer)

def encode_image_to_base64(image: Union[str, Image.Image], max_size: Tuple[int, int] = (512,512), quality: int = 80, auto_compress: bool = False) -> str:
    """
    Encode image to base64 string
    
    Args:
        image: PIL Image object or path to image file
        max_size: Maximum size for compression
        quality: JPEG quality for compression
        auto_compress: Whether to auto-compress large images
    
    Returns:
        Base64 encoded string
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    # Auto-compress if requested and image is large
    if auto_compress and (image.width > max_size[0] or image.height > max_size[1]):
        image = compress_image(image, max_size, quality)
    
    # Handle transparency
    if _has_transparency(image):
        # Keep as PNG for transparency
        buffer = BytesIO()
        image.save(buffer, format='PNG')
    else:
        # Convert to RGB and save as JPEG
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
    
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def _has_transparency(image: Image.Image) -> bool:
    """Check if image has transparency"""
    return (
        image.mode in ('RGBA', 'LA') or
        (image.mode == 'P' and 'transparency' in image.info)
    )

def get_qwenvl_info(messages: List[Dict[str, Any]], processor: AutoProcessor) -> Dict[str, Any]:
    """
    Get the meta information from the messages.
    """

    image_inputs, video_inputs = process_vision_info(messages)  
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text = [text],
        images = image_inputs,
        videos = video_inputs,
        padding=True,
        return_tensors="pt",
    )
    image_info = {
        "height": inputs['image_grid_thw'][0][1]*14,
        "width": inputs['image_grid_thw'][0][2]*14,
        "image": image_inputs[0]
    }
    return image_info