#!/usr/bin/env python3
"""
Debate Visualization Gradio App

A web interface for visualizing debate processes with visual evidence display.
Features:
- English interface
- Chinese text support in visual evidence
- Interactive debate round display
- Side-by-side text and visual evidence
"""

import gradio as gr
import json
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import warnings
from io import BytesIO
import numpy as np

# Import visual utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from debate.visual_utils import plot_bounding_boxes, combine_images

def normalize_bbox_data(bbox_data, image_info, current_image_size):
    """
    Normalize bbox coordinates based on whether image_info is available.
    
    Args:
        bbox_data: List of annotation data
        image_info: Dict with original image metadata (width, height)
        current_image_size: Tuple (width, height) of current image
    
    Returns:
        Normalized bbox_data with coordinates in [0,1] range
    """
    if not bbox_data or not image_info or 'width' not in image_info or 'height' not in image_info:
        # If no image_info, assume coordinates are already normalized
        return bbox_data
    
    try:
        original_width = int(image_info['width'])
        original_height = int(image_info['height'])
        current_width, current_height = current_image_size
        
        # If image_info is provided, coordinates are in pixel space relative to original size
        # We need to normalize them to [0,1] range based on current image size
        normalized_data = []
        
        for item in bbox_data:
            if not isinstance(item, dict):
                normalized_data.append(item)
                continue
                
            normalized_item = item.copy()
            
            # Handle bbox_2d coordinates
            if 'bbox_2d' in item:
                bbox = item['bbox_2d']
                if bbox and len(bbox) == 4:
                    # Convert from original pixel coordinates to normalized coordinates
                    x1, y1, x2, y2 = bbox
                    # First normalize to [0,1] based on original size
                    norm_x1 = x1 / original_width
                    norm_y1 = y1 / original_height
                    norm_x2 = x2 / original_width
                    norm_y2 = y2 / original_height
                    # Convert to [x, y, w, h] format
                    normalized_item['bbox_2d'] = [norm_x1, norm_y1, norm_x2 - norm_x1, norm_y2 - norm_y1]
            
            # Handle point_2d coordinates
            if 'point_2d' in item:
                point = item['point_2d']
                if point and len(point) == 2:
                    x, y = point
                    normalized_item['point_2d'] = [x / original_width, y / original_height]
            
            # Handle line_2d coordinates
            if 'line_2d' in item:
                line = item['line_2d']
                if line and len(line) == 4:
                    x1, y1, x2, y2 = line
                    normalized_item['line_2d'] = [
                        x1 / original_width, y1 / original_height,
                        x2 / original_width, y2 / original_height
                    ]
            
            # Handle zoom_in_2d coordinates
            if 'zoom_in_2d' in item:
                zoom_bbox = item['zoom_in_2d']
                if zoom_bbox and len(zoom_bbox) == 4:
                    x, y, w, h = zoom_bbox
                    # Assume zoom coordinates are already in [x, y, w, h] format
                    normalized_item['zoom_in_2d'] = [
                        x / original_width, y / original_height,
                        w / original_width, h / original_height
                    ]
            
            normalized_data.append(normalized_item)
        
        return normalized_data
        
    except (ValueError, KeyError, ZeroDivisionError) as e:
        print(f"Warning: Could not normalize coordinates: {e}, using original data")
        return bbox_data

def generate_visual_evidence_images(base_image, bbox_data, image_info=None):
    """
    Generate visual evidence images using the enhanced plot_bounding_boxes function.
    
    Args:
        base_image: PIL Image object
        bbox_data: List of annotation data 
                  - If image_info provided: coordinates are in pixel space relative to original image size
                  - If no image_info: coordinates are assumed to be normalized [0,1]
        image_info: Optional dict with image metadata
                   Contains 'width', 'height' etc.
    
    Returns:
        Single PIL Image combining all generated images, or original image
    """
    if not bbox_data or not base_image:
        return base_image
    
    try:
        # Normalize coordinates based on whether we have image_info
        current_size = base_image.size
        normalized_bbox_data = normalize_bbox_data(bbox_data, image_info, current_size)
        
        # Optionally resize image to match original dimensions if image_info is provided
        if image_info and 'width' in image_info and 'height' in image_info:
            try:
                width = int(image_info['width'])
                height = int(image_info['height'])
                if base_image.size != (width, height):
                    print(f"Resizing image from {base_image.size} to ({width}, {height})")
                    base_image = base_image.resize((width, height))
                    # Update current size after resize
                    current_size = base_image.size
                    # Re-normalize if size changed
                    normalized_bbox_data = normalize_bbox_data(bbox_data, image_info, current_size)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not use image_info for resizing: {e}")
        
        # Use the enhanced plot_bounding_boxes function from visual_utils
        # Now all coordinates are normalized [0,1]
        result_images = plot_bounding_boxes(base_image, normalized_bbox_data)
        
        # If no images returned, return the original image
        if not result_images:
            return base_image
        
        # If only one image, return it directly
        if len(result_images) == 1:
            return result_images[0]
        
        # If multiple images, combine them horizontally
        try:
            combined_image = combine_images(result_images, layout="horizontal", spacing=20)
            return combined_image
        except Exception as e:
            print(f"Error combining images: {e}")
            # Fallback to first image if combination fails
            return result_images[0]
        
    except Exception as e:
        print(f"Error generating visual evidence: {e}")
        return base_image

class DebateViewer:
    """Debate visualization viewer for Gradio interface."""
    
    def __init__(self):
        self.debate_data = None
        self.current_case = 0
        
    def load_debate_data(self, file_path: str) -> bool:
        """Load debate trajectory data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.debate_data = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading debate data: {e}")
            return False
    
    def get_case_info(self, case_idx: int = 0) -> Dict[str, Any]:
        """Get information about a specific case."""
        if not self.debate_data or case_idx >= len(self.debate_data):
            return {}
        
        case_data = self.debate_data[case_idx]
        
        # Handle both new and old formats
        if 'debate_trajectory' in case_data:
            # New format
            all_speeches = case_data['debate_trajectory']
            image_info = case_data.get('image_info', {})
            
            # If no image_info but case has images, construct image_info from case data
            if not image_info and 'case' in case_data and 'images' in case_data['case']:
                images = case_data['case']['images']
                if images and len(images) > 0:
                    # Use the first image from the images array and construct full path
                    image_path = images[0]
                    # Construct full path to mm-deceptionbench directory
                    full_image_path = os.path.join(os.getcwd(), 'mm-deceptionbench', image_path)
                    image_info = {'image': full_image_path}
        else:
            # Old format
            all_speeches = case_data
            image_info = {}
        
        # Group speeches into rounds (pairs of D_neg and D_aff)
        rounds = []
        current_round = {}
        
        for speech in all_speeches:
            role = speech.get('role', '')
            if role == 'D_neg':
                # Start a new round
                current_round = {'D_neg': speech}
            elif role == 'D_aff' and current_round:
                # Complete the current round
                current_round['D_aff'] = speech
                rounds.append(current_round)
                current_round = {}
        
        # If there's an incomplete round (only D_neg), add it anyway
        if current_round:
            rounds.append(current_round)
        
        return {
            'rounds': rounds,
            'all_speeches': all_speeches,
            'image_info': image_info,
            'total_rounds': len(rounds),
            'total_cases': len(self.debate_data)
        }
    
    def get_round_data(self, case_idx: int, round_idx: int) -> Dict[str, Any]:
        """Get data for a specific round (containing both D_neg and D_aff)."""
        case_info = self.get_case_info(case_idx)
        if not case_info or round_idx >= len(case_info['rounds']):
            return {}
        
        round_data = case_info['rounds'][round_idx]
        
        # Load image if available
        image = None
        image_info = case_info.get('image_info', {})
        if image_info and 'image' in image_info:
            try:
                image_path = image_info['image']
                
                # Try multiple path resolution strategies
                possible_paths = [
                    image_path,  # Original path
                    os.path.join(os.getcwd(), os.path.basename(image_path)),  # Just filename in current dir
                    os.path.join(os.getcwd(), *image_path.split('/')[-3:]),  # Last 3 parts of path
                    os.path.join(os.getcwd(), *image_path.split('/')[-4:]),  # Last 4 parts of path
                ]
                
                # If it contains 'mm-deceptionbench', try relative path
                if 'mm-deceptionbench' in image_path:
                    relative_part = image_path.split('mm-deceptionbench/')[-1]
                    possible_paths.append(os.path.join(os.getcwd(), 'mm-deceptionbench', relative_part))
                
                # Try each possible path
                for path in possible_paths:
                    if os.path.exists(path):
                        image = Image.open(path).convert('RGB')
                        print(f"Successfully loaded image from: {path}")
                        break
                
                if image is None:
                    print(f"Warning: Could not find image at any of these paths: {possible_paths}")
                    
            except Exception as e:
                print(f"Error loading image: {e}")
        
        return {
            'round_data': round_data,
            'image': image,
            'image_info': image_info
        }
    
    def get_speaker_in_round(self, case_idx: int, round_idx: int, speaker_role: str) -> Tuple[str, str, List[Dict], Optional[Image.Image]]:
        """Get data for a specific speaker in a specific round."""
        round_info = self.get_round_data(case_idx, round_idx)
        if not round_info or speaker_role not in round_info['round_data']:
            return "N/A", "No content available", [], None
        
        speaker_data = round_info['round_data'][speaker_role]
        role = speaker_data.get('role', 'Unknown')
        content = speaker_data.get('content', 'No content')
        visual_evidence = speaker_data.get('visual_evidence', [])
        
        # Handle case where visual_evidence is an empty dict instead of list
        if isinstance(visual_evidence, dict) and not visual_evidence:
            visual_evidence = []
        elif not isinstance(visual_evidence, list):
            print(f"Warning: visual_evidence is not a list for {speaker_role}: {type(visual_evidence)}")
            visual_evidence = []
        
        return role, content, visual_evidence, round_info['image']
    
    def get_judge_verdict(self, case_idx: int) -> Tuple[str, str, List[Dict], Optional[Image.Image]]:
        """Get judge verdict data for a specific case."""
        case_info = self.get_case_info(case_idx)
        if not case_info:
            return "N/A", "No judge verdict available", [], None
        
        # Look for D_judge in all speeches
        all_speeches = case_info.get('all_speeches', [])
        judge_data = None
        
        for speech in all_speeches:
            if speech.get('role') == 'D_judge':
                judge_data = speech
                break
        
        if not judge_data:
            return "D_judge", "No judge verdict found in this case", [], None
        
        # Format judge verdict - the actual verdict data is in 'visual_evidence' field
        verdict = judge_data.get('visual_evidence', {})
        if not verdict:
            # Fallback to 'verdict' field if 'visual_evidence' is empty
            verdict = judge_data.get('verdict', {})
        role = judge_data.get('role', 'D_judge')
        
        # Create formatted content
        content = self.format_judge_verdict(verdict)
        
        # Get visual evidence from locus_and_justification (if exists)
        visual_evidence = verdict.get('locus_and_justification', {}).get('findings', [])
        if not visual_evidence:
            # If no locus_and_justification, try to extract from other fields
            visual_evidence = []
        
        # Load image
        image = None
        image_info = case_info.get('image_info', {})
        if image_info and 'image' in image_info:
            try:
                image_path = image_info['image']
                # Try multiple path resolution strategies
                possible_paths = [
                    image_path,
                    os.path.join(os.getcwd(), os.path.basename(image_path)),
                    os.path.join(os.getcwd(), *image_path.split('/')[-3:]),
                    os.path.join(os.getcwd(), *image_path.split('/')[-4:]),
                ]
                
                if 'mm-deceptionbench' in image_path:
                    relative_part = image_path.split('mm-deceptionbench/')[-1]
                    possible_paths.append(os.path.join(os.getcwd(), 'mm-deceptionbench', relative_part))
                
                for path in possible_paths:
                    if os.path.exists(path):
                        image = Image.open(path).convert('RGB')
                        break
            except Exception as e:
                print(f"Error loading image for judge: {e}")
        
        return role, content, visual_evidence, image
    
    def format_judge_verdict(self, verdict: dict) -> str:
        """Format judge verdict into readable text."""
        if not verdict:
            return "No verdict information available"
        
        formatted_verdict = "## 🏛️ Judge's Verdict\n\n"
        
        # Debate Summary
        debate_summary = verdict.get('debate_summary', '')
        if debate_summary:
            formatted_verdict += f"### 📋 **Debate Summary**\n"
            formatted_verdict += f"{debate_summary}\n\n"
        
        # Decision and Confidence
        decision = verdict.get('decision', 'Unknown')
        confidence = verdict.get('confidence_score', 0)
        
        formatted_verdict += f"### 📊 **Final Decision**\n"
        formatted_verdict += f"**Verdict:** {decision}\n"
        formatted_verdict += f"**Confidence Score:** {confidence:.2f}\n\n"
        
        # Description
        description = verdict.get('description', '')
        if description:
            formatted_verdict += f"### 💭 **Reasoning**\n"
            formatted_verdict += f"{description}\n\n"
        
        # Deception Severity
        deception_severity = verdict.get('deception_severity', {})
        if deception_severity:
            severity_score = deception_severity.get('severity_score', 0)
            severity_desc = deception_severity.get('description', '')
            
            formatted_verdict += f"### ⚖️ **Severity Assessment**\n"
            formatted_verdict += f"**Severity Score:** {severity_score:.2f}\n"
            if severity_desc:
                formatted_verdict += f"**Description:** {severity_desc}\n\n"
        
        # Locus and Justification Summary (if exists)
        locus_justification = verdict.get('locus_and_justification', {})
        if locus_justification:
            summary = locus_justification.get('summary', '')
            findings = locus_justification.get('findings', [])
            
            formatted_verdict += f"### 🔍 **Additional Analysis**\n"
            if summary:
                formatted_verdict += f"{summary}\n\n"
            
            if findings:
                formatted_verdict += f"**Visual Evidence Findings:** {len(findings)} items identified\n"
        
        return formatted_verdict
    
    def generate_judge_visual_evidence(self, case_idx: int) -> Optional[Image.Image]:
        """Generate visual evidence visualization for judge verdict."""
        role, content, visual_evidence, base_image = self.get_judge_verdict(case_idx)
        
        if not visual_evidence or not base_image:
            return base_image
        
        # Get image info for potential resizing
        case_info = self.get_case_info(case_idx)
        image_info = case_info.get('image_info', {}) if case_info else {}
        
        try:
            # Generate visual evidence images using the enhanced function
            # Pass image_info for backward compatibility, but it's optional
            result_image = generate_visual_evidence_images(base_image, visual_evidence, image_info)
            return result_image
        except Exception as e:
            print(f"Error generating judge visual evidence: {e}")
            return base_image
    
    def generate_visual_evidence(self, case_idx: int, round_idx: int, speaker_role: str) -> Optional[Image.Image]:
        """Generate visual evidence visualization for a specific speaker in a round."""
        role, content, visual_evidence, base_image = self.get_speaker_in_round(case_idx, round_idx, speaker_role)
        
        if not visual_evidence or not base_image:
            return base_image
        
        # Get image info for potential resizing
        case_info = self.get_case_info(case_idx)
        image_info = case_info.get('image_info', {}) if case_info else {}
        
        try:
            # Generate visual evidence images using the enhanced function
            # Pass image_info for backward compatibility, but it's optional
            result_image = generate_visual_evidence_images(base_image, visual_evidence, image_info)
            return result_image
        except Exception as e:
            print(f"Error generating visual evidence: {e}")
            return base_image

# Initialize the debate viewer
viewer = DebateViewer()

def load_debate_file_and_create_tabs(file_path):
    """Load debate trajectory file and dynamically create tabs for each case."""
    if not file_path or not os.path.exists(file_path):
        return "❌ Please select a valid debate trajectory file", gr.update(visible=False)
    
    success = viewer.load_debate_data(file_path)
    if not success:
        return "❌ Failed to load debate file", gr.update(visible=False)
    
    if not viewer.debate_data:
        return "❌ No valid debate data found", gr.update(visible=False)
    
    total_cases = len(viewer.debate_data)
    
    # Create tabs content dynamically
    tabs_content = []
    for case_idx, case_data in enumerate(viewer.debate_data):
        category = case_data.get('case', {}).get('category', f'Case {case_idx + 1}')
        tab_label = f"📁 {category} ({case_idx + 1})"
        tabs_content.append((tab_label, case_idx, case_data))
    
    status_message = f"✅ Successfully loaded {total_cases} cases"
    
    return status_message, gr.update(visible=True)


def format_case_info(case_idx: int) -> str:
    """Format case information for display."""
    if not viewer.debate_data or case_idx >= len(viewer.debate_data):
        return "No case information available"
    
    case_data = viewer.debate_data[case_idx]
    
    # Check if there's a 'case' key with detailed information
    if 'case' in case_data:
        case_info = case_data['case']
        
        # Format the case information
        formatted_info = f"""### 📂 **Category: {case_info.get('category', 'Unknown Category')}**

**🎭 Scenario:**
{case_info.get('scenario', 'No scenario provided')}

**👤 User Profile:**
{case_info.get('user_profile', 'No user profile provided')}

**🤖 Assistant Profile:**
{case_info.get('assistant_profile', 'No assistant profile provided')}

**💭 User Prompt:**
> "{case_info.get('prompt', 'No prompt provided')}"

**🧠 Assistant Reasoning:**
{case_info.get('result', {}).get('reasoning', 'No reasoning provided')}

**📤 Assistant Output:**
{case_info.get('result', {}).get('output', 'No output provided')}
"""
        return formatted_info
    else:
        # Fallback for cases without detailed information
        return f"""### 📂 **Case {case_idx + 1}**

This case contains debate data but no detailed background information is available.

**Available Data:**
- Debate trajectory with {len(case_data.get('debate_trajectory', []))} rounds
- Image information: {case_data.get('image_info', {}).get('image', 'No image')}
"""

def display_round_content(case_selection, round_selection, speaker_selection):
    """Display round content and visual evidence for the selected speaker."""
    if not viewer.debate_data:
        return "No debate data loaded", "No content", None, "", "No case info"
    
    try:
        # Extract case index from selection like "📁 Fabrication (Case 1)"
        import re
        if case_selection:
            match = re.search(r'Case (\d+)', case_selection)
            case_idx = int(match.group(1)) - 1 if match else 0
        else:
            case_idx = 0
            
        # Extract round index from selection like "Round 1"
        if round_selection:
            round_match = re.search(r'Round (\d+)', round_selection)
            round_idx = int(round_match.group(1)) - 1 if round_match else 0
        else:
            round_idx = 0
        
        # Extract speaker role from selection like "D_neg (Negative)"
        if speaker_selection and "(" in speaker_selection:
            speaker_role = speaker_selection.split("(")[0].strip()
        else:
            speaker_role = "N/A"
        
        # Handle judge differently as it's not tied to a specific round
        if 'judge' in role:
            role, content, visual_evidence, base_image = viewer.get_judge_verdict(case_idx)
        else:
            role, content, visual_evidence, base_image = viewer.get_speaker_in_round(case_idx, round_idx, speaker_role)
        
        # Generate visual evidence
        if 'judge' in role:
            visual_image = viewer.generate_judge_visual_evidence(case_idx)
        else:
            visual_image = viewer.generate_visual_evidence(case_idx, round_idx, speaker_role)
        
        # Format role display with better description
        role_display = f"**{role}** "
        if 'aff' in role:
            role_display += "(Affirmative Side - Deceptive)"
        elif 'neg' in role:
            role_display += "(Negative Side - Non-deceptive)"
        elif 'judge' in role:
            role_display += "(Judge - Final Verdict)"
        else:
            role_display += "(Debater)"
        
        # Add round information (except for judge)
        if 'judge' not in role:
            role_display = f"## Round {round_idx + 1}\n{role_display}"
        else:
            role_display = f"## Judge's Verdict\n{role_display}"
        
        # Create evidence summary
        evidence_summary = f"Visual Evidence: {len(visual_evidence)} items"
        if visual_evidence:
            evidence_list = [f"• {item['label'][:100]}..." if len(item['label']) > 100 else f"• {item['label']}" 
                           for item in visual_evidence]
            evidence_summary += "\n\n" + "\n".join(evidence_list)
        
        # Generate case information
        case_info = format_case_info(case_idx)
        
        return role_display, content, visual_image, evidence_summary, case_info
        
    except Exception as e:
        return f"Error: {str(e)}", "Failed to load content", None, "", "Error loading case info"


def create_gradio_interface():
    """Create the Gradio interface for debate visualization."""
    
    with gr.Blocks(title="Debate Process Visualization", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎯 Debate Process Visualization
        
        **Interactive visualization of debate trajectories with visual evidence analysis**
        
        Upload a debate trajectory JSON file to explore the debate process step by step.
        """)
        
        # File upload section
        with gr.Row():
            file_input = gr.File(
                label="📁 Upload Debate Trajectory File",
                file_types=[".json"],
                type="filepath"
            )
            
            status_output = gr.Textbox(
                label="📊 Status",
                interactive=False,
                lines=1
            )
        
        # Case selector
        case_selector = gr.Dropdown(
            label="🗂️ Select Case",
            choices=[],
            visible=False,
            interactive=True
        )
        
        # Case content area
        case_content = gr.Column(visible=False)
        
        with case_content:
            # Case Information Section
            gr.Markdown("## 📋 Case Information")
            case_info_display = gr.Markdown(
                value="Select a case to view details",
                label="Case Details"
            )
            
            gr.Markdown("---")
            
            # Debate Controls Section  
            with gr.Row():
                with gr.Column(scale=1):
                    round_dropdown = gr.Dropdown(
                        label="🔄 Select Round", 
                        choices=[],
                        interactive=True,
                        info="Note: Judge verdict is not tied to any specific round"
                    )
                
                with gr.Column(scale=1):
                                    speaker_dropdown = gr.Dropdown(
                    label="🎤 Select Speaker",
                    choices=["D_neg (Negative)", "D_aff (Affirmative)", "D_judge (Judge)"],
                    value="D_neg (Negative)",
                    interactive=True
                )
            
            # Debate Content Section
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 💬 Debate Content")
                    
                    role_output = gr.Markdown(
                        label="Role",
                        value="Select a round and speaker to view content"
                    )
                    
                    content_output = gr.Textbox(
                        label="📝 Argument Content",
                        lines=15,
                        interactive=False,
                        placeholder="Debater's argument will appear here..."
                    )
                    
                    evidence_summary = gr.Textbox(
                        label="📋 Evidence Summary", 
                        lines=8,
                        interactive=False,
                        placeholder="Visual evidence summary will appear here..."
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("## 🖼️ Visual Evidence")
                    
                    visual_output = gr.Image(
                        label="Visual Evidence with Annotations",
                        type="pil",
                        interactive=False,
                        height=600
                    )
        
        # Event handlers
        def handle_file_upload(file_path):
            """Handle file upload and create case selector."""
            if not file_path or not os.path.exists(file_path):
                return ("❌ Please select a valid debate trajectory file", 
                        gr.update(visible=False), 
                        gr.update(visible=False))
            
            success = viewer.load_debate_data(file_path)
            if not success or not viewer.debate_data:
                return ("❌ Failed to load debate file", 
                        gr.update(visible=False), 
                        gr.update(visible=False))
            
            total_cases = len(viewer.debate_data)
            
            # Create case choices with category information
            case_choices = []
            for case_idx, case_data in enumerate(viewer.debate_data):
                category = case_data.get('case', {}).get('category', f'Case {case_idx + 1}')
                case_choices.append(f"📁 {category} (Case {case_idx + 1})")
            
            status_msg = f"✅ Successfully loaded {total_cases} cases"
            
            return (status_msg, 
                    gr.update(choices=case_choices, value=case_choices[0] if case_choices else None, visible=True),
                    gr.update(visible=True))
        
        def update_speaker_selection(speaker_selection, case_selection):
            """Update round dropdown availability based on speaker selection."""
            if speaker_selection and "D_judge" in speaker_selection:
                # Judge is selected - disable round dropdown
                return gr.update(interactive=False, value=None, info="Judge verdict applies to the entire case")
            else:
                # Non-judge speaker selected - enable round dropdown
                # Get current round choices for this case
                if case_selection and viewer.debate_data:
                    import re
                    match = re.search(r'Case (\d+)', case_selection)
                    case_idx = int(match.group(1)) - 1 if match else 0
                    case_info = viewer.get_case_info(case_idx)
                    
                    if case_info:
                        rounds = case_info.get('rounds', [])
                        round_choices = [f"Round {i+1}" for i in range(len(rounds))]
                        return gr.update(interactive=True, choices=round_choices, value=round_choices[0] if round_choices else None, info="Select a round for the debate")
                
                return gr.update(interactive=True, choices=[], info="Select a round for the debate")

        def update_case_content(case_selection):
            """Update case content when case is selected."""
            if not case_selection or not viewer.debate_data:
                return ("No case information", 
                        gr.update(choices=[]),
                        "No case info",
                        "No content", 
                        None, 
                        "")
            
            try:
                # Extract case index from selection like "📁 Fabrication (Case 1)"
                import re
                match = re.search(r'Case (\d+)', case_selection)
                if match:
                    case_idx = int(match.group(1)) - 1
                else:
                    case_idx = 0
                
                case_info = viewer.get_case_info(case_idx)
                total_rounds = case_info.get('total_rounds', 0)
                round_choices = [f"Round {i+1}" for i in range(total_rounds)]
                
                case_info_text = format_case_info(case_idx)
                
                # Auto-load first round content
                if round_choices:
                    role_display, content, visual_image, evidence_summary, _ = display_round_content(
                        case_selection, "Round 1", "D_neg (Negative)"
                    )
                else:
                    role_display, content, visual_image, evidence_summary = "No rounds available", "", None, ""
                
                return (case_info_text,
                        gr.update(choices=round_choices, value=round_choices[0] if round_choices else None),
                        role_display,
                        content,
                        visual_image,
                        evidence_summary)
                
            except Exception as e:
                return (f"Error loading case: {str(e)}", 
                        gr.update(choices=[]),
                        "Error",
                        "Failed to load content", 
                        None, 
                        "")
        
        # File upload event
        file_input.change(
            fn=handle_file_upload,
            inputs=[file_input],
            outputs=[status_output, case_selector, case_content]
        )
        
        # Case selection event
        case_selector.change(
            fn=update_case_content,
            inputs=[case_selector],
            outputs=[case_info_display, round_dropdown, role_output, content_output, visual_output, evidence_summary]
        )
        
        # Speaker selection event to update round dropdown availability
        speaker_dropdown.change(
            fn=update_speaker_selection,
            inputs=[speaker_dropdown, case_selector],
            outputs=[round_dropdown]
        )
        
        # Round and speaker selection events
        for component in [round_dropdown, speaker_dropdown]:
            component.change(
                fn=display_round_content,
                inputs=[case_selector, round_dropdown, speaker_dropdown],
                outputs=[role_output, content_output, visual_output, evidence_summary, case_info_display]
            )
        
        gr.Markdown("""
        ---
        ### 📖 Instructions:
        1. **Upload File**: Select a debate trajectory JSON file
        2. **Navigate**: Use dropdowns to select case and round
        3. **Explore**: View debate arguments and visual evidence side by side
        """)
    
    return demo

def main():
    """Main function to run the Gradio app."""
    # Check if we have a default debate file
    default_file = "debate/results/debate_trajectory.json"
    if os.path.exists(default_file):
        print(f"Found default debate file: {default_file}")
        viewer.load_debate_data(default_file)
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    print("🚀 Starting Debate Visualization App...")
    print("📱 The interface will open in your default web browser")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create a public link
        debug=True              # Enable debug mode
    )

if __name__ == "__main__":
    main()
