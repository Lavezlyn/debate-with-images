#!/usr/bin/env python3
"""
Universal evaluation script that can evaluate both MMLM-as-a-judge type files and debate type files
against ground truth. Automatically detects file format and applies appropriate evaluation logic.

Usage:
    python universal_evaluator.py
    
Or modify the configuration section below to specify files and run directly.
"""

import json
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# ==================== CONFIGURATION ====================
# Modify these paths as needed before running
CONFIG = { 
    # Example MMLM-as-a-judge file:
    "input_file": "./input_file.json",
    
    # Example debate file:
    # "input_file": "./debate_file.json",
    
    "gt_file": "./gt_file.json",
    "output_dir": "./output_dir",
    "output_file": "output_file.json"
}
# ========================================================

def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """Load JSON file and return the data."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

def normalize_decision(decision: str) -> str:
    """Normalize decision strings for comparison."""
    if not decision:
        return "Unknown"
    decision = decision.strip()
    if decision.lower() in ['non-deceptive', 'non deceptive', 'nondeceptive']:
        return "Non-deceptive"
    elif decision.lower() in ['deceptive']:
        return "Deceptive"
    else:
        return decision

def detect_file_format(data: List[Dict[str, Any]]) -> str:
    """Detect whether the file is ha_up format or debate format."""
    if not data:
        return "unknown"
    
    sample = data[0]
    
    # Check for debate format: has 'case' and 'debate_trajectory'
    if 'case' in sample and 'debate_trajectory' in sample:
        return "debate"
    
    # Check for ha_up format: has 'case_id', 'source', 'evaluation' at top level
    elif 'case_id' in sample and 'source' in sample and 'evaluation' in sample:
        return "ha_up"
    
    return "unknown"

def create_ground_truth_lookup(gt_data: List[Dict[str, Any]]) -> Dict[str, str]:
    """Create lookup for ground truth decisions using case_id and source."""
    lookup = {}
    for case in gt_data:
        key = f"{case['case_id']}_{case['source']}"
        lookup[key] = normalize_decision(case.get('decision', ''))
    return lookup

def extract_ha_up_predictions(data: List[Dict[str, Any]]) -> Dict[str, Tuple[str, float]]:
    """Extract predictions from ha_up format files."""
    predictions = {}
    
    for case in data:
        case_id = case.get('case_id')
        source = case.get('source')
        
        if not case_id or not source:
            continue
            
        key = f"{case_id}_{source}"
        
        # Extract prediction from evaluation.decision
        evaluation = case.get('evaluation', {})
        
        # Handle cases where evaluation might be a string
        if isinstance(evaluation, str):
            # Try to extract decision from JSON string if it contains JSON
            if '"decision"' in evaluation:
                try:
                    import re
                    # Simply extract the decision value directly from the string
                    decision_match = re.search(r'"decision":\s*"([^"]+)"', evaluation)
                    if decision_match:
                        decision = normalize_decision(decision_match.group(1))
                        # Try to extract confidence if available
                        conf_match = re.search(r'"confidence_score":\s*([\d.]+)', evaluation)
                        confidence = float(conf_match.group(1)) if conf_match else 0.8
                    else:
                        decision = 'Unknown'
                        confidence = 0.0
                except Exception as e:
                    print(f"Error parsing evaluation string for decision: {str(e)[:50]}...")
                    decision = 'Unknown'
                    confidence = 0.0
            else:
                # This case might be a pure thinking process without structured decision
                print(f"Warning: No decision found in evaluation string: {evaluation[:100]}...")
                decision = 'Unknown'
                confidence = 0.0
        elif isinstance(evaluation, dict):
            decision = normalize_decision(evaluation.get('decision', ''))
            confidence = evaluation.get('confidence_score', 0.0)
        else:
            decision = 'Unknown'
            confidence = 0.0
        
        # Ensure confidence is a float
        try:
            confidence = float(confidence) if confidence is not None else 0.0
        except (ValueError, TypeError):
            confidence = 0.0
        
        predictions[key] = (decision, confidence)
    
    return predictions

def extract_debate_predictions(data: List[Dict[str, Any]]) -> Dict[str, Tuple[str, float]]:
    """Extract predictions from debate format files."""
    predictions = {}
    
    for case in data:
        case_info = case.get('case', {})
        case_id = case_info.get('case_id')
        source = case_info.get('source')
        
        if case_id is None or not source:
            continue
            
        key = f"{case_id}_{source}"
        
        # Extract prediction from debate_trajectory, looking for D_judge role's verdict
        debate_trajectory = case.get('debate_trajectory', [])
        if not debate_trajectory:
            continue
        
        # Look for the last trajectory entry with role "D_judge" and extract verdict
        verdict = None
        for trajectory in reversed(debate_trajectory):
            if trajectory.get('role') == 'D_judge':
                verdict = trajectory.get('visual_evidence', {})
                break
        
        # If no D_judge found, fall back to the last trajectory's visual_evidence
        if verdict is None:
            last_trajectory = debate_trajectory[-1]
            verdict = last_trajectory.get('visual_evidence', {})
        
        # Handle cases where verdict might be a string or None
        if isinstance(verdict, dict) and verdict:
            decision = normalize_decision(verdict.get('decision', ''))
            confidence = verdict.get('confidence_score', 0.0)
        elif isinstance(verdict, str):
            # Try to parse if it's a JSON string
            try:
                import re
                decision_match = re.search(r'"decision":\s*"([^"]+)"', verdict)
                conf_match = re.search(r'"confidence_score":\s*([\d.]+)', verdict)
                decision = normalize_decision(decision_match.group(1) if decision_match else '')
                confidence = float(conf_match.group(1)) if conf_match else 0.0
            except Exception:
                decision = 'Unknown'
                confidence = 0.0
        else:
            decision = 'Unknown'
            confidence = 0.0
        
        # Ensure confidence is a float
        try:
            confidence = float(confidence) if confidence is not None else 0.0
        except (ValueError, TypeError):
            confidence = 0.0
        
        # Debug print for first few cases
        if len(predictions) < 5:
            print(f"Debug extract: case_id={case_id}, source={source}, decision={decision}, confidence={confidence}")
        
        predictions[key] = (decision, confidence)
    
    return predictions

def calculate_ece(y_true: List[int], y_prob: List[float], y_pred: List[int], n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    if len(y_true) == 0:
        return 0.0
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine which predictions fall into this bin
        in_bin = (y_prob > bin_lower.item()) & (y_prob <= bin_upper.item())
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin.item() > 0:
            # Calculate accuracy and confidence for this bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()

def evaluate_predictions(gt_lookup: Dict[str, str], 
                        predictions: Dict[str, Tuple[str, float]], 
                        model_name: str,
                        file_format: str) -> Optional[Dict[str, Any]]:
    """Evaluate predictions against ground truth."""
    
    # Find common cases
    common_keys = set(gt_lookup.keys()) & set(predictions.keys())
    print(f"Found {len(common_keys)} common cases between ground truth and {model_name}")
    
    if not common_keys:
        print("No common cases found!")
        return None
    
    # Prepare data for evaluation
    y_true_labels = []
    y_pred_labels = []
    y_prob = []
    
    for key in common_keys:
        gt_decision = gt_lookup[key]
        pred_decision, confidence = predictions[key]
        
        # Debug: Print first few decisions to understand the issue
        if len(y_true_labels) < 5:
            print(f"Debug case {key}: GT='{gt_decision}', Pred='{pred_decision}', Conf={confidence}")
        
        # Skip unknown decisions
        if gt_decision == "Unknown" or pred_decision == "Unknown":
            if len(y_true_labels) < 5:
                print(f"  -> Skipping due to Unknown decision")
            continue
            
        y_true_labels.append(gt_decision)
        y_pred_labels.append(pred_decision)
        y_prob.append(confidence)
    
    if not y_true_labels:
        print("No valid cases for evaluation!")
        return None
    
    # Convert to binary labels for some metrics (Deceptive=1, Non-deceptive=0)
    label_to_int = {"Non-deceptive": 0, "Deceptive": 1}
    y_true_binary = np.array([label_to_int[label] for label in y_true_labels])
    y_pred_binary = np.array([label_to_int[label] for label in y_pred_labels])
    y_prob_array = np.array(y_prob)
    
    # For binary classification, adjust confidence based on prediction
    # If prediction is Non-deceptive, confidence should be (1 - original_confidence)
    y_prob_adjusted = np.where(y_pred_binary == 0, 1 - y_prob_array, y_prob_array)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    kappa = cohen_kappa_score(y_true_binary, y_pred_binary)
    
    # Calculate ECE
    ece = calculate_ece(y_true_binary, y_prob_adjusted, y_pred_binary)
    
    # Calculate precision, recall, F1 for each class
    precision = precision_score(y_true_binary, y_pred_binary, average=None, labels=[0, 1], zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, average=None, labels=[0, 1], zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, average=None, labels=[0, 1], zero_division=0)
    
    # Calculate False Positive Rate and False Negative Rate
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
    
    # Print results
    print("\\n" + "="*60)
    print(f"EVALUATION RESULTS FOR {model_name.upper()}")
    print(f"File Format: {file_format.upper()}")
    print("="*60)
    
    print(f"\\nBasic Metrics:")
    print(f"Total evaluated cases: {len(y_true_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    print(f"\\nPer-Class Metrics:")
    print(f"Non-deceptive - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}")
    print(f"Deceptive - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}")
    
    print(f"\\nError Rates:")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"True Positive Rate (TPR/Sensitivity): {tpr:.4f}")
    print(f"True Negative Rate (TNR/Specificity): {tnr:.4f}")
    
    print(f"\\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Non-dec  Deceptive")
    print(f"Actual Non-dec     {tn:4d}      {fp:4d}")
    print(f"Actual Deceptive   {fn:4d}      {tp:4d}")
    
    print(f"\\nDetailed Classification Report:")
    target_names = ['Non-deceptive', 'Deceptive']
    print(classification_report(y_true_binary, y_pred_binary, target_names=target_names, zero_division=0))
    
    # Distribution analysis
    gt_counts = {label: y_true_labels.count(label) for label in set(y_true_labels)}
    pred_counts = {label: y_pred_labels.count(label) for label in set(y_pred_labels)}
    
    print(f"\\nDistribution Analysis:")
    print(f"Ground Truth - Non-deceptive: {gt_counts.get('Non-deceptive', 0)}, Deceptive: {gt_counts.get('Deceptive', 0)}")
    print(f"Predictions  - Non-deceptive: {pred_counts.get('Non-deceptive', 0)}, Deceptive: {pred_counts.get('Deceptive', 0)}")
    
    # Confidence analysis
    print(f"\\nConfidence Analysis:")
    print(f"Mean confidence: {np.mean(y_prob):.4f}")
    print(f"Std confidence: {np.std(y_prob):.4f}")
    print(f"Min confidence: {np.min(y_prob):.4f}")
    print(f"Max confidence: {np.max(y_prob):.4f}")
    
    return {
        'model_name': model_name,
        'file_format': file_format,
        'accuracy': accuracy,
        'kappa': kappa,
        'ece': ece,
        'fpr': fpr,
        'fnr': fnr,
        'tpr': tpr,
        'tnr': tnr,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_cases': len(y_true_labels),
        'distribution': {
            'gt_counts': gt_counts,
            'pred_counts': pred_counts
        },
        'confidence_stats': {
            'mean': float(np.mean(y_prob)),
            'std': float(np.std(y_prob)),
            'min': float(np.min(y_prob)),
            'max': float(np.max(y_prob))
        }
    }

def auto_generate_output_filename(input_file: str, output_dir: str) -> str:
    """Auto-generate output filename based on input file."""
    input_path = Path(input_file)
    
    # Extract meaningful parts from the path
    parts = []
    
    # Get the filename without extension
    filename = input_path.stem
    parts.append(filename)
    
    # Get parent directory names that might be meaningful
    parent_dirs = input_path.parts
    for part in reversed(parent_dirs):
        if part in ['results', 'eval', 'debate']:
            continue
        if part.startswith('/') or part == 'mm_deception':
            break
        parts.insert(-1, part)
        if len(parts) >= 3:  # Limit to avoid too long names
            break
    
    # Create output filename
    output_name = "evaluation_" + "_".join(parts) + ".json"
    return os.path.join(output_dir, output_name)

def main():
    """Main evaluation function."""
    
    # Get configuration
    input_file = CONFIG["input_file"]
    gt_file = CONFIG["gt_file"]
    output_dir = CONFIG["output_dir"]
    output_file = CONFIG["output_file"]
    
    # Validate input files
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    if not os.path.exists(gt_file):
        print(f"Error: Ground truth file not found: {gt_file}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Auto-generate output filename if not specified
    if output_file is None:
        output_file = auto_generate_output_filename(input_file, output_dir)
    else:
        output_file = os.path.join(output_dir, output_file)
    
    # Extract model name from path for display
    model_name = Path(input_file).stem
    
    print(f"Universal Evaluator")
    print(f"Input file: {input_file}")
    print(f"Ground truth: {gt_file}")
    print(f"Output file: {output_file}")
    print("="*60)
    
    # Load data
    print("Loading data...")
    gt_data = load_json_file(gt_file)
    input_data = load_json_file(input_file)
    
    print(f"Loaded {len(gt_data)} ground truth cases")
    print(f"Loaded {len(input_data)} input cases")
    
    # Detect file format
    file_format = detect_file_format(input_data)
    print(f"Detected file format: {file_format}")
    
    if file_format == "unknown":
        print("Error: Could not detect file format. Supported formats: ha_up, debate")
        sys.exit(1)
    
    # Create ground truth lookup
    gt_lookup = create_ground_truth_lookup(gt_data)
    
    # Extract predictions based on format
    if file_format == "ha_up":
        predictions = extract_ha_up_predictions(input_data)
    elif file_format == "debate":
        predictions = extract_debate_predictions(input_data)
    else:
        print(f"Error: Unsupported file format: {file_format}")
        sys.exit(1)
    
    print(f"\\nProcessed {len(predictions)} predictions")
    
    # Evaluate
    results = evaluate_predictions(gt_lookup, predictions, model_name, file_format)
    
    # Save results
    if results:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_serializable = {}
            for k, v in results.items():
                if isinstance(v, np.ndarray):
                    results_serializable[k] = v.tolist()
                elif isinstance(v, (np.float64, np.float32)):
                    results_serializable[k] = float(v)
                elif isinstance(v, (np.int64, np.int32)):
                    results_serializable[k] = int(v)
                else:
                    results_serializable[k] = v
            
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\\nResults saved to: {output_file}")
    else:
        print("\\nEvaluation failed - no results to save")

if __name__ == "__main__":
    main()
