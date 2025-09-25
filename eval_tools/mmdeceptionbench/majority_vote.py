#!/usr/bin/env python3
"""
Majority Vote Script for MM-Deceptionbench human agreement Evaluation

This script performs majority vote on the evaluation decisions from 3 JSON files
and synthesizes them into a single file while preserving the original data structure.

Usage:
    python majority_vote.py

The script will:
1. Load the 3 evaluation files
2. Perform majority vote on the 'decision' field
3. Average the 'severity_score' and 'confidence_score' fields
4. Generate a synthesized description based on majority decision
5. Save the result to a new file
"""

import json
import os
from collections import Counter
from typing import Dict, List, Any

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Warning: Expected list format in {file_path}, got {type(data)}")
                return []
            return data
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def perform_majority_vote(decisions: List[str]) -> str:
    """Perform majority vote on decisions."""
    if not decisions:
        return "Non-deceptive"  # Default fallback
    
    counter = Counter(decisions)
    return counter.most_common(1)[0][0]

def average_scores(scores: List[float]) -> float:
    """Calculate average of scores."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def generate_synthesized_description(decisions: List[str], descriptions: List[str], majority_decision: str) -> str:
    """Generate a synthesized description based on majority decision."""
    # Filter descriptions that match the majority decision
    matching_descriptions = [desc for desc, decision in zip(descriptions, decisions) 
                           if decision == majority_decision and desc is not None]
    
    if not matching_descriptions:
        return f"Majority vote result: {majority_decision}"
    
    # Use the first non-null description that matches the majority decision
    return matching_descriptions[0]

def parse_evaluation_string(eval_str: str) -> Dict[str, Any]:
    """Parse evaluation string that might contain JSON."""
    if isinstance(eval_str, dict):
        return eval_str
    
    if not isinstance(eval_str, str):
        return {}
    
    try:
        # Try to extract JSON from markdown code blocks
        if '```json' in eval_str:
            # Extract content between ```json and ```
            start = eval_str.find('```json') + 7
            end = eval_str.find('```', start)
            if end != -1:
                json_str = eval_str[start:end].strip()
            else:
                # If no closing ```, try to extract everything after ```json
                json_str = eval_str[start:].strip()
            
            # Try to parse the JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If JSON is incomplete, try to extract basic fields manually
                return extract_fields_from_incomplete_json(json_str)
        
        # Try to parse as direct JSON
        return json.loads(eval_str)
    except (json.JSONDecodeError, ValueError) as e:
        # Try to extract fields manually from incomplete JSON
        return extract_fields_from_incomplete_json(eval_str)

def extract_fields_from_incomplete_json(json_str: str) -> Dict[str, Any]:
    """Extract fields from incomplete JSON string."""
    result = {}
    
    try:
        # Extract decision field
        if '"decision"' in json_str:
            decision_start = json_str.find('"decision"')
            if decision_start != -1:
                # Find the value after the colon
                colon_pos = json_str.find(':', decision_start)
                if colon_pos != -1:
                    # Find the next quote
                    quote_start = json_str.find('"', colon_pos)
                    if quote_start != -1:
                        quote_end = json_str.find('"', quote_start + 1)
                        if quote_end != -1:
                            decision_value = json_str[quote_start + 1:quote_end]
                            result['decision'] = decision_value
        
        # Extract description field
        if '"description"' in json_str:
            desc_start = json_str.find('"description"')
            if desc_start != -1:
                colon_pos = json_str.find(':', desc_start)
                if colon_pos != -1:
                    quote_start = json_str.find('"', colon_pos)
                    if quote_start != -1:
                        # Find the end of the description (next unescaped quote)
                        desc_value = ""
                        i = quote_start + 1
                        while i < len(json_str):
                            if json_str[i] == '"' and json_str[i-1] != '\\':
                                break
                            desc_value += json_str[i]
                            i += 1
                        result['description'] = desc_value
        
        # Set default values if not found
        if 'decision' not in result:
            result['decision'] = 'Non-deceptive'
        
        # Add default severity and confidence
        result['deception_severity'] = {
            'severity_score': 0.5,
            'description': 'Parsed from incomplete JSON'
        }
        result['confidence_score'] = 0.5
        
    except Exception:
        # If all else fails, return default values
        result = {
            'decision': 'Non-deceptive',
            'description': 'Error parsing evaluation',
            'deception_severity': {
                'severity_score': 0.0,
                'description': 'Parse error'
            },
            'confidence_score': 0.0
        }
    
    return result

def synthesize_evaluation(evaluations: List[Any]) -> Dict[str, Any]:
    """Synthesize evaluation data from multiple sources."""
    if not evaluations:
        return {}
    
    # Parse evaluation data (handle both dict and string formats)
    parsed_evaluations = []
    for eval_data in evaluations:
        if isinstance(eval_data, str):
            parsed_eval = parse_evaluation_string(eval_data)
        elif isinstance(eval_data, dict):
            parsed_eval = eval_data
        else:
            print(f"Warning: Unexpected evaluation data type: {type(eval_data)}")
            parsed_eval = {}
        
        parsed_evaluations.append(parsed_eval)
    
    # Extract decisions
    decisions = [eval_data.get('decision', 'Non-deceptive') for eval_data in parsed_evaluations]
    descriptions = [eval_data.get('description') for eval_data in parsed_evaluations]
    
    # Perform majority vote on decision
    majority_decision = perform_majority_vote(decisions)
    
    # Extract and average scores
    severity_scores = []
    confidence_scores = []
    severity_descriptions = []
    
    for eval_data in parsed_evaluations:
        # Extract severity score
        if 'deception_severity' in eval_data and 'severity_score' in eval_data['deception_severity']:
            severity_scores.append(eval_data['deception_severity']['severity_score'])
            if 'description' in eval_data['deception_severity']:
                severity_descriptions.append(eval_data['deception_severity']['description'])
        
        # Extract confidence score
        if 'confidence_score' in eval_data:
            confidence_scores.append(eval_data['confidence_score'])
    
    # Calculate averages
    avg_severity_score = average_scores(severity_scores)
    avg_confidence_score = average_scores(confidence_scores)
    
    # Generate synthesized description
    synthesized_description = generate_synthesized_description(decisions, descriptions, majority_decision)
    
    # Generate synthesized severity description
    synthesized_severity_description = None
    if severity_descriptions:
        # Use the first non-null severity description
        synthesized_severity_description = next((desc for desc in severity_descriptions if desc), None)
    
    # Build the synthesized evaluation
    synthesized_eval = {
        "decision": majority_decision,
        "description": synthesized_description,
        "deception_severity": {
            "severity_score": avg_severity_score,
            "description": synthesized_severity_description
        },
        "confidence_score": avg_confidence_score
    }
    
    return synthesized_eval

def main():
    """Main function to perform majority vote synthesis."""
    # Define file paths
    base_dir = "/Users/Shared/Files From d.localized/pku/mm_deception/mm_deception/eval/results/gpt-4o/mv"
    files = [
        "584_middle_prompt-1.json",
        "584_middle_prompt-2.json", 
        "584_middle_prompt-3.json"
    ]
    
    # Load all files
    all_data = []
    for filename in files:
        file_path = os.path.join(base_dir, filename)
        data = load_json_file(file_path)
        if data:
            all_data.append(data)
            print(f"Loaded {len(data)} cases from {filename}")
        else:
            print(f"Failed to load {filename}")
    
    if len(all_data) != 3:
        print("Error: Could not load all 3 files")
        return
    
    # Verify all files have the same number of cases
    case_counts = [len(data) for data in all_data]
    if len(set(case_counts)) != 1:
        print(f"Warning: Files have different numbers of cases: {case_counts}")
        min_cases = min(case_counts)
        print(f"Using minimum number of cases: {min_cases}")
    else:
        min_cases = case_counts[0]
    
    # Synthesize data
    synthesized_data = []
    error_count = 0
    
    for i in range(min_cases):
        try:
            # Extract evaluations for this case
            evaluations = []
            for data in all_data:
                if i < len(data) and 'evaluation' in data[i]:
                    evaluations.append(data[i]['evaluation'])
                else:
                    print(f"Warning: Missing evaluation for case {i}")
                    evaluations.append({})
            
            # Create synthesized case (use first file as base structure)
            synthesized_case = all_data[0][i].copy()
            
            # Replace evaluation with synthesized version
            synthesized_case['evaluation'] = synthesize_evaluation(evaluations)
            
            synthesized_data.append(synthesized_case)
            
        except Exception as e:
            print(f"Error processing case {i}: {e}")
            error_count += 1
            # Skip this case or use default evaluation
            if i < len(all_data[0]):
                synthesized_case = all_data[0][i].copy()
                synthesized_case['evaluation'] = {
                    "decision": "Non-deceptive",
                    "description": f"Error in processing: {str(e)[:100]}",
                    "deception_severity": {
                        "severity_score": 0.0,
                        "description": "Error in evaluation processing"
                    },
                    "confidence_score": 0.0
                }
                synthesized_data.append(synthesized_case)
    
    # Save synthesized data
    output_file = os.path.join(base_dir, "584_eval_by_gemini-2.5-pro_majority_vote.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(synthesized_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved synthesized data to {output_file}")
        print(f"Total cases processed: {len(synthesized_data)}")
        if error_count > 0:
            print(f"Cases with errors: {error_count}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return
    
    # Print summary statistics
    try:
        decisions = [case['evaluation']['decision'] for case in synthesized_data]
        decision_counts = Counter(decisions)
        print(f"\nSummary:")
        print(f"Deceptive cases: {decision_counts.get('Deceptive', 0)}")
        print(f"Non-deceptive cases: {decision_counts.get('Non-deceptive', 0)}")
        
        # Calculate averages safely
        severity_scores = []
        confidence_scores = []
        
        for case in synthesized_data:
            try:
                if 'evaluation' in case and 'deception_severity' in case['evaluation']:
                    severity_scores.append(case['evaluation']['deception_severity'].get('severity_score', 0.0))
                if 'evaluation' in case:
                    confidence_scores.append(case['evaluation'].get('confidence_score', 0.0))
            except Exception as e:
                print(f"Warning: Error extracting scores from case: {e}")
        
        if severity_scores:
            avg_severity = sum(severity_scores) / len(severity_scores)
            print(f"Average severity score: {avg_severity:.3f}")
        
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            print(f"Average confidence score: {avg_confidence:.3f}")
            
    except Exception as e:
        print(f"Error generating summary statistics: {e}")

if __name__ == "__main__":
    main()
