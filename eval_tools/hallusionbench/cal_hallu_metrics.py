#!/usr/bin/env python3
"""
Adapted from https://github.com/tianyi-lab/HallusionBench/blob/main/utils.py
"""

import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import numpy as np


def extract_data_fields(item: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Extract case data and is_correct field from data item

    Returns:
        Tuple[case_data, is_correct]
    """
    if "case" in item:
        # data in case field
        case_data = item["case"]
        is_correct = item["is_correct"]
    else:
        # data directly at top level
        case_data = item
        is_correct = item["is_correct"]
    
    return case_data, is_correct


def get_eval_all(data, model_correctness_entry="is_correct"):
    """
    Per question statistics - accuracy per question
    Corresponding to get_eval_all function in utils.py
    """
    eval_all_dict = dict()
    eval_all_stat = {}
    eval_all_stat["LH"] = 0  # Language Hallucination
    eval_all_stat["VI"] = 0  # Visual Illusion
    eval_all_stat["Mix"] = 0  # Mixed

    for r in data:
        case_data, is_correct = extract_data_fields(r)
        
        # Use set_id, figure_id, question_id, subcategory as unique identifier
        name = "_".join([case_data["set_id"], case_data["figure_id"], case_data["question_id"], case_data["subcategory"]])
        assert name not in eval_all_dict, f"Duplicate entry: {name}"
        
        # Use is_correct field
        eval_all_dict[name] = 1 if is_correct else 0
        
        # Analyze LH/VI/Mix according to your data structure
        if str(case_data["category"]) == "VD":  # VD type
            if str(case_data["figure_id"]) == "0":  # Original image
                if not is_correct:  # Wrong answer
                    eval_all_stat["VI"] += 1  # Visual Illusion
            else:  # Modified image
                if not is_correct:
                    eval_all_stat["Mix"] += 1  # Mixed error
        else:  # VS type
            if str(case_data["visual_input"]) == "0":  # No visual
                if not is_correct:
                    eval_all_stat["LH"] += 1  # Language Hallucination
            else:  # With visual input
                if not is_correct:
                    eval_all_stat["Mix"] += 1

    eval_all_stat["note"] = "all accuracy per question"
    eval_all_stat["total"] = len(eval_all_dict.keys())
    eval_all_stat["correct"] = np.count_nonzero(list(eval_all_dict.values()))
    eval_all_stat["wrong"] = eval_all_stat["total"] - eval_all_stat["correct"]
    
    # Add percentage information
    eval_all_stat["accuracy_percentage"] = eval_all_stat["correct"] / eval_all_stat["total"] if eval_all_stat["total"] > 0 else 0.0
    eval_all_stat["LH_percentage"] = eval_all_stat["LH"] / eval_all_stat["total"] if eval_all_stat["total"] > 0 else 0.0
    eval_all_stat["VI_percentage"] = eval_all_stat["VI"] / eval_all_stat["total"] if eval_all_stat["total"] > 0 else 0.0
    eval_all_stat["Mix_percentage"] = eval_all_stat["Mix"] / eval_all_stat["total"] if eval_all_stat["total"] > 0 else 0.0

    return eval_all_stat


def get_eval_pair_all(data, model_correctness_entry="is_correct"):
    """
    Per question pair statistics - accuracy per question pair
    Corresponding to get_eval_pair_all function in utils.py
    """
    # Find original question (figure_id=0) correctness
    orig_correctness = dict()
    counter = 0
    lh_counter = 0
    vi_counter = 0
    both_counter = 0

    for r in data:
        case_data, is_correct = extract_data_fields(r)
            
        if str(case_data["figure_id"]) == "0":
            key = "_".join([case_data["set_id"], case_data["question_id"], case_data["subcategory"]])
            orig_correctness[key] = is_correct

    get_eval_pair_dict = dict()
    get_analysis_pair_dict = dict()

    for r in data:
        case_data, is_correct = extract_data_fields(r)
            
        # Use set_id, question_id, subcategory as pairing identifier
        name = "_".join([case_data["set_id"], case_data["question_id"], case_data["subcategory"]])
        if name in get_eval_pair_dict:
            c, t = get_eval_pair_dict[name]
            get_eval_pair_dict[name] = (c + (1 if is_correct else 0), t+1)
        else:
            get_eval_pair_dict[name] = (1 if is_correct else 0, 1)    
        counter += 1    

        # Analyze LH/VI/Mix
        analysis = (0, 0)  # (LH, VI)
        if str(case_data["figure_id"]) == "0":  # Original question
            if str(case_data["category"]) == "VD":  # VD type
                if not is_correct:
                    analysis = (0, 1)  # VI - Visual Illusion
            else:  # VS type
                if not is_correct:
                    analysis = (1, 0)  # LH - Language Hallucination
        else:  # Modified question
            key = "_".join([case_data["set_id"], case_data["question_id"], case_data["subcategory"]])
            orig_c = orig_correctness[key]
            if str(case_data["category"]) == "VD":  # VD type
                if orig_c and not is_correct:
                    # Original correct but modified wrong
                    analysis = (0, 1)  # VI - Visual Illusion
                elif not orig_c and not is_correct:
                    analysis = (0, 1)  # VI - Visual Illusion
            else:  # VS type
                if not orig_c and not is_correct:
                    analysis = (1, 1)  # Mixed - Language and visual problems
                elif orig_c and not is_correct:
                    if str(case_data["visual_input"]) == "1":  # Common visual problem
                        analysis = (0, 1)  # VI - Visual Illusion
                    else:  # Anti-common visual problem
                        analysis = (1, 0)  # LH - Language Hallucination

        if analysis[0] > 0 and analysis[1] > 0:
            both_counter += 1
        elif analysis[0] > 0:
            lh_counter += 1
        elif analysis[1] > 0:
            vi_counter += 1

        if name in get_analysis_pair_dict:
            lh, vi = get_analysis_pair_dict[name]
            get_analysis_pair_dict[name] = (lh + analysis[0], vi + analysis[1])
        else:
            get_analysis_pair_dict[name] = analysis     

    eval_all_pair_stat = {}
    eval_all_pair_stat["note"] = "all accuracy per question pair"
    eval_all_pair_stat["total"] = len(get_eval_pair_dict.keys())
    eval_all_pair_stat["total_q"] = counter
    eval_all_pair_stat["correct"] = 0
    eval_all_pair_stat["wrong"] = 0
    eval_all_pair_stat["LH"] = 0
    eval_all_pair_stat["VI"] = 0
    eval_all_pair_stat["Mix"] = 0

    eval_all_pair_stat["LH_cg"] = lh_counter
    eval_all_pair_stat["VI_cg"] = vi_counter
    eval_all_pair_stat["Mix_cg"] = both_counter
    
    # Add percentage information (will be calculated after the loop)
    eval_all_pair_stat["accuracy_percentage"] = 0.0
    eval_all_pair_stat["LH_percentage"] = 0.0
    eval_all_pair_stat["VI_percentage"] = 0.0
    eval_all_pair_stat["Mix_percentage"] = 0.0

    for k in get_eval_pair_dict.keys():
        v = get_eval_pair_dict[k]
        a = get_analysis_pair_dict[k]
        if v[0] == v[1]:  # 配对中所有问题都正确
            eval_all_pair_stat["correct"] += 1
        else:
            eval_all_pair_stat["wrong"] += 1
        if a[0] > 0 and a[1] > 0:
            eval_all_pair_stat["Mix"] += 1
        elif a[0] > 0:
            eval_all_pair_stat["LH"] += 1
        elif a[1] > 0:
            eval_all_pair_stat["VI"] += 1

    # Calculate percentages after the loop
    eval_all_pair_stat["accuracy_percentage"] = eval_all_pair_stat["correct"] / eval_all_pair_stat["total"] if eval_all_pair_stat["total"] > 0 else 0.0
    eval_all_pair_stat["LH_percentage"] = eval_all_pair_stat["LH"] / eval_all_pair_stat["total"] if eval_all_pair_stat["total"] > 0 else 0.0
    eval_all_pair_stat["VI_percentage"] = eval_all_pair_stat["VI"] / eval_all_pair_stat["total"] if eval_all_pair_stat["total"] > 0 else 0.0
    eval_all_pair_stat["Mix_percentage"] = eval_all_pair_stat["Mix"] / eval_all_pair_stat["total"] if eval_all_pair_stat["total"] > 0 else 0.0

    return eval_all_pair_stat


def get_eval_fig(data):
    """
    Per figure statistics - internal consistency per image
    Corresponding to get_eval_fig function in utils.py
    """
    eval_fig_dict = dict()

    for r in data:
        case_data, is_correct = extract_data_fields(r)
            
        if case_data["category"] == "VS" and str(case_data["figure_id"]) == "0":  # No image
            continue
        # Use set_id, figure_id, subcategory as image identifier
        name = "_".join([case_data["set_id"], case_data["figure_id"], case_data["subcategory"]])
        if name in eval_fig_dict:
            c, t = eval_fig_dict[name]
            eval_fig_dict[name] = (c + (1 if is_correct else 0), t+1)
        else:
            eval_fig_dict[name] = (1 if is_correct else 0, 1)

    eval_fig_stat = {}
    eval_fig_stat["note"] = "all accuracy per image (consistency test)"
    eval_fig_stat["total"] = len(eval_fig_dict.keys())
    eval_fig_stat["correct"] = 0
    eval_fig_stat["wrong"] = 0
    eval_fig_stat["inconsistent"] = 0
    eval_fig_stat["score"] = 0

    for v in eval_fig_dict.values():
        if v[0] == v[1]:  # All questions in the image are correct
            eval_fig_stat["correct"] += 1
        elif v[0] == 0:  # All questions in the image are wrong
            eval_fig_stat["wrong"] += 1
        else:  # Questions in the image have right and wrong (inconsistent)
            eval_fig_stat["inconsistent"] += 1
        eval_fig_stat["score"] += (v[0] / v[1])
            
    eval_fig_stat["score"] = eval_fig_stat["score"] / eval_fig_stat["total"]
    
    # Add percentage information
    eval_fig_stat["correct_percentage"] = eval_fig_stat["correct"] / eval_fig_stat["total"] if eval_fig_stat["total"] > 0 else 0.0
    eval_fig_stat["wrong_percentage"] = eval_fig_stat["wrong"] / eval_fig_stat["total"] if eval_fig_stat["total"] > 0 else 0.0
    eval_fig_stat["inconsistent_percentage"] = eval_fig_stat["inconsistent"] / eval_fig_stat["total"] if eval_fig_stat["total"] > 0 else 0.0
    
    return eval_fig_stat


def analyze_correctness_statistics(json_file_path: str) -> Dict[str, Any]:
    """
    Comprehensive analysis of correctness statistics in JSON file
    """
    # Read JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Analyzing file: {json_file_path}")
    print(f"Total number of items: {len(data)}")
    
    # Group by set_id
    set_groups = defaultdict(list)
    for item in data:
        case_data, _ = extract_data_fields(item)
        set_groups[case_data["set_id"]].append(item)
    
    print(f"Total number of sets: {len(set_groups)}")
    
    # Execute three statistics
    per_question_stats = get_eval_all(data)
    per_pair_stats = get_eval_pair_all(data)
    per_figure_stats = get_eval_fig(data)
    
    results = {
        "per_question": per_question_stats,
        "per_pair": per_pair_stats,
        "per_figure": per_figure_stats,
        "set_count": len(set_groups),
        "total_items": len(data)
    }
    
    return results


def print_results(results: Dict[str, Any]):
    """Print statistics results"""
    print("\n" + "="*80)
    print("Statistics results (based on utils.py logic)")
    print("="*80)
    
    # Per Question statistics
    pq = results["per_question"]
    print(f"\n📊 Per Question statistics:")
    print(f"  Total number of questions: {pq['total']}")
    print(f"  Correct number: {pq['correct']}")
    print(f"  Wrong number: {pq['wrong']}")
    print(f"  Accuracy: {pq['correct']/pq['total']:.4f} ({pq['correct']/pq['total']*100:.2f}%)")
    print(f"  Language Hallucination (LH): {pq['LH']} ({pq['LH']/pq['total']*100:.2f}%)")
    print(f"  Visual Illusion (VI): {pq['VI']} ({pq['VI']/pq['total']*100:.2f}%)")
    print(f"  Mixed error (Mix): {pq['Mix']} ({pq['Mix']/pq['total']*100:.2f}%)")
    
    # Per Pair statistics
    pp = results["per_pair"]
    print(f"\n📊 Per Question Pair statistics:")
    print(f"  Total number of pairs: {pp['total']}")
    print(f"  Correct number: {pp['correct']}")
    print(f"  Wrong number: {pp['wrong']}")
    print(f"  Accuracy: {pp['correct']/pp['total']:.4f} ({pp['correct']/pp['total']*100:.2f}%)")
    print(f"  Language Hallucination (LH): {pp['LH']} ({pp['LH']/pp['total']*100:.2f}%)")
    print(f"  Visual Illusion (VI): {pp['VI']} ({pp['VI']/pp['total']*100:.2f}%)")
    print(f"  Mixed error (Mix): {pp['Mix']} ({pp['Mix']/pp['total']*100:.2f}%)")
    
    # Per Figure statistics
    pf = results["per_figure"]
    print(f"\n📊 Per Figure statistics:")
    print(f"  Total number of images: {pf['total']}")
    print(f"  All correct: {pf['correct']} ({pf['correct']/pf['total']*100:.2f}%)")
    print(f"  All wrong: {pf['wrong']} ({pf['wrong']/pf['total']*100:.2f}%)")
    print(f"  Inconsistent: {pf['inconsistent']} ({pf['inconsistent']/pf['total']*100:.2f}%)")
    print(f"  Consistency score: {pf['score']:.4f}")
    
    print(f"\n📊 Overall information:")
    print(f"  Set number: {results['set_count']}")
    print(f"  Total number of items: {results['total_items']}")


def main():
    """Main function"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python cal_hallu_metrics.py <json_file_path>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    
    try:
        results = analyze_correctness_statistics(json_file_path)
        print_results(results)
        
        # Save results to file
        output_file = json_file_path.replace('.json', '_hallu_metrics.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File not found {json_file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: JSON file format error - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
