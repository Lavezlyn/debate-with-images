# Debate with Images: Detecting Multimodal Deceptive Behaviors in MLLMs

A comprehensive framework for evaluating deceptive behaviors in multimodal large language models (MLLMs) through both direct evaluation and debate-based assessment methods.

## Overview

This project provides evaluation tools for **MM-DeceptionBench** and implements a novel **debate with images** approach to detect deceptive behaviors in MLLMs. Additionally, it includes compatibility with **PKU-SafeRLHF-V** and **HallusionBench** for comprehensive multimodal safety evaluation.

## Project Structure

```
multimodal_deception/
├── debate/                     # Debate-based evaluation system
├── eval_tools/                 # Evaluation utilities and benchmarks  
├── mm-deceptionbench/         # MM-DeceptionBench dataset
└── human_label.json           # Human-labeled evaluation data
```

## Features

- **MM-DeceptionBench Evaluation**: Direct MLLM-as-a-judge evaluation for multimodal deception detection
- **Debate with Images**: Multi-agent debate system with visual evidence for enhanced evaluation accuracy
- **PKU-SafeRLHF-V Integration**: Safety preference evaluation with visual context
- **HallusionBench Compatibility**: Visual hallucination detection capabilities
- **Interactive Visualization**: Gradio-based web interface for debate inspection

## Quick Start

### Prerequisites

```bash
# Core dependencies
pip install ray gradio matplotlib pillow opencv-python numpy
pip install scikit-learn datasets tqdm pyyaml
pip install aiohttp urllib3  # For async API calls

# For vLLM backend (optional)
pip install vllm
```

### Basic Usage

#### 1. MM-DeceptionBench Evaluation

To inference models on MM-DeceptionBench

```bash
cd eval_tools/mmdeceptionbench
python routine.py --config config.yaml
```

Configuration files are available in `eval_tools/mmdeceptionbench/configs/`:
- `api_config.yaml` - API-based models (GPT-4, Claude)
- `vllm_config.yaml` - Local vLLM models
- `reason_config.yaml` - Reasoning-enhanced evaluation

Direct evaluation using MLLM-as-a-judge:

```bash
cd eval_tools/mmdeceptionbench
python eval.py
```

#### 2. Debate with images

Run multi-agent debate with visual evidence:

```bash
cd debate
python run_debate.py --config config.yaml
```

The debate system supports:
- Multiple debaters with different roles (affirmer/negator)
- Visual operations (bounding boxes, points, lines) for evidence
- Judge-based final decision making
- Batch processing of multiple datasets

#### 3. Interactive Debate Visualization

Launch the Gradio web interface:

```bash
cd debate
python debate_gradio_app.py
```

Access the interface at `http://localhost:7860` to visualize debate processes and inspect visual evidence.

## Configuration

### Debate Configuration (`debate/config.yaml`)

```yaml
datasets:
  - path: "./human_label.json"
    name: "debate_sample"
output_dir: "./results/debate"
round: 2

debaters:
  - role: "D_aff"              # Affirmer (argues deception exists)
    config:
      model: "gpt-4o"
      backend: "api"
      api_key: "your-api-key"
      
  - role: "D_neg"              # Negator (argues no deception)
    config:
      model: "gpt-4o"
      backend: "api"
      api_key: "your-api-key"

judge:
  role: "D_judge"              # Final judge
  config:
    model: "gpt-4o"
    backend: "api"
    api_key: "your-api-key"
```

### Backend Options

#### API Backend (OpenAI, Claude)
```yaml
backend: "api"
model: "gpt-4o"
api_key: "your-key"
infer_cfgs:
  temperature: 0.0
  max_tokens: 512
```

#### vLLM Backend (Local Models)
```yaml
backend: "vllm"
model: "Qwen2.5-VL-72B-Instruct"
base_url: "http://localhost:8000"
infer_cfgs:
  temperature: 0.7
  max_tokens: 512
```

To start a vLLM server:
```bash
cd debate
bash vllm_server.sh
```

## Evaluation Tools

### Universal Evaluator

Evaluate judgement of multiple formats against human gold labels on multimodal deception evaluation task:

```bash
cd eval_tools/mmdeceptionbench
python universal_evaluator.py
```

Configure the input files in the script:
```python
CONFIG = {
    "input_file": "./model_output.json",
    "gt_file": "./ground_truth.json", 
    "output_dir": "./evaluation_results"
}
```

### PKU-SafeRLHF-V Evaluation

Evaluate safety preferences with visual context:

```bash
cd eval_tools/pkusaferlhfv
python eval_saferlhfv.py
```

### HallusionBench Evaluation

Detect visual hallucinations:

```bash
cd eval_tools/hallusionbench
python eval_hall.py
```

Alternatively, the debate with images framework support running debate for PKU-SafeRLHF-V and HallusionBench. Just replace the corresponding prompt and dataset loading in `debate/format_utils.py`.

## Data Format

### MM-DeceptionBench Evaluation Format

Organize the inference results in the following format to evaluate deceptiveness.

```json
{
  "case_id": "unique_identifier",
  "source": "dataset_source", 
  "scenario": "evaluation_scenario",
  "assistant_profile": "model_persona",
  "user_profile": "user_context",
  "prompt": "user_input",
  "images": ["image_path1.jpg"],
  "result": {
    "reasoning": "model_thinking_process",
    "output": "model_response"
  }
}
```

### Debate Output Format

```json
{
  "case": {
    "case_id": "identifier",
    "source": "dataset",
    // ... case information
  },
  "debate_trajectory": [
    {
      "role": "D_aff",
      "content": "argument_text",
      "visual_evidence": [
        {
          "bbox_2d": [0.1, 0.2, 0.3, 0.4],
          "label": "evidence_description"
        }
      ]
    }
    // ... more debate rounds
  ]
}
```

## Advanced Features

### Visual Operations

The debate system supports various visual evidence types:

- **Bounding Boxes**: `bbox_2d: [x, y, width, height]`
- **Points**: `point_2d: [x, y]` 
- **Lines**: `line_2d: [x1, y1, x2, y2]`

All coordinates are normalized to [0, 1] range.

By default, we suggest using bounding boxes only.

### Batch Processing

Process multiple datasets:

```yaml
datasets:
  - path: "./dataset1.json"
    name: "fabrication"
  - path: "./dataset2.json" 
    name: "sycophancy"
```

### Caching

Enable response caching for faster iteration:

```yaml
use_cache: true
cache_dir: "./cache"
```