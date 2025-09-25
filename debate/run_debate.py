"""
Debate Runner
"""

import yaml, json, os
from datetime import datetime

from utils import api, vllm_api
from utils import load_debate_ds,build_inference_payload, build_init_user_prompt, format_history_item, clear_visual_cache
from logger import create_debate_logger


class Debater:
    def __init__(self, role: str, config: dict):
        self.role = role
        self.config = config

class Judge:
    def __init__(self, role: str, config: dict):
        self.role = role
        self.config = config
        
class DebateRunner:
    def __init__(self, config: dict):
        self.config = config
        self.round = config['round']
        self.history = None
        self.debaters = self.init_debaters(config['debaters'])
        self.judge = Judge(role=config['judge']['role'], config=config['judge']['config'])
        self.logger = create_debate_logger("DebateRunner")

    def decide_infer_backend(self, config: dict):
        """
        Initialize the models.
        """
        if config['backend'] == 'vllm':
            return vllm_api
        elif config['backend'] == 'api':
            return api
        else:
            raise ValueError(f"Invalid backend: {config['backend']}")
            
    def init_debaters(self, debater_cfg_list: list[dict]) -> list[Debater]:
        """
        Initialize the debaters.
        """
        debaters = []
        for debater_cfg in debater_cfg_list:
            debaters.append(Debater(role=debater_cfg['role'], config=debater_cfg['config']))
        return debaters
    
    def init_history(self, dataset_size: int) -> list[list[dict]]:
        """
        Initialize the history.
        """
        return [[] for _ in range(dataset_size)]

    def run_debate(self, dataset_path: str) -> list[dict]:
        """
        Run the debate for a single dataset.
        """
        # Load dataset
        ds = load_debate_ds(dataset_path)
        
        self.logger.log_initialization_start()
        init_user_prompts = [build_init_user_prompt(item) for item in ds]
        # Create simplified image_infos with combined images for normalized coordinates
        from utils import combine_images
        from PIL import Image
        import os
        
        image_infos = []
        for item in ds:
            # Load images from file paths
            loaded_images = []
            for image_path in item['images']:
                full_path = os.path.join('./mm-deceptionbench', image_path)
                loaded_images.append(Image.open(full_path))
            #loaded_images = [item['image']]
            
            if len(loaded_images) == 1:
                combined_image = loaded_images[0]
            else:
                # Combine multiple images horizontally
                combined_image = combine_images(loaded_images, layout="horizontal", spacing=20)
                print(f"Combined {len(loaded_images)} images into single image of size {combined_image.size}")
            image_infos.append({'image': combined_image})
        self.history = self.init_history(len(ds))
       
        self.logger.log_debate_rounds_start(self.round)
        debate_start_time = datetime.now()
        
        for round_idx in range(self.round):
            round_start_time = datetime.now()
            self.logger.log_round_start(round_idx, self.round)

            debater_start_time = datetime.now()

            for debater_idx, debater in enumerate(self.debaters):
                
                self.logger.log_debater_start(debater.role)
                
                # build inference payload
                payload = [build_inference_payload(debater.role, history, init_user_prompt, image_info) \
                    for history, init_user_prompt, image_info in zip(self.history, init_user_prompts, image_infos)]
                
                self.logger.log_payload_built(len(payload))
                
                # Inference
                backend = debater.config.get('backend', 'unknown')
                self.logger.log_debater_inference_start(backend)
                results = self.decide_infer_backend(debater.config)(payload, debater.config)
                
                debater_inference_time = datetime.now() - debater_start_time
                self.logger.log_debater_inference_end(debater.role, debater_inference_time.total_seconds())

                # Post process
                formatted_history_items = [format_history_item(result, debater.role) for result in results]

                # Update debate history
                for history, history_item in zip(self.history, formatted_history_items):
                    history.append(history_item)
                
                self.logger.log_history_updated([len(h) for h in self.history])
                
            round_time = datetime.now() - round_start_time
            self.logger.log_round_end(round_idx, round_time.total_seconds())

        # Judge

        payload = [build_inference_payload(self.judge.role, history, init_user_prompt, image_info) \
                    for history, init_user_prompt, image_info in zip(self.history, init_user_prompts, image_infos)]

        self.logger.log_payload_built(len(payload))
        backend = self.judge.config.get('backend', 'unknown')
        self.logger.log_debater_inference_start(backend)
        results = self.decide_infer_backend(self.judge.config)(payload, self.judge.config)
        judge_inference_time = datetime.now() - debater_start_time

        formatted_history_items = [format_history_item(result, self.judge.role) for result in results]

        for history, history_item in zip(self.history, formatted_history_items):
            history.append(history_item)

        self.logger.log_history_updated([len(h) for h in self.history])
        
        total_debate_time = datetime.now() - debate_start_time
        stats = {
            'cases': len(ds),
            'rounds': self.round,
            'debaters': len(self.debaters)
        }
        self.logger.log_debate_complete(total_debate_time.total_seconds(), stats)

        # Clear visual cache to reduce memory usage before saving
        for history in self.history:
            clear_visual_cache(history)

        debate_trajectory = []

        for history, item in zip(self.history, ds):
            
            debate_trajectory.append({
                "case": item,
                "debate_trajectory": history,
            })  
        
        return debate_trajectory

    def judge_debate(self, history:list[dict] | dict) -> list[dict]:
        """
        Deliver the verdict given the debate trajectory
        """
        pass

def extract_dataset_name(dataset_path: str) -> str:
    """
    Extract dataset name from file path.
    Examples:
    - "/path/to/bluffing.json" -> "bluffing"
    - "/path/to/responses_fabrication_20250904_022603.json" -> "fabrication"
    """
    import os
    filename = os.path.basename(dataset_path)
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Handle different naming patterns
    if name.startswith('responses_'):
        # Extract type from "responses_type_timestamp" pattern
        parts = name.split('_')
        if len(parts) >= 2:
            return parts[1]
    
    return name

def process_datasets_config(config: dict) -> list[dict]:
    """
    Process datasets configuration to support both single and multiple datasets.
    
    Returns:
        List of dataset info dictionaries with 'path' and 'name' keys
    """
    datasets = []
    
    # Check if using new batch configuration
    if 'datasets' in config:
        if isinstance(config['datasets'], list):
            # List of dataset configurations
            for dataset_config in config['datasets']:
                if isinstance(dataset_config, dict):
                    # Full config with path and optional name
                    path = dataset_config['path']
                    name = dataset_config.get('name', extract_dataset_name(path))
                    datasets.append({'path': path, 'name': name})
                else:
                    # Just a path string
                    path = dataset_config
                    name = extract_dataset_name(path)
                    datasets.append({'path': path, 'name': name})
        else:
            # Single dataset path
            path = config['datasets']
            name = extract_dataset_name(path)
            datasets.append({'path': path, 'name': name})
    elif 'dataset' in config:
        # Backward compatibility with old single dataset config
        path = config['dataset']
        name = extract_dataset_name(path)
        datasets.append({'path': path, 'name': name})
    else:
        raise ValueError("No dataset configuration found. Please specify 'datasets' or 'dataset' in config.")
    
    return datasets

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run debate with specified config file')
    parser.add_argument('--config', default='config.yaml', help='Path to config file (default: config.yaml)')
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    runner = DebateRunner(config)
    
    # Process datasets configuration
    datasets = process_datasets_config(config)
    
    # Log startup information
    debater_roles = [debater.role for debater in runner.debaters]
    runner.logger.log_debate_start(runner.round, len(datasets), debater_roles)
    runner.logger.info(f"Starting batch processing for {len(datasets)} datasets")
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)
    
    successful_datasets = 0
    
    try:
        # Process each dataset in the batch
        for i, dataset_info in enumerate(datasets, 1):
            dataset_name = dataset_info['name']
            dataset_path = dataset_info['path']
            
            runner.logger.info(f"Processing dataset {i}/{len(datasets)}: {dataset_name} ({dataset_path})")
            
            try:
                # Run debate for this dataset (including judge)
                debate_trajectory = runner.run_debate(dataset_path)
                
                # Generate filename: {dataset_name}-debate_trajectory_r{rounds}.json
                filename = f"{dataset_name}-debate_trajectory_r{runner.round}.json"
                output_path = os.path.join(config['output_dir'], filename)
                
                # Save results
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(debate_trajectory, f, indent=4, ensure_ascii=False)
                
                runner.logger.log_output_saved(output_path)
                runner.logger.info(f"Successfully processed {dataset_name}: {output_path}")
                successful_datasets += 1
                
            except Exception as e:
                runner.logger.error(f"Failed to process dataset {dataset_name}: {str(e)}")
                # Continue with other datasets
                continue
        
        runner.logger.log_program_complete()
        runner.logger.info(f"Batch processing completed. Successfully processed {successful_datasets}/{len(datasets)} datasets with {runner.round} rounds each")
        
    except Exception as e:
        runner.logger.log_error(str(e))
        runner.logger.exception("Detailed error information:")
        raise










