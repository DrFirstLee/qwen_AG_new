import os
import torch
import random
from PIL import Image
import my_prompt4 as my_prompt
from file_managing import (
    load_selected_samples,
    get_actual_path,
    get_gt_path,
)
from config import AGD20K_PATH, model_name
from VLM_model_dot import QwenVLModel, MetricsTracker
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ENABLE_SDPA"] = "1"


def affordance_grounding(model, action, object_name, image_path, gt_path, exo_path=None, exo_type=None, failed_heatmap_path=None, validation_reason=None):
    """
    Process each image using Qwen VL model
    """
    # print(f"Processing image: Action: {action}, Object: {object_name}, Image path: {image_path.split('/')[-1]}, GT path: {gt_path.split('/')[-1]}, Image exists: {os.path.exists(image_path)}, GT exists: {os.path.exists(gt_path)}")
    

    if exo_path is None:
        prompt = my_prompt.process_image_ego_prompt(action, object_name)
               
        results = model.process_image_ego(image_path, prompt, gt_path, action, exo_type)

        
    else:
        if failed_heatmap_path is not None:
            # When we have a failed heatmap, include it in the prompt for better context
            
            prompt = my_prompt.process_image_exo_with_heatmap_prompt(action, object_name, validation_reason)
        
            results = model.process_image_exo_with_heatmap(image_path, prompt, gt_path, exo_path, failed_heatmap_path, action, exo_type)
        else:
            prompt = my_prompt.process_image_exo_prompt(action, object_name)
            results = model.process_image_exo(image_path, prompt, gt_path, exo_path, action, exo_type)

    return results
    # return {
    #     'text_result': result.strip(),
    #     'bboxes': bboxes,
    #     'bbox_image_path': bbox_image_path,
    #     'heatmap_tensor': heatmap_tensor,
    #     'metrics': metrics
    # }


    # Save results



    """
    Get a random exocentric image path based on the egocentric image path
    Args:
        ego_path (str): Path to the egocentric image
    Returns:
        str: Path to a random exocentric image, or None if no exo images found
    """
    try:
        # Extract action and object from ego path
        # Example ego path: .../Seen/testset/egocentric/wash/cup/cup_003621.jpg
        parts = ego_path.split('/')
        action_idx = parts.index('egocentric') + 1
        action = parts[action_idx]
        object_name = parts[action_idx + 1]
        
        # Construct exo directory path
        # Change 'testset/egocentric' to 'trainset/exocentric'
        exo_dir = os.path.join(
            AGD20K_reference_PATH,
            'Seen',
            'trainset',
            'exocentric',
            action,
            object_name
        )
        
        # Check if directory exists
        if not os.path.exists(exo_dir):
            print(f"⚠️ No exocentric directory found: {exo_dir}")
            return None
            
        # Get all jpg files in the directory
        exo_files = [f for f in os.listdir(exo_dir) if f.endswith('.jpg')]
        
        if not exo_files:
            print(f"⚠️ No exocentric images found in: {exo_dir}")
            return None
            
        # Select a random file
        random_exo = random.choice(exo_files)
        exo_path = os.path.join(exo_dir, random_exo)
        
        print(f"Selected exo image: {exo_path}")
        return exo_path
        
    except Exception as e:
        print(f"⚠️ Error finding exocentric image: {str(e)}")
        return None


def main():
    # Initialize Qwen VL model
    model = QwenVLModel(model_name = model_name)
    metrics_tracker_ego = MetricsTracker(name="only_ego")
    metrics_tracker_exo_best = MetricsTracker(name="with_exo_best")

    json_path = os.path.join("selected_samples.json")
    data = load_selected_samples(json_path)

    # Get total number of samples
    total_samples = len(data['selected_samples'])
    
    # Process each sample
    print(f"Processing {total_samples} samples...")
    print("=" * 50)    
    for pair_key, sample_info in data["selected_samples"].items():
        print("--- Start  ", "-"*80) 
        
        action = sample_info["action"]
        object_name = sample_info["object"]

        image_path = get_actual_path(sample_info["image_path"])
        gt_path = get_gt_path(image_path)    
        print(f"Action : {action}, Object : {object_name} image_name : {image_path.split('/')[-1]}")
        exo_best_path = "dogs.jpg"
        if  (exo_best_path is None):
            print(f"NO SEEN DATA SET : {action}/{object_name}")
            continue
        # Process the image
        results_ego = affordance_grounding(model, action, object_name, image_path, gt_path)
        metrics_ego = results_ego['metrics']
        if metrics_ego:
            # Update and print metrics
            metrics_tracker_ego.update(metrics_ego)
            metrics_tracker_ego.print_metrics(metrics_ego, image_path.split('/')[-1])
            
        # with exo random
        results_exo_best = affordance_grounding(model, action, object_name, image_path, gt_path, exo_best_path, "selected_exo")     
        metrics_exo_best = results_exo_best['metrics']

        if metrics_exo_best:
            metrics_tracker_exo_best.update(metrics_exo_best)
            metrics_tracker_exo_best.print_metrics(metrics_exo_best, image_path.split('/')[-1])
            
           
        # Count missing GT files
        if not os.path.exists(gt_path):
            missing_gt += 1

        print("*** End  ", "*"*150)
        print("\n\n")

    # Print final summary
    print("=" * 50)
    print(f"Total number of action-object pairs processed: {total_samples}")
    print(f"Number of missing GT files: {missing_gt}")
    print(f"All images successfully processed!")

if __name__ == "__main__":
    main() 