import os

def update_model_name_in_config(model):
    """
    Updates the model name in the default.yaml configuration file for fine-tuning.
    """
    config_path = f"tools/cli/conf/finetune/default.yaml"
    
    try:
        # Read the current config file
        with open(config_path, 'r') as file:
            lines = file.readlines()
        
        # Find and update the model line
        for i, line in enumerate(lines):
            if line.strip().startswith("- model:"):
                lines[i] = f"  - model: {model} \n"
                break
        
        # Write the updated config back to the file
        with open(config_path, 'w') as file:
            file.writelines(lines)
        
        print(f"Updated model in config to: {model}")
    except Exception as e:
        print(f"Error updating model name in config: {e}")

def update_patch_size_in_config(patch_size):
    """
    Updates the patch_sizes in the val_dataset.yaml configuration file to match the current patch_size.
    """
    import yaml
    
    yaml_path = "./tools/cli/conf/finetune/val_data/val_dataset.yaml"
    
    try:
        # Read the current yaml file
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Update the patch_sizes value
        # If patch_size is "auto", keep the existing configuration
        if patch_size != "auto":
            # Replace the patch_sizes with the current patch_size and patch_size*2
            config['_args_']['patch_sizes'] = [patch_size, patch_size*2]
        
        # Write the updated config back to the file
        with open(yaml_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(f"Updated patch_sizes in {yaml_path} to {config['_args_']['patch_sizes']}")
    except Exception as e:
        print(f"Error updating patch_sizes in val_dataset.yaml: {e}")

def update_batch_size_in_config(batch_size):
    """
    Updates the batch_size in the default.yaml configuration file for train and validation dataloaders.
    """
    config_path = f"tools/cli/conf/finetune/default.yaml"
    
    try:
        # Read the current config file
        with open(config_path, 'r') as file:
            lines = file.readlines()
        
        # Find and update the batch_size lines for both train and val dataloaders
        for i, line in enumerate(lines):
            if "batch_size:" in line and ("train_dataloader" in "".join(lines[max(0, i-10):i]) or "val_dataloader" in "".join(lines[max(0, i-10):i])):
                leading_spaces = len(line) - len(line.lstrip())
                lines[i] = " " * leading_spaces + f"batch_size: {batch_size} \n"
        
        # Write the updated config back to the file
        with open(config_path, 'w') as file:
            file.writelines(lines)
        
        print(f"Updated batch_size in config to: {batch_size}")
    except Exception as e:
        print(f"Error updating batch_size in config: {e}")

def update_num_epochs_in_config(num_epochs):
    """
    Updates the max_epochs in the default.yaml configuration file for the trainer.
    """
    config_path = f"tools/cli/conf/finetune/default.yaml"
    
    try:
        # Read the current config file
        with open(config_path, 'r') as file:
            lines = file.readlines()
        
        # Find and update the max_epochs line
        for i, line in enumerate(lines):
            if line.strip().startswith("max_epochs:"):
                leading_spaces = len(line) - len(line.lstrip())
                lines[i] = " " * leading_spaces + f"max_epochs: {num_epochs}\n"
                break
        
        # Write the updated config back to the file
        with open(config_path, 'w') as file:
            file.writelines(lines)
        
        print(f"Updated max_epochs in config to: {num_epochs}")
    except Exception as e:
        print(f"Error updating max_epochs in config: {e}")

def update_patience_in_config(patience):
    """
    Updates the patience in the default.yaml configuration file for the trainer.
    """
    config_path = f"tools/cli/conf/finetune/default.yaml"
    
    try:
        # Read the current config file
        with open(config_path, 'r') as file:
            lines = file.readlines()
        
        # Find and update the patience line
        for i, line in enumerate(lines):
            if line.strip().startswith("patience:"):
                leading_spaces = len(line) - len(line.lstrip())
                lines[i] = " " * leading_spaces + f"patience: {patience}\n"
                break
        
        # Write the updated config back to the file
        with open(config_path, 'w') as file:
            file.writelines(lines)
        
        print(f"Updated patience in config to: {patience}")
    except Exception as e:
        print(f"Error updating patience in config: {e}")

def update_context_length_in_config(context_length):
    """
    Updates the context_length in the val_dataset.yaml configuration file.
    """
    import yaml
    
    yaml_path = "./tools/cli/conf/finetune/val_data/val_dataset.yaml"
    
    try:
        # Read the current yaml file
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Update the context_length value as a single-element list
        config['_args_']['context_lengths'] = [context_length]
        
        # Write the updated config back to the file
        with open(yaml_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        print(f"Updated context_lengths in {yaml_path} to [{context_length}]")
    except Exception as e:
        print(f"Error updating context_length in val_dataset.yaml: {e}")

def update_seed_in_config(seed):
    """
    Updates the seed in the default.yaml configuration file.
    """
    config_path = f"tools/cli/conf/finetune/default.yaml"
    
    try:
        # Read the current config file
        with open(config_path, 'r') as file:
            lines = file.readlines()
        
        # Find and update the seed line
        for i, line in enumerate(lines):
            if line.strip().startswith("seed:"):
                leading_spaces = len(line) - len(line.lstrip())
                lines[i] = " " * leading_spaces + f"seed: {seed}\n"
                break
        
        # Write the updated config back to the file
        with open(config_path, 'w') as file:
            file.writelines(lines)
        
        print(f"Updated seed in config to: {seed}")
    except Exception as e:
        print(f"Error updating seed in config: {e}")

def update_ft_schedule_in_config(ft_schedule, model_name):
    """
    Updates the ft_schedule path in the default_fts.yaml configuration file.
    The path is constructed based on the ft_schedule value and points to the generated schedule file.
    """
    import os
    
    config_path = f"tools/cli/conf/finetune/default.yaml"
    
    try:
        # Get the absolute path to the workspace root
        workspace_root = os.path.abspath(".")
        
        # Construct the schedule file path based on ft_schedule value
        schedule_file_path = f"{workspace_root}/tools/cli/finetune_scheduler_schedules/{model_name}/{ft_schedule}_schedule.yaml"
        
        # Read the current config file
        with open(config_path, 'r') as file:
            lines = file.readlines()
        
        # Find and update the ft_schedule line
        for i, line in enumerate(lines):
            if line.strip().startswith("ft_schedule:"):
                leading_spaces = len(line) - len(line.lstrip())
                lines[i] = " " * leading_spaces + f"ft_schedule: {schedule_file_path}\n"
                break
        
        # Write the updated config back to the file
        with open(config_path, 'w') as file:
            file.writelines(lines)
        
        print(f"Updated ft_schedule in config to: {schedule_file_path}")
    except Exception as e:
        print(f"Error updating ft_schedule in config: {e}") 