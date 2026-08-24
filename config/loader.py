"""Load configuration from YAML files."""

import os
import yaml

from env_manager import EnvManager

def load_config(env: EnvManager, config_id):
    """Load configuration from a YAML file based on config_id.
    
    Args:
        config_id: Integer or string identifier (e.g., 1 loads config_1.yaml)
    
    Returns:
        Dictionary containing the configuration
    
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    config_file = os.path.join(env.get_configs_dir(), f"config_{config_id}.yaml")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Config file 'config_{config_id}.yaml' not found in {env.get_configs_dir()}. "
            f"Available configs: {list_available_configs()}"
        )
    
    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))

    yaml.SafeLoader.add_constructor('!tuple', tuple_constructor)
    with open(config_file, 'r') as f:
        conf = yaml.safe_load(f)
    
    if not isinstance(conf, dict):
        raise ValueError(f"Config file '{config_file}' must contain a YAML dictionary at the root level")
    
    return conf

def _load_yaml_file(path):
    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))
    yaml.SafeLoader.add_constructor('!tuple', tuple_constructor)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}

def list_available_configs(env: EnvManager):
    """List all available config_*.yaml files in the config directory."""
    configs = []
    for filename in sorted(os.listdir(env.get_configs_dir())):
        if filename.startswith('config_') and filename.endswith('.yaml'):
            configs.append(filename)
    return configs
