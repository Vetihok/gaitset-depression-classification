"""Configuration system for GaitSet.

This package provides YAML-based configuration management.
"""

from .loader import load_config, list_available_configs, _load_yaml_file
from .manager import set_conf, conf

__all__ = ['_load_yaml_file', 'load_config', 'list_available_configs', 'set_conf', 'conf']
