"""Configuration system for GaitSet.

This package provides YAML-based configuration management.
"""

from .loader import load_config, list_available_configs
from .manager import set_conf, conf

__all__ = ['load_config', 'list_available_configs', 'set_conf', 'conf']
