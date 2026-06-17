"""Global configuration state manager.

All modules should import `conf_getter` to access configuration.
The main script should call `set_conf()` to initialize the configuration.
"""

_conf_data = {}


def set_conf(config_dict):
    """Set the global configuration dictionary.
    
    Args:
        config_dict: Dictionary containing the configuration
    """
    global _conf_data
    _conf_data.clear()
    _conf_data.update(config_dict)


class _ConfProxy(dict):
    """Proxy to access global configuration as a dictionary."""
    def __init__(self):
        super().__init__()
        
    def __getitem__(self, key):
        return _conf_data[key]
    
    def __setitem__(self, key, value):
        _conf_data[key] = value
    
    def __contains__(self, key):
        return key in _conf_data
    
    def get(self, key, default=None):
        return _conf_data.get(key, default)
    
    def keys(self):
        return _conf_data.keys()
    
    def values(self):
        return _conf_data.values()
    
    def items(self):
        return _conf_data.items()


conf = _ConfProxy()


