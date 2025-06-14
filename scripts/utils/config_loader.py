import yaml
import os
import warnings

DEFAULT_CONFIG_PATH = 'config.yml'

def load_config(config_path=None):
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str, optional): Path to the configuration file.
                                     Defaults to 'config.yml' in the current directory.

    Returns:
        dict: The loaded configuration dictionary. Returns an empty dictionary
              if the file is not found, cannot be parsed, or is empty.
    """
    if config_path is None:
        # Check if running from a script in the main directory or from utils itself
        # This basic check might need to be more robust depending on project structure
        if os.path.basename(os.getcwd()) == 'utils':
            # If in utils, config.yml is one level up
            config_path = os.path.join('..', DEFAULT_CONFIG_PATH)
        else:
            # Assume in main project directory
            config_path = DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        warnings.warn(f"Configuration file '{os.path.abspath(config_path)}' not found. Returning empty config.", UserWarning)
        return {}

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: # Handles empty YAML file
            warnings.warn(f"Configuration file '{os.path.abspath(config_path)}' is empty. Returning empty config.", UserWarning)
            return {}
        return config
    except yaml.YAMLError as e:
        warnings.warn(f"Error parsing YAML file '{os.path.abspath(config_path)}': {e}. Returning empty config.", UserWarning)
        return {}
    except Exception as e:
        warnings.warn(f"An unexpected error occurred while loading config '{os.path.abspath(config_path)}': {e}. Returning empty config.", UserWarning)
        return {}
