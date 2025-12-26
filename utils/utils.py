import yaml

def load_property_config(filepath="properties.yaml"):
    """
    Loads the property configuration from a YAML file.
    
    How to use:
    PROPERTY_CONFIG = load_property_config("properties.yaml")
    """
    with open(filepath, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error loading YAML: {exc}")
            return None

def get_property_details(config, property_name):
    """
    Recursively searches for a property name in a nested configuration.
    Returns the dictionary associated with that property.

    How to use:
    property_details = get_property_details(PROPERTY_CONFIG, "solubility")
    """
    if property_name in config:
        if isinstance(config[property_name], dict) and 'target' in config[property_name]:
            return config[property_name]

    for key, value in config.items():
        if isinstance(value, dict):
            if property_name in value:
                return value[property_name]
    
    return None

def extract_property_keys(config):
    """
    Recursively finds all keys that represent actual properties 
    (leaves of the config tree that contain 'target' or 'threshold').
    """
    keys = []
    for key, value in config.items():
        if isinstance(value, dict):
            if 'target' in value or 'threshold' in value:
                keys.append(key)
            else:
                keys.extend(extract_property_keys(value))
    return keys