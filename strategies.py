import json
from indicators import IndicatorRegistry # Assuming indicators.py is in the same directory or accessible

# Attempt to import jsonschema, but allow fallback if not available
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "indicators": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "params": {"type": "object"}
                },
                "required": ["id", "type", "params"]
            }
        },
        "conditions": {
            "type": "object",
            "properties": {
                "entry_long": {"type": "array", "items": {"type": "object"}},
                "exit_long": {"type": "array", "items": {"type": "object"}},
                "entry_short": {"type": "array", "items": {"type": "object"}},
                "exit_short": {"type": "array", "items": {"type": "object"}}
            }
            # "required": [] # No specific conditions required by default for now
        }
    },
    "required": ["version", "name", "indicators", "conditions"]
}

class StrategyConfigLoader:
    def __init__(self, indicator_registry: IndicatorRegistry):
        if not isinstance(indicator_registry, IndicatorRegistry):
            raise TypeError("indicator_registry must be an instance of IndicatorRegistry")
        self.indicator_registry = indicator_registry

    def load_strategy_from_json(self, file_path: str) -> dict:
        try:
            with open(file_path, 'r') as f:
                raw_content = f.read()
                # Handle empty or whitespace-only files before JSON decoding
                if not raw_content.strip():
                    raise ValueError(f"File '{file_path}' is empty or contains only whitespace.")
                parsed_data = json.loads(raw_content)
        except FileNotFoundError:
            raise FileNotFoundError(f"Strategy file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error decoding JSON from {file_path}: {e.msg}", e.doc, e.pos)
        
        self._validate_schema(parsed_data)
        
        try:
            initialized_indicators = self._initialize_indicators(parsed_data.get('indicators', []))
            parsed_data['initialized_indicators'] = initialized_indicators
        except ValueError as e:
            # Propagate errors from indicator initialization (e.g., unknown type, invalid params)
            raise ValueError(f"Error initializing indicators for strategy from {file_path}: {e}")
            
        return parsed_data

    def _validate_schema(self, config_data: dict):
        if JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(instance=config_data, schema=STRATEGY_SCHEMA)
            except jsonschema.exceptions.ValidationError as e:
                # Simplify the error message a bit for clarity
                raise ValueError(f"Strategy schema validation failed: {e.message} (Path: {'/'.join(map(str, e.path))})")
        else:
            # Perform basic manual checks
            for required_key in STRATEGY_SCHEMA.get("required", []):
                if required_key not in config_data:
                    raise ValueError(f"Missing required top-level key in strategy config: '{required_key}'")

            indicators_schema = STRATEGY_SCHEMA.get("properties", {}).get("indicators", {})
            if "indicators" in config_data:
                if not isinstance(config_data["indicators"], list):
                    raise ValueError("'indicators' must be a list.")
                for i, indicator_conf in enumerate(config_data["indicators"]):
                    if not isinstance(indicator_conf, dict):
                        raise ValueError(f"Each item in 'indicators' must be an object (dictionary). Found: {type(indicator_conf)} at index {i}")
                    
                    item_schema_props = indicators_schema.get("items", {}).get("properties", {})
                    item_schema_required = indicators_schema.get("items", {}).get("required", [])

                    for req_prop in item_schema_required:
                        if req_prop not in indicator_conf:
                            raise ValueError(f"Missing required key '{req_prop}' in indicator config at index {i}: {indicator_conf}")
                    
                    for prop_key, prop_schema in item_schema_props.items():
                        if prop_key in indicator_conf:
                            expected_type_str = prop_schema.get("type")
                            python_type = None
                            if expected_type_str == "string": python_type = str
                            elif expected_type_str == "object": python_type = dict
                            # Add more type mappings if needed for manual check
                            
                            if python_type and not isinstance(indicator_conf[prop_key], python_type):
                                raise ValueError(f"Indicator property '{prop_key}' at index {i} has incorrect type. Expected {expected_type_str}, got {type(indicator_conf[prop_key]).__name__}.")
            
            conditions_schema = STRATEGY_SCHEMA.get("properties", {}).get("conditions", {})
            if "conditions" in config_data:
                 if not isinstance(config_data["conditions"], dict):
                    raise ValueError("'conditions' must be an object (dictionary).")
                 # Further checks for condition structure can be added here if needed without jsonschema


    def _initialize_indicators(self, indicator_configs: list) -> dict:
        initialized_indicators = {}
        if not isinstance(indicator_configs, list):
            # This should ideally be caught by schema validation
            raise ValueError("Indicator configurations must be a list.")

        for i, config in enumerate(indicator_configs):
            if not isinstance(config, dict):
                 raise ValueError(f"Indicator configuration at index {i} is not a dictionary: {config}")

            indicator_id = config.get('id')
            indicator_type = config.get('type')
            params = config.get('params', {})

            if not indicator_id or not indicator_type:
                raise ValueError(f"Indicator configuration at index {i} is missing 'id' or 'type': {config}")
            if not isinstance(params, dict):
                raise ValueError(f"Indicator 'params' for id '{indicator_id}' must be a dictionary. Found: {type(params)}")

            try:
                # Pass params as keyword arguments
                indicator_instance = self.indicator_registry.get_indicator(name=indicator_type, **params)
                initialized_indicators[indicator_id] = indicator_instance
            except ValueError as e: # Catch errors from get_indicator (unknown type, invalid params)
                raise ValueError(f"Error initializing indicator id '{indicator_id}' (type: '{indicator_type}'): {e}")
            except TypeError as e: # Catch potential TypeError if params are not passed correctly
                 raise TypeError(f"Error with parameters for indicator id '{indicator_id}' (type: '{indicator_type}'): {e}")


        return initialized_indicators
