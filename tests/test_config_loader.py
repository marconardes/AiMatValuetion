import pytest
import os
import yaml
from utils.config_loader import load_config
import warnings

TEST_CONFIG_VALID = {
    'general': {'setting1': 'value1'},
    'paths': {'data': '/path/to/data'}
}
TEST_CONFIG_EMPTY_CONTENT = {} # For when the file is empty

@pytest.fixture
def temp_config_file(tmp_path):
    def _create_config(content, filename="test_config.yml", valid_yaml=True, empty_file=False):
        file_path = tmp_path / filename
        if empty_file:
            open(file_path, 'w').close() # Create an empty file
        elif valid_yaml:
            with open(file_path, 'w') as f:
                yaml.dump(content, f)
        else:
            with open(file_path, 'w') as f:
                f.write("invalid: yaml: content:") # Malformed YAML
        return str(file_path)
    return _create_config

def test_load_valid_config(temp_config_file):
    config_path = temp_config_file(TEST_CONFIG_VALID)
    loaded = load_config(config_path)
    assert loaded == TEST_CONFIG_VALID

def test_load_missing_config():
    # Capture warnings for this specific test
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always") # Cause all warnings to always be triggered.
        config = load_config("non_existent_config.yml")
        assert config == {} # Should return an empty dict as per implementation

        # Check that a warning was issued
        assert len(w) > 0
        # Check that the warning message contains the expected text
        # Note: The actual path in the warning might be absolute.
        # We check for the presence of the filename.
        found_warning = False
        for warn_item in w:
            if "Configuration file 'non_existent_config.yml' not found" in str(warn_item.message) or \
               "Configuration file" in str(warn_item.message) and "non_existent_config.yml' not found" in str(warn_item.message):
                found_warning = True
                break
        assert found_warning, "Expected 'file not found' warning was not issued."


def test_load_malformed_config(temp_config_file):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config_path = temp_config_file(None, valid_yaml=False) # content=None, not valid yaml
        loaded = load_config(config_path)
        assert loaded == {}

        assert len(w) > 0
        found_warning = False
        for warn_item in w:
            if "Error parsing YAML file" in str(warn_item.message):
                found_warning = True
                break
        assert found_warning, "Expected 'Error parsing YAML file' warning was not issued."

def test_load_empty_yaml_file(temp_config_file):
    # This test is for a file that IS YAML but just empty (e.g. contains "{}")
    # or a file that is literally empty (0 bytes)

    # Test 1: YAML file with empty dictionary representation (e.g., created from yaml.dump({}, f))
    # This should load as an empty dict, and no "is empty" warning should be issued,
    # as it's a valid YAML document representing an empty mapping.
    config_path_yaml_empty_dict = temp_config_file(TEST_CONFIG_EMPTY_CONTENT) # file with "{}"
    loaded_yaml_empty_dict = load_config(config_path_yaml_empty_dict)
    assert loaded_yaml_empty_dict == {}

    # Test 2: Literally empty file (0 bytes)
    # This should load as None from yaml.safe_load, and our function should then
    # convert it to {} and issue the "is empty" warning.
    with warnings.catch_warnings(record=True) as w_actual_empty:
        warnings.simplefilter("always")
        config_path_actual_empty = temp_config_file(None, empty_file=True) # 0-byte file
        loaded_actual_empty = load_config(config_path_actual_empty)
        assert loaded_actual_empty == {} # Should also be an empty dict

        assert len(w_actual_empty) > 0, "A warning should be issued for a 0-byte file."
        found_warning_actual_empty = False
        for warn_item in w_actual_empty:
            # yaml.safe_load(f) on an empty file returns None.
            # The load_config function then warns that the file "is empty".
            if "is empty. Returning empty config." in str(warn_item.message):
                found_warning_actual_empty = True
                break
        assert found_warning_actual_empty, "Expected 'is empty' warning for 0-byte file was not issued."
