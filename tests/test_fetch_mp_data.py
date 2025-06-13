import pytest
import json
import os
import warnings
from unittest.mock import patch, MagicMock, mock_open, call

# Module to be tested
from scripts.fetch_mp_data import fetch_data
# from utils.schema import DATA_SCHEMA # Not directly used by fetch_data, but good for reference if creating complex mocks

# --- Mock Data and Fixtures ---

@pytest.fixture
def mock_api_key_config():
    """Provides a mock config with an API key."""
    return {
        "mp_api_key": "TEST_API_KEY_FROM_CONFIG",
        "fetch_data": {
            "max_total_materials": 5, # Keep low for tests
            "output_filename": "test_output_raw_data.json",
            "criteria_sets": [
                {"target_n_elements": 1, "limit_per_set": 2, "description": "test mono"},
                {"target_n_elements": 2, "limit_per_set": 3, "description": "test binary"},
            ]
        }
    }

@pytest.fixture
def mock_no_api_key_config():
    """Provides a mock config without an API key, relying on environment or default."""
    return {
        "mp_api_key": "YOUR_MP_API_KEY", # Placeholder, so it should try env var
        "fetch_data": {
            "max_total_materials": 3,
            "output_filename": "test_output_no_api_key.json",
            "criteria_sets": [
                {"target_n_elements": 1, "limit_per_set": 3, "description": "test mono"},
            ]
        }
    }

# --- MPRester Mock Data ---

# Mock for SummaryDoc objects
class MockSummaryDoc:
    def __init__(self, material_id, formula_pretty, nelements, band_gap, formation_energy_per_atom):
        self.material_id = material_id
        self.formula_pretty = formula_pretty
        self.nelements = nelements
        self.band_gap = band_gap
        self.formation_energy_per_atom = formation_energy_per_atom

# Mock for Structure objects
class MockStructure:
    def __init__(self, cif_content="mock cif data"):
        self._cif_content = cif_content
    def to(self, fmt):
        if fmt == "cif":
            return self._cif_content
        raise ValueError(f"Unsupported format: {fmt}")

# Mock for DOS objects
class MockDos:
    def __init__(self, dos_data={"efermi": 0.0, "densities": {}, "energies": []}):
        self._dos_data = dos_data
    def as_dict(self):
        return self._dos_data

# Sample data MPRester might return
MOCK_SUMMARY_DOCS = [
    MockSummaryDoc("mp-1", "Fe", 1, 0.0, -0.1),
    MockSummaryDoc("mp-2", "O2", 1, 1.5, 0.0), # Should be filtered by elements=["Fe"] in query for summary
    MockSummaryDoc("mp-3", "Fe2O3", 2, 2.0, -2.5),
    MockSummaryDoc("mp-4", "FeO", 2, 0.0, -1.5),
    MockSummaryDoc("mp-5", "FeSi", 2, 0.0, -0.5),
    MockSummaryDoc("mp-6", "Si", 1, 1.1, -0.2), # Should be filtered by elements=["Fe"]
    MockSummaryDoc("mp-7", "Fe3O4", 2, 0.1, -2.0)
]

MOCK_STRUCTURE_MP1 = MockStructure("cif_for_mp-1")
MOCK_DOS_MP1 = MockDos({"efermi": 0.1, "densities": {"1": [1,2]}, "energies": [0,1]})

MOCK_STRUCTURE_MP3 = MockStructure("cif_for_mp-3")
MOCK_DOS_MP3 = MockDos({"efermi": 0.3, "densities": {"1": [3,4]}, "energies": [0,1]})

MOCK_STRUCTURE_MP4 = MockStructure("cif_for_mp-4")
MOCK_DOS_MP4 = MockDos({"efermi": 0.4, "densities": {"1": [5,6]}, "energies": [0,1]})

MOCK_STRUCTURE_MP5 = MockStructure("cif_for_mp-5")
MOCK_DOS_MP5 = MockDos({"efermi": 0.5, "densities": {"1": [7,8]}, "energies": [0,1]})

MOCK_STRUCTURE_MP7 = MockStructure("cif_for_mp-7")
MOCK_DOS_MP7 = MockDos({"efermi": 0.7, "densities": {"1": [9,10]}, "energies": [0,1]})


# --- Test Cases ---

@patch('builtins.open', new_callable=mock_open)
@patch('scripts.fetch_mp_data.MPRester')
def test_successful_data_fetching(mock_MPRester, mock_file_open, mock_api_key_config, tmp_path):
    """Test successful fetching and saving of data based on criteria."""

    # Configure the mock MPRester instance
    mock_mpr_api = mock_MPRester.return_value.__enter__.return_value

    # Configure side effects for API calls
    # materials.summary.search should return Fe-containing materials
    fe_summary_docs = [doc for doc in MOCK_SUMMARY_DOCS if "Fe" in doc.formula_pretty]
    mock_mpr_api.materials.summary.search.return_value = fe_summary_docs

    def get_structure_side_effect(material_id):
        if material_id == "mp-1": return MOCK_STRUCTURE_MP1
        if material_id == "mp-3": return MOCK_STRUCTURE_MP3
        if material_id == "mp-4": return MOCK_STRUCTURE_MP4
        if material_id == "mp-5": return MOCK_STRUCTURE_MP5
        if material_id == "mp-7": return MOCK_STRUCTURE_MP7
        return None
    mock_mpr_api.get_structure_by_material_id.side_effect = get_structure_side_effect

    def get_dos_side_effect(material_id):
        if material_id == "mp-1": return MOCK_DOS_MP1
        if material_id == "mp-3": return MOCK_DOS_MP3
        if material_id == "mp-4": return MOCK_DOS_MP4
        if material_id == "mp-5": return MOCK_DOS_MP5
        if material_id == "mp-7": return MOCK_DOS_MP7
        return None
    mock_mpr_api.get_dos_by_material_id.side_effect = get_dos_side_effect

    # Mock load_config to return our controlled config
    # The output filename will be inside tmp_path due to how config_loader handles relative paths if needed
    # but here, fetch_data gets it from config, so we don't need to worry about tmp_path directly in fetch_data call.
    # However, the test_output_raw_data.json will be "written" relative to current dir if not path joined.
    # Let's ensure the output_filename in config is a full path for clarity in test.
    output_filename_in_config = tmp_path / mock_api_key_config["fetch_data"]["output_filename"]
    mock_api_key_config["fetch_data"]["output_filename"] = str(output_filename_in_config)

    with patch('scripts.fetch_mp_data.load_config', return_value=mock_api_key_config):
        fetch_data() # max_total_materials is controlled by config

    # Assertions
    mock_MPRester.assert_called_once_with(api_key="TEST_API_KEY_FROM_CONFIG")
    mock_mpr_api.materials.summary.search.assert_called_once_with(
        elements=["Fe"], band_gap=(0,100), fields=["material_id", "formula_pretty", "nelements", "band_gap", "formation_energy_per_atom"]
    )

    # Check calls for structure and DOS based on criteria_sets and limits
    # Config: mono (limit 2), binary (limit 3). Max total 5.
    # Fe-containing docs from MOCK_SUMMARY_DOCS that are Fe-related:
    # mp-1 (Fe, nelem=1)
    # mp-3 (Fe2O3, nelem=2)
    # mp-4 (FeO, nelem=2)
    # mp-5 (FeSi, nelem=2)
    # mp-7 (Fe3O4, nelem=2)
    # Processing:
    # 1. Set "test mono" (nelem=1, limit_per_set=2): mp-1 is fetched. (Total fetched = 1)
    # 2. Set "test binary" (nelem=2, limit_per_set=3): mp-3, mp-4, mp-5 are fetched. (Total fetched = 1+3 = 4)
    # mp-7 is not fetched because limit_per_set for binaries (3) is met by mp-3, mp-4, mp-5.
    # Max_total_materials (5) is not reached by these 4.
    expected_detail_calls = ["mp-1", "mp-3", "mp-4", "mp-5"]

    assert mock_mpr_api.get_structure_by_material_id.call_count == len(expected_detail_calls)
    assert mock_mpr_api.get_dos_by_material_id.call_count == len(expected_detail_calls)

    for mid in expected_detail_calls:
        mock_mpr_api.get_structure_by_material_id.assert_any_call(mid)
        mock_mpr_api.get_dos_by_material_id.assert_any_call(mid)

    # Check file writing
    mock_file_open.assert_called_once_with(str(output_filename_in_config), 'w')

    # Inspect what was written to the file mock
    # Consolidate all data passed to write calls
    all_written_parts = []
    for write_call in mock_file_open.return_value.write.call_args_list:
        all_written_parts.append(write_call[0][0])
    written_content_str = "".join(all_written_parts)
    written_data = json.loads(written_content_str)

    assert len(written_data) == len(expected_detail_calls)
    assert written_data[0]['material_id'] == "mp-1"
    assert written_data[0]['cif_string'] == "cif_for_mp-1"
    assert written_data[0]['dos_object_mp']['efermi'] == 0.1
    assert written_data[1]['material_id'] == "mp-3" # First binary after filtering by criteria


@patch('scripts.fetch_mp_data.MPRester')
def test_api_key_handling(mock_MPRester, mock_no_api_key_config, mock_api_key_config, tmp_path):
    # Case 1: API key from config
    mock_MPRester.reset_mock()
    output_fn_1 = tmp_path / mock_api_key_config["fetch_data"]["output_filename"]
    mock_api_key_config["fetch_data"]["output_filename"] = str(output_fn_1)
    with patch('scripts.fetch_mp_data.load_config', return_value=mock_api_key_config):
        with patch('builtins.open', mock_open()): # Mock file open to prevent actual write
            fetch_data()
            mock_MPRester.assert_called_with(api_key="TEST_API_KEY_FROM_CONFIG")

    # Case 2: API key from environment variable
    mock_MPRester.reset_mock()
    output_fn_2 = tmp_path / mock_no_api_key_config["fetch_data"]["output_filename"] # ensure unique filename
    mock_no_api_key_config["fetch_data"]["output_filename"] = str(output_fn_2)
    with patch('scripts.fetch_mp_data.load_config', return_value=mock_no_api_key_config):
        with patch.dict(os.environ, {"MP_API_KEY": "TEST_ENV_KEY"}):
            with patch('builtins.open', mock_open()):
                fetch_data()
                mock_MPRester.assert_called_with(api_key="TEST_ENV_KEY")

    # Case 3: No API key (should use None, potentially warn)
    mock_MPRester.reset_mock()
    output_fn_3 = tmp_path / mock_no_api_key_config["fetch_data"]["output_filename"].replace(".json", "_3.json")
    mock_no_api_key_config["fetch_data"]["output_filename"] = str(output_fn_3)
    with patch('scripts.fetch_mp_data.load_config', return_value=mock_no_api_key_config):
        with patch.dict(os.environ, {}, clear=True): # Clear MP_API_KEY from env
             with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with patch('builtins.open', mock_open()):
                    fetch_data()
                mock_MPRester.assert_called_with(api_key=None) # Expects None when no key found
                assert any("MP_API_KEY not found" in str(warn.message) for warn in w)


@patch('builtins.open', new_callable=mock_open)
@patch('scripts.fetch_mp_data.MPRester')
def test_api_error_handling(mock_MPRester, mock_file_open, mock_api_key_config, tmp_path, capsys):
    """Test graceful handling of API errors."""
    mock_mpr_api = mock_MPRester.return_value.__enter__.return_value
    mock_mpr_api.materials.summary.search.side_effect = Exception("Simulated API Error")

    output_filename = tmp_path / mock_api_key_config["fetch_data"]["output_filename"]
    mock_api_key_config["fetch_data"]["output_filename"] = str(output_filename)

    with patch('scripts.fetch_mp_data.load_config', return_value=mock_api_key_config):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fetch_data()
            assert any("API call for initial summary search failed" in str(warn.message) for warn in w)

    # Script should exit before attempting to save if summary search fails.
    mock_file_open.assert_not_called()

    captured = capsys.readouterr()
    assert "No initial candidate materials found. Exiting." in captured.out # From the script
    # "No data collected to save." is not reached due to early exit.


@patch('builtins.open', new_callable=mock_open)
@patch('scripts.fetch_mp_data.MPRester')
def test_empty_api_response(mock_MPRester, mock_file_open, mock_api_key_config, tmp_path, capsys):
    """Test handling of empty list from API."""
    mock_mpr_api = mock_MPRester.return_value.__enter__.return_value
    mock_mpr_api.materials.summary.search.return_value = [] # Empty list

    output_filename = tmp_path / mock_api_key_config["fetch_data"]["output_filename"]
    mock_api_key_config["fetch_data"]["output_filename"] = str(output_filename)

    with patch('scripts.fetch_mp_data.load_config', return_value=mock_api_key_config):
        fetch_data()

    # Script should exit before attempting to save if summary search returns empty.
    mock_file_open.assert_not_called()

    captured = capsys.readouterr()
    assert "No initial candidate materials found. Exiting." in captured.out
    # "No data collected to save." is not reached due to early exit.

# More specific test for criteria set filtering could be added if needed,
# by providing a list of MOCK_SUMMARY_DOCS and checking which ones
# lead to calls on get_structure_by_material_id / get_dos_by_material_id.
# The current test_successful_data_fetching covers this implicitly.

# Test for max_total_materials limit
@patch('builtins.open', new_callable=mock_open)
@patch('scripts.fetch_mp_data.MPRester')
def test_max_total_materials_limit(mock_MPRester, mock_file_open, mock_api_key_config, tmp_path):
    mock_mpr_api = mock_MPRester.return_value.__enter__.return_value

    # Provide more docs than max_total_materials allows
    # Config has max_total_materials = 5
    # Criteria: mono (limit 2), binary (limit 3)
    # Fe-docs: mp-1 (mono), mp-3, mp-4, mp-5, mp-7 (all binary)
    # If all were processed, it would be 5. Let's make max_total_materials smaller.

    test_config = mock_api_key_config.copy() # Deep copy might be better if nested dicts are modified
    test_config["fetch_data"] = test_config["fetch_data"].copy()
    test_config["fetch_data"]["max_total_materials"] = 2 # Set a small limit

    output_filename = tmp_path / test_config["fetch_data"]["output_filename"]
    test_config["fetch_data"]["output_filename"] = str(output_filename)

    # Return enough Fe-containing docs to hit the limit
    fe_summary_docs = [doc for doc in MOCK_SUMMARY_DOCS if "Fe" in doc.formula_pretty]
    mock_mpr_api.materials.summary.search.return_value = fe_summary_docs

    # Mock detail calls to return valid objects so processing continues
    mock_mpr_api.get_structure_by_material_id.side_effect = lambda mid: MockStructure(f"cif_{mid}")
    mock_mpr_api.get_dos_by_material_id.side_effect = lambda mid: MockDos({"efermi":0})

    with patch('scripts.fetch_mp_data.load_config', return_value=test_config):
        fetch_data()

    # Should fetch mp-1 (mono, 1st set, count=1)
    # Then try mp-3 (binary, 2nd set, count=2). Limit reached.
    # So, only mp-1 and mp-3 should be fully processed.
    assert mock_mpr_api.get_structure_by_material_id.call_count == 2
    assert mock_mpr_api.get_dos_by_material_id.call_count == 2

    mock_mpr_api.get_structure_by_material_id.assert_any_call("mp-1")
    mock_mpr_api.get_structure_by_material_id.assert_any_call("mp-3")

    all_written_parts_limit_test = []
    for write_call in mock_file_open.return_value.write.call_args_list:
        all_written_parts_limit_test.append(write_call[0][0])
    written_content_str_limit_test = "".join(all_written_parts_limit_test)
    written_data_limit_test = json.loads(written_content_str_limit_test)
    assert len(written_data_limit_test) == 2
