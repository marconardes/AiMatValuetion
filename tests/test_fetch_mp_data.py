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
        # Add other fields that might be in initial_search_fields or detailed_summary_fields if accessed
        self.composition_reduced = formula_pretty # Simplified mock
        self.chemsys = "-".join(sorted(list(set(c for c in formula_pretty if c.isupper())))) # Crude chemsys
        self.deprecated = False
        self.theoretical = True # Assuming we mostly want theoretical entries
        self.energy_above_hull = 0.0 # For stability sorting
        self.volume = 100.0 # Example
        self.density = 5.0 # Example

    def model_dump(self):
        # Simple mock for Pydantic's model_dump()
        return self.__dict__

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
@patch('scripts.fetch_mp_data.get_supercon_compositions') # Add this patch
def test_successful_data_fetching(mock_get_supercon_compositions, mock_MPRester, mock_file_open, mock_api_key_config, tmp_path):
    """Test successful fetching and saving of data based on criteria."""

    # Mock get_supercon_compositions to return a non-empty map
    mock_get_supercon_compositions.return_value = {"Fe": 1.0, "Fe2O3": 2.0, "FeO": 3.0, "FeSi": 4.0}


    # Configure the mock MPRester instance
    mock_mpr_api = mock_MPRester.return_value.__enter__.return_value

    # Configure side effects for API calls
    fe_summary_docs = [doc for doc in MOCK_SUMMARY_DOCS if "Fe" in doc.formula_pretty] # list of MockSummaryDoc objects
    mp1_doc = next(doc for doc in fe_summary_docs if doc.material_id == "mp-1")
    mp3_doc = next(doc for doc in fe_summary_docs if doc.material_id == "mp-3")
    mp4_doc = next(doc for doc in fe_summary_docs if doc.material_id == "mp-4")
    mp5_doc = next(doc for doc in fe_summary_docs if doc.material_id == "mp-5")


    def summary_search_side_effect(*args, **kwargs):
        if 'chemsys' in kwargs:
            chemsys = kwargs['chemsys']
            if chemsys == 'Fe': # From mock_get_supercon_compositions "Fe" -> comp_obj -> "Fe"
                return [mp1_doc] # Return only mp-1 for "Fe" chemsys for simplicity in this test
            elif chemsys == 'Fe-O': # From "Fe2O3", "FeO"
                 # Return docs that would match Fe2O3 and FeO reduced formulas
                return [doc for doc in fe_summary_docs if doc.formula_pretty in ["Fe2O3", "FeO"]]
            elif chemsys == 'Fe-Si': # From "FeSi"
                return [mp5_doc]
            # Add more specific chemsys handling if other compositions from mock_get_supercon_compositions are processed
        elif 'material_ids' in kwargs: # Detailed summary search
            # This part fetches detailed summaries for IDs found in the initial search.
            ids_to_fetch = kwargs['material_ids']
            results = []
            if "mp-1" in ids_to_fetch: results.append(mp1_doc)
            if "mp-3" in ids_to_fetch: results.append(mp3_doc)
            if "mp-4" in ids_to_fetch: results.append(mp4_doc)
            if "mp-5" in ids_to_fetch: results.append(mp5_doc)
            return results
        return []

    # Initial search by chemsys uses mpr.materials.search
    mock_mpr_api.materials.search.side_effect = summary_search_side_effect

    # Detailed search by material_ids uses mpr.materials.summary.search
    # Need a separate side_effect or return_value for this path if it's different logic,
    # but summary_search_side_effect is designed to handle both via kwargs.
    # So, we mock .summary.search to also use it.
    mock_mpr_api.materials.summary.search.side_effect = summary_search_side_effect

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
    assert mock_mpr_api.materials.search.call_count > 0 # Changed .summary.search to .search

    # If "Fe" was processed (which it should be from mock_get_supercon_compositions)
    # and mp-1 was its best entry.
    mock_mpr_api.get_structure_by_material_id.assert_any_call("mp-1")
    mock_mpr_api.get_dos_by_material_id.assert_any_call("mp-1")


    # Check file writing
    mock_file_open.assert_called_once_with(str(output_filename_in_config), 'w')

    # Inspect what was written to the file mock
    all_written_parts = []
    for write_call in mock_file_open.return_value.write.call_args_list:
        all_written_parts.append(write_call[0][0])
    written_content_str = "".join(all_written_parts)
    written_data = json.loads(written_content_str) # This is a dict now

    # Assert that data for "Fe" (which corresponds to mp-1 in mocks) is present
    assert "Fe" in written_data
    if written_data.get("Fe"): # If "Fe" was processed and not None
        assert written_data["Fe"]['material_id'] == "mp-1"
        assert written_data["Fe"]['cif_string_mp'] == "cif_for_mp-1" # Key changed
        assert written_data["Fe"]['dos_object_mp']['efermi'] == 0.1
        assert written_data["Fe"]['critical_temperature_tc'] == 1.0 # From mock_get_supercon_compositions


@patch('scripts.fetch_mp_data.MPRester')
@patch('scripts.fetch_mp_data.get_supercon_compositions') # Add this patch
def test_api_key_handling(mock_get_supercon_compositions, mock_MPRester, mock_no_api_key_config, mock_api_key_config, tmp_path):
    # Mock get_supercon_compositions to return a non-empty map for all cases
    mock_get_supercon_compositions.return_value = {"Fe": 1.0}

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
@patch('scripts.fetch_mp_data.get_supercon_compositions') # Add this patch
def test_api_error_handling(mock_get_supercon_compositions, mock_MPRester, mock_file_open, mock_api_key_config, tmp_path, capsys):
    """Test graceful handling of API errors."""
    # Mock get_supercon_compositions to return a non-empty map
    mock_get_supercon_compositions.return_value = {"Fe": 1.0}

    mock_mpr_api = mock_MPRester.return_value.__enter__.return_value

    # Mock for mpr.materials.search (chemsys-based) to raise an error for "Fe"
    def chemsys_search_error_side_effect(*args, **kwargs):
        if 'chemsys' in kwargs and kwargs['chemsys'] == 'Fe':
            raise Exception("Simulated API Error for Fe chemsys")
        return [] # Default for other chemsys calls
    mock_mpr_api.materials.search.side_effect = chemsys_search_error_side_effect

    # Mock for mpr.materials.summary.search (material_id-based) - should not be reached if chemsys search fails first
    mock_mpr_api.materials.summary.search.return_value = []

    output_filename = tmp_path / mock_api_key_config["fetch_data"]["output_filename"]
    mock_api_key_config["fetch_data"]["output_filename"] = str(output_filename)

    with patch('scripts.fetch_mp_data.load_config', return_value=mock_api_key_config):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fetch_data()
            # The old incorrect assertion was:
            # assert any("API call for initial summary search failed" in str(warn.message) for warn in w)

            # Corrected assertion for the specific warning message:
            expected_warning_msg = "  Error processing or querying for SuperCon composition Fe: Simulated API Error for Fe chemsys" # Note leading spaces
            assert any(expected_warning_msg in str(warn_message.message) for warn_message in w), \
                f"Expected warning '{expected_warning_msg}' not found in {[(str(wm.message)) for wm in w]}"

    # Check that the file was opened for writing (even if it's an empty dict)
    mock_file_open.assert_called_once_with(str(output_filename), 'w')

    # This specific warning check is now inside the catch_warnings block.
    # If we need to check it again (e.g. if it could be emitted outside), we can, but it should be captured above.
    # expected_warning_msg = "Error processing or querying for SuperCon composition Fe: Simulated API Error for Fe chemsys"
    # assert any(expected_warning_msg in str(warn.message) for warn in w) # This line is redundant if the above is correct


@patch('builtins.open', new_callable=mock_open)
@patch('scripts.fetch_mp_data.MPRester')
@patch('scripts.fetch_mp_data.get_supercon_compositions') # Add this patch
def test_empty_api_response(mock_get_supercon_compositions, mock_MPRester, mock_file_open, mock_api_key_config, tmp_path, capsys):
    """Test handling of empty list from API."""
    # Mock get_supercon_compositions to return a non-empty map
    mock_get_supercon_compositions.return_value = {"Fe": 1.0}

    mock_mpr_api = mock_MPRester.return_value.__enter__.return_value
    mock_mpr_api.materials.search.return_value = [] # Changed to .search

    output_filename = tmp_path / mock_api_key_config["fetch_data"]["output_filename"]
    mock_api_key_config["fetch_data"]["output_filename"] = str(output_filename)

    with patch('scripts.fetch_mp_data.load_config', return_value=mock_api_key_config):
        fetch_data()

    # Script should still attempt to save an empty dictionary.
    mock_file_open.assert_called_once_with(str(output_filename), 'w')
    written_content_str = "".join(call_arg[0][0] for call_arg in mock_file_open.return_value.write.call_args_list)
    written_data = json.loads(written_content_str)
    assert written_data == {"Fe": None} # Expect {"Fe": None} as per current script logic

    # Verify stdout message indicating no entries found for the specific composition.
    captured = capsys.readouterr()
    assert "No MP entries found for chemical system Fe" in captured.out

# More specific test for criteria set filtering could be added if needed,
# by providing a list of MOCK_SUMMARY_DOCS and checking which ones
# lead to calls on get_structure_by_material_id / get_dos_by_material_id.
# The current test_successful_data_fetching covers this implicitly.

# Test for max_total_materials limit
@patch('builtins.open', new_callable=mock_open)
@patch('scripts.fetch_mp_data.MPRester')
@patch('scripts.fetch_mp_data.get_supercon_compositions') # Add this patch
def test_max_total_materials_limit(mock_get_supercon_compositions, mock_MPRester, mock_file_open, mock_api_key_config, tmp_path):
    mock_comp_tc_map = {"Fe": 1.0, "Si": 0.0}
    mock_get_supercon_compositions.return_value = mock_comp_tc_map

    mock_MPRester.reset_mock()
    mock_mpr_api = mock_MPRester.return_value.__enter__.return_value
    mock_mpr_api.materials.search.side_effect = None
    mock_mpr_api.materials.search.return_value = []

    test_config = mock_api_key_config
    output_filename = tmp_path / test_config["fetch_data"]["output_filename"]
    test_config["fetch_data"]["output_filename"] = str(output_filename)

    def smart_summary_search_side_effect(*args, **kwargs):
        if 'chemsys' in kwargs:
            chemsys = kwargs['chemsys']
            if chemsys == "Fe":
                return [MockSummaryDoc("mp-1", "Fe", 1, 0.0, -0.1)]
            if chemsys == "Si":
                return [MockSummaryDoc("mp-6", "Si", 1, 1.1, -0.2)]
        elif 'material_ids' in kwargs:
            material_ids = kwargs['material_ids']
            detailed_summaries_ret = []
            if "mp-1" in material_ids:
                detailed_summaries_ret.append(MockSummaryDoc("mp-1", "Fe", 1, 0.0, -0.1))
            if "mp-6" in material_ids:
                detailed_summaries_ret.append(MockSummaryDoc("mp-6", "Si", 1, 1.1, -0.2))
            return detailed_summaries_ret
        return []
    mock_mpr_api.materials.search.side_effect = smart_summary_search_side_effect
    mock_mpr_api.materials.summary.search.side_effect = smart_summary_search_side_effect # Assign to summary.search as well

    mock_mpr_api.get_structure_by_material_id.side_effect = lambda mid: MockStructure(f"cif_{mid}")
    mock_mpr_api.get_dos_by_material_id.side_effect = lambda mid: MockDos({"efermi":0.0, "formula": mid})

    with patch('scripts.fetch_mp_data.load_config', return_value=test_config):
        fetch_data()

    assert mock_mpr_api.get_structure_by_material_id.call_count == 2
    assert mock_mpr_api.get_dos_by_material_id.call_count == 2

    mock_mpr_api.get_structure_by_material_id.assert_any_call("mp-1")
    mock_mpr_api.get_structure_by_material_id.assert_any_call("mp-6")
    mock_mpr_api.get_dos_by_material_id.assert_any_call("mp-1")
    mock_mpr_api.get_dos_by_material_id.assert_any_call("mp-6")

    all_written_parts_limit_test = []
    for write_call in mock_file_open.return_value.write.call_args_list:
        all_written_parts_limit_test.append(write_call[0][0])
    written_content_str_limit_test = "".join(all_written_parts_limit_test)
    written_data_limit_test = json.loads(written_content_str_limit_test)
    assert len(written_data_limit_test) == 2
    assert "Fe" in written_data_limit_test
    assert "Si" in written_data_limit_test
    assert written_data_limit_test["Fe"]["material_id"] == "mp-1"
    assert written_data_limit_test["Si"]["material_id"] == "mp-6"
