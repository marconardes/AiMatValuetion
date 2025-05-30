import pytest
from utils.schema import DATA_SCHEMA, MANUAL_ENTRY_CSV_HEADERS

def test_data_schema_import():
    assert isinstance(DATA_SCHEMA, dict)
    assert "material_id" in DATA_SCHEMA # Check for a known key
    assert len(DATA_SCHEMA) > 5 # Check it's not overly empty

def test_manual_entry_csv_headers_import():
    assert isinstance(MANUAL_ENTRY_CSV_HEADERS, list)
    assert "material_id" in MANUAL_ENTRY_CSV_HEADERS # Check for a known header
    assert len(MANUAL_ENTRY_CSV_HEADERS) > 5 # Check it's not overly empty
