import unittest
import os
import json
import pandas as pd
import numpy as np # For np.nan
import torch
from torch_geometric.data import Data
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice

# Adjust the import path based on how the script will be run or PYTHONPATH
# Assuming PYTHONPATH includes the project root, so 'scripts' is a top-level package
from scripts.prepare_gnn_data import create_graph_dataset

class TestPrepareGNNData(unittest.TestCase):
    test_csv_path = "data/test_suite_oqmd_data.csv"
    test_config_path = "config_test_suite.yml"

    processed_graphs_path = "data/test_suite_processed_graphs.pt"
    train_graphs_path = "data/test_suite_train_graphs.pt"
    val_graphs_path = "data/test_suite_val_graphs.pt"
    test_graphs_path = "data/test_suite_test_graphs.pt"

    def setUp(self):
        # Add new material "FeCube1" with 3x3 matrix lattice
        csv_data = {
            'supercon_composition': ['NaCl1', 'Si1', 'InvalidStruct1', 'MissingTarget1', 'MissingCell1', 'FeCube1'],
            'oqmd_entry_id': [1, 2, 3, 4, 5, 6],
            'oqmd_formula': ['NaCl', 'Si', 'Invalid', 'Al', 'MissingCell', 'Fe'],
            'oqmd_spacegroup': ['Fm-3m', 'Fd-3m', 'Pm-3m', 'Pm-3m', 'Pm-3m', 'Pm-3m'],
            'oqmd_delta_e': [-3.0, 0.1, -1.0, -0.5, -0.7, -0.2], # Added -0.2 for FeCube1
            'oqmd_stability': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'oqmd_band_gap': [2.5, 0.5, 1.0, np.nan, 0.9, 0.1], # Added 0.1 for FeCube1
            'oqmd_prototype': ['NaCl', 'Si', 'Invalid', 'Al', 'MissingCell', 'Fe'],
            'oqmd_unit_cell_json': [
                "[5.64, 5.64, 5.64, 90.0, 90.0, 90.0]",  # NaCl (6 params)
                "[3.867, 3.867, 3.867, 90.0, 90.0, 90.0]",# Si (6 params)
                "[3.0, 3.0, 3.0, 90.0, 90.0, 90.0]",     # InvalidStruct1 (sites json is malformed)
                "[2.0, 2.0, 2.0, 90.0, 90.0, 90.0]",     # MissingTarget1 (Al, 6 params, band_gap is nan)
                None,                                    # MissingCell1 (cell json is None)
                json.dumps([[3.5, 0.0, 0.0], [0.0, 3.5, 0.0], [0.0, 0.0, 3.5]]) # FeCube1 (3x3 matrix)
            ],
            'oqmd_sites_json': [
                json.dumps([{"species": [{"element": "Na", "occu": 1.0}], "xyz": [0.0, 0.0, 0.0]},
                              {"species": [{"element": "Cl", "occu": 1.0}], "xyz": [0.5, 0.5, 0.5]}]), # NaCl (2 sites)
                json.dumps([{"species": [{"element": "Si", "occu": 1.0}], "xyz": [0.0, 0.0, 0.0]},
                              {"species": [{"element": "Si", "occu": 1.0}], "xyz": [0.25, 0.25, 0.25]}]),# Si (2 sites)
                "[{species: [{\"element\": \"X\", \"occu\": 1.0}], \"xyz\": [0.0, 0.0, 0.0]}]", # InvalidStruct1 (malformed)
                json.dumps([{"species": [{"element": "Al", "occu": 1.0}], "xyz": [0.0, 0.0, 0.0]}]), # MissingTarget1 (Al, 1 site)
                json.dumps([{"species": [{"element": "Y", "occu": 1.0}], "xyz": [0.0, 0.0, 0.0]}]), # MissingCell1
                json.dumps([{"species": [{"element": "Fe", "occu": 1.0}], "xyz": [0.0, 0.0, 0.0]}])  # FeCube1 (1 site)
            ],
            'oqmd_icsd_id': [10, 11, 12, 13, 14, 15]
        }
        df = pd.DataFrame(csv_data)
        if not os.path.exists("data"):
            os.makedirs("data")
        df.to_csv(self.test_csv_path, index=False)

        # Config file remains largely the same, but train/val/test ratios might need adjustment
        # if more samples are processed, or keep as is to see how splitting handles it.
        # For 4 samples: 0.5 train (2), 0.25 val (1), 0.25 test (1). This is fine.
        config_content = f"""
prepare_gnn_data:
  processed_oqmd_csv_filename: "{self.test_csv_path}"
  processed_graphs_filename: "{self.processed_graphs_path}"
  train_graphs_filename: "{self.train_graphs_path}"
  val_graphs_filename: "{self.val_graphs_path}"
  test_graphs_filename: "{self.test_graphs_path}"
  train_ratio: 0.5
  val_ratio: 0.25
  test_ratio: 0.25
  random_seed: 42
"""
        with open(self.test_config_path, 'w') as f:
            f.write(config_content)

    def tearDown(self):
        if os.path.exists(self.test_csv_path): os.remove(self.test_csv_path)
        if os.path.exists(self.test_config_path): os.remove(self.test_config_path)
        if os.path.exists(self.processed_graphs_path): os.remove(self.processed_graphs_path)
        if os.path.exists(self.train_graphs_path): os.remove(self.train_graphs_path)
        if os.path.exists(self.val_graphs_path): os.remove(self.val_graphs_path)
        if os.path.exists(self.test_graphs_path): os.remove(self.test_graphs_path)

    def test_dataset_creation_and_skipping(self):
        create_graph_dataset(config_path=self.test_config_path)
        self.assertTrue(os.path.exists(self.processed_graphs_path))

        all_graphs = []
        if os.path.exists(self.processed_graphs_path):
            try:
                all_graphs = torch.load(self.processed_graphs_path, weights_only=False)
            except TypeError:
                try:
                    torch.serialization.add_safe_globals([Data])
                    all_graphs = torch.load(self.processed_graphs_path)
                except Exception as e:
                    self.fail(f"torch.load failed: {e}")

        # Expected processed: NaCl1, Si1, MissingTarget1 (Al), FeCube1. (4 materials)
        # InvalidStruct1: Skipped (JSON parse error for sites) - Unknown_ID_3 from script
        # MissingCell1: Skipped (unit_cell_json is None) - Unknown_ID_5 from script
        self.assertEqual(len(all_graphs), 4, f"Expected 4 graphs, found {len(all_graphs)}")

        # Material ID mapping from CSV to script's processing order:
        # NaCl1 -> Unknown_ID_1
        # Si1 -> Unknown_ID_2
        # InvalidStruct1 (skipped) -> Would have been Unknown_ID_3
        # MissingTarget1 -> Unknown_ID_4
        # MissingCell1 (skipped) -> Would have been Unknown_ID_5
        # FeCube1 -> Unknown_ID_6

        processed_material_ids = {g.material_id for g in all_graphs if hasattr(g, 'material_id')}
        self.assertIn('Unknown_ID_1', processed_material_ids) # NaCl1
        self.assertIn('Unknown_ID_2', processed_material_ids) # Si1
        self.assertIn('Unknown_ID_4', processed_material_ids) # MissingTarget1 (Al)
        self.assertIn('Unknown_ID_6', processed_material_ids) # FeCube1


        for g in all_graphs:
            if hasattr(g, 'material_id') and g.material_id == 'Unknown_ID_1': # NaCl1
                expected_y_nacl = torch.tensor([[2.5, -1.5]], dtype=torch.float) # -3.0 / 2 sites
                self.assertTrue(torch.allclose(g.y, expected_y_nacl))
                self.assertEqual(g.x.shape[0], 2)
            elif hasattr(g, 'material_id') and g.material_id == 'Unknown_ID_2': # Si1
                expected_y_si = torch.tensor([[0.5, 0.05]], dtype=torch.float) # 0.1 / 2 sites
                self.assertTrue(torch.allclose(g.y, expected_y_si))
                self.assertEqual(g.x.shape[0], 2)
            elif hasattr(g, 'material_id') and g.material_id == 'Unknown_ID_4': # MissingTarget1 (Al)
                # oqmd_band_gap=np.nan, oqmd_delta_e=-0.5, 1 site
                # Normalized formation energy = -0.5 / 1 = -0.5
                self.assertTrue(torch.isnan(g.y[0,0]))
                self.assertAlmostEqual(g.y[0,1].item(), -0.5, places=5)
                self.assertEqual(g.x.shape[0], 1)
            elif hasattr(g, 'material_id') and g.material_id == 'Unknown_ID_6': # FeCube1
                # oqmd_band_gap=0.1, oqmd_delta_e=-0.2, 1 site
                # Normalized formation energy = -0.2 / 1 = -0.2
                expected_y_fecube = torch.tensor([[0.1, -0.2]], dtype=torch.float)
                self.assertTrue(torch.allclose(g.y, expected_y_fecube))
                self.assertEqual(g.x.shape[0], 1)


        self.assertTrue(os.path.exists(self.train_graphs_path), "Train graph file not created.")
        self.assertTrue(os.path.exists(self.val_graphs_path), "Validation graph file not created.")
        self.assertTrue(os.path.exists(self.test_graphs_path), "Test graph file not created.")

        # With 4 samples, train_ratio=0.5 -> 2 train. temp_data gets 2 samples.
        # val_ratio=0.25, test_ratio=0.25. For temp (2 samples), test_size = 0.25/(0.25+0.25) = 0.5
        # So, temp_data (2 samples) is split into 1 val, 1 test.
        # Splits should be: Train 2, Val 1, Test 1.
        if os.path.exists(self.train_graphs_path):
             train_data = torch.load(self.train_graphs_path, weights_only=False); self.assertEqual(len(train_data), 2)
        if os.path.exists(self.val_graphs_path):
             val_data = torch.load(self.val_graphs_path, weights_only=False); self.assertEqual(len(val_data), 1)
        if os.path.exists(self.test_graphs_path):
             test_data = torch.load(self.test_graphs_path, weights_only=False); self.assertEqual(len(test_data), 1)

if __name__ == '__main__':
    unittest.main()
