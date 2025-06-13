# Material Property Predictor Prototype

## Description

This project is a Tkinter GUI application designed as a prototype for predicting material properties from Crystallographic Information Files (CIF) and for managing a small dataset of materials properties. It allows users to:
*   Select a CIF file and predict properties using pre-trained machine learning models.
*   Manually enter material data and save it to a local CSV dataset.
*   Generate an initial dataset using the Materials Project API (requires an API key).
*   Train machine learning models based on the generated dataset.

## Core Components and Workflow

The project is structured into several key components:
*   **Data Acquisition Scripts (`fetch_mp_data.py`, `process_raw_data.py`):** These scripts handle the creation and preparation of the dataset from external sources like the Materials Project.
*   **Model Training Script (`train_model.py`):** This script is responsible for training machine learning models using the processed dataset.
*   **Graph Dataset Preparation Script (`prepare_gnn_data.py`):** This script processes raw material data (e.g., from CIF strings in the JSON output of `fetch_mp_data.py`) into graph representations suitable for Graph Neural Networks (GNNs). It converts structures to `torch_geometric.data.Data` objects, saves the full processed graph dataset, and splits it into training, validation, and test sets. This is essential for GNN-based model development.
*   **GUI Application (`material_predictor_gui.py`):** Provides the main user interface for interacting with the prediction models and managing data.
*   **Configuration File (`config.yml`):** A central YAML file for managing all important operational settings, file paths, API keys, and model parameters. This improves maintainability by separating settings from code, making it easier for users to adapt the project to their needs or different environments without altering Python scripts.
*   **Utilities (`utils/`):** This directory contains shared Python modules:
    *   `config_loader.py`: Provides a standardized way to load the `config.yml` file, ensuring consistent access to configuration parameters across all scripts and handling potential errors like missing files or malformed YAML.
    *   `schema.py`: Centralizes the definitions of data structures, such as `DATA_SCHEMA` (used in data fetching and processing to define expected fields and their descriptions) and `MANUAL_ENTRY_CSV_HEADERS` (used in the GUI for manual data input to ensure CSV compatibility). This avoids redundancy and ensures consistency in how data is structured and interpreted throughout the project.
*   **Tests (`tests/`):** This directory houses all test files. Unit tests focus on individual modules and functions, using mocking to isolate components (e.g., API calls, file system interactions). Integration tests verify that different parts of the system work together as expected, such as the complete data processing pipeline from data fetching through model training.

The general workflow involves:

1.  **Configuration (New!):**
    *   Modify `config.yml` to set your Materials Project API key, define file paths, and adjust model parameters. (More details in the "Configuration" section below).

2.  **Data Acquisition (Optional, for dataset generation):**
    *   Primarily designed for creating a dataset of Fe-based compounds using the Materials Project API.
    *   **`fetch_mp_data.py`**: This script queries the Materials Project API.
        *   **Requirement**: You **must** provide your Materials Project API key. The primary method is to set the `mp_api_key` in the `config.yml` file. If not found there, the script will check for an environment variable named `MP_API_KEY`. You can obtain a key by registering at [materialsproject.org](https://materialsproject.org).
        *   It fetches raw data for materials (defaulting to Iron-based if not otherwise configured) and saves it to a JSON file (default: `mp_raw_data.json`, configurable in `config.yml`).
    *   **`process_raw_data.py`**: This script processes the raw JSON data file.
        *   It uses `pymatgen` to parse CIF strings and extract structural features.
        *   It combines these with API-sourced data and saves the result to `Fe_materials_dataset.csv`.
    *   **`prepare_gnn_data.py` (For GNN Models):** This script takes the raw data (e.g., `mp_raw_data.json`) and converts it into graph datasets.
        *   It processes materials into `torch_geometric.data.Data` objects.
        *   Saves the full dataset and pre-split train/validation/test sets as `.pt` files.
        *   Configuration for this script (input/output paths, split ratios) is managed in `config.yml` under the `prepare_gnn_data` section.
    *   **Placeholder Dataset**: A placeholder `Fe_materials_dataset.csv` is included in the repository. This allows the GUI and model training script to run for demonstration purposes even if you don't immediately fetch or process fresh data. The filenames for input and output are configurable via `config.yml`.

3.  **Model Training (`train_model.py`):**
    *   This script loads the processed dataset (default: `Fe_materials_dataset.csv`, configurable).
    *   It trains several machine learning models as defined in the script.
    *   Model parameters (e.g., test size, estimators) and output filenames for models and preprocessors are managed via `config.yml`.
    *   **Usage**: `python train_model.py`

4.  **GUI Application (`material_predictor_gui.py`):**
    *   A Tkinter-based graphical user interface with two main tabs.
    *   **"Predict from CIF" Tab:**
        *   Allows users to select a local CIF file.
        *   Extracts structural features using `pymatgen`.
        *   Uses the pre-trained `.joblib` models (loaded on startup) to predict properties: Band Gap, Formation Energy, Metallicity (with confidence score), and DOS at Fermi level (if predicted as metal).
        *   If any required model/preprocessor is not found (e.g., if `train_model.py` hasn't been run), predictions for those specific properties will show as "N/A (model not loaded)".
    *   **"Manual Data Entry" Tab:**
        *   Provides a form to manually input data for all features defined in the project's schema.
        *   **"Load CIF for Feature Extraction" button:** Allows selecting a CIF file to auto-populate `pymatgen`-derived fields (e.g., formula, density, lattice parameters).
        *   **"Save to Dataset" button:** Appends the entered data as a new row to `Fe_materials_dataset.csv`. This allows users to augment the dataset or build one if API access is unavailable.
        *   **"Clear Fields" button:** Resets all entry fields.

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    # Replace <repository_url> with the actual URL of the repository
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Make sure you have Python 3.x installed. The required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    This includes `pymatgen`, `scikit-learn`, `pandas`, `numpy`, `mp-api`, `joblib`, and `PyYAML`.

4.  **Configure `config.yml` (Crucial First Step):**
    *   Open `config.yml` in a text editor.
    *   **Set your `mp_api_key`**. This is essential for `fetch_mp_data.py`.
    *   Review other settings like file paths and model parameters, and adjust if necessary.

5.  **Running the Application & Workflow:**
    *   **Option A: Use placeholder data and pre-trained models (if provided in repo and configured in `config.yml`)**
        1.  Ensure `config.yml` points to existing dataset and model files if you are not training them locally.
        2.  Run the GUI: `python material_predictor_gui.py`
        3.  Use the "Predict from CIF" tab with your own CIF files, or explore the "Manual Data Entry" tab.
    *   **Option B: Generate dataset and train models locally**
        1.  **Ensure API Key is set in `config.yml`**. (Fallback to `MP_API_KEY` environment variable is also possible if `mp_api_key` in `config.yml` is placeholder or missing).
        2.  Run data fetching: `python fetch_mp_data.py` (uses settings from `config.yml`)
        3.  Process raw data: `python process_raw_data.py` (uses settings from `config.yml`)
        3b. Prepare GNN dataset (if using GNN models): `python prepare_gnn_data.py` (uses settings from `config.yml`)
        4.  Train models: `python train_model.py` (uses settings from `config.yml`)
        5.  Run the GUI: `python material_predictor_gui.py` (loads models and datasets as per `config.yml`)

## Configuration (`config.yml`)

Project settings are managed centrally in the `config.yml` file located in the root directory. This file allows you to customize various parameters without modifying the scripts directly, which improves maintainability by separating settings from code, making it easier for users to adapt the project to their needs or different environments without altering Python scripts.

**Key settings include:**
*   `mp_api_key`: **Your Materials Project API key. This is essential for fetching data using `fetch_mp_data.py`.** Ensuring the `mp_api_key` is correctly set in this file is the first and most crucial step for enabling the data fetching capabilities.
*   `fetch_data`: Parameters for `fetch_mp_data.py`, such as `max_total_materials` to fetch, `output_filename` for the raw JSON data, and `criteria_sets` to define the search criteria on Materials Project (e.g., number of elements, specific elements like 'Fe'). A special value of `-5` for `max_total_materials` will instruct the script to attempt to fetch all materials matching the combined criteria from the initial API query, ignoring individual `limit_per_set` and the overall `max_total_materials` cap.
*   `process_data`: Settings for `process_raw_data.py`, including `raw_data_filename` (input) and `output_filename` for the processed CSV dataset.
*   `train_model`: Configuration for `train_model.py`, such as the `dataset_filename` (input CSV), `test_size` for train-test split, `random_state` for reproducibility, `n_estimators` for Random Forest models, and paths for saving trained `models` and `preprocessors`.
*   `prepare_gnn_data`: Settings for `prepare_gnn_data.py`.
    *   `raw_data_filename`: Input JSON file containing raw material data (e.g., `mp_raw_data.json`).
    *   `processed_graphs_filename`: Output path for the file containing the full list of processed `torch_geometric.data.Data` objects (e.g., `data/processed/processed_graphs.pt`).
    *   `train_graphs_filename`, `val_graphs_filename`, `test_graphs_filename`: Output paths for the split datasets (training, validation, and test graph objects).
    *   `random_seed`: Integer seed for reproducible dataset splitting.
    *   `train_ratio`, `val_ratio`, `test_ratio`: Floating point values for dataset split proportions (e.g., 0.7, 0.2, 0.1).
*   `gui`: Settings for `material_predictor_gui.py`, like the application `title`, window `geometry`, paths to `models_to_load`, and the `manual_entry_csv_filename` for saving manually entered data.

**Important:** Before running `fetch_mp_data.py` for the first time, you **must** update the `mp_api_key` field in `config.yml` with your personal Materials Project API key. If this key is not found or is set to the placeholder `"YOUR_MP_API_KEY"` in `config.yml`, the system will then check for the `MP_API_KEY` environment variable as a fallback.

## Running Tests

The project includes a suite of unit and integration tests located in the `tests/` directory. These tests are built using the `pytest` framework.

To run all tests, navigate to the root directory of the project in your terminal and execute:
```bash
pytest
```
This will discover and run all test files (e.g., `test_*.py`).
*   **Unit tests** verify the functionality of individual modules (e.g., configuration loading from `utils/config_loader.py`, schema definitions in `utils/schema.py`). They also test the core logic within each script, such as data transformation rules in `process_raw_data.py`, correct parameter usage in `train_model.py` based on the configuration, and the data fetching workflow in `fetch_mp_data.py` (simulating various API responses using mocks).
*   **Integration tests** check if different parts of the system work together correctly, specifically the main data pipeline (`fetch_data` -> `process_data` -> `train_models`) ensuring that these components correctly pass data (via files, as configured) from one stage to the next.
*   **Note on GUI Testing**: GUI functionality related to model loading and predictions is tested at the code level (e.g., ensuring models are loaded as per config by the GUI script's logic), but automated GUI interaction tests (e.g., simulating button clicks) are not currently implemented due to environment-specific `tkinter` challenges encountered during development.

## Error Handling & Model Availability
*   The GUI will show warnings if model files (`.joblib`, paths configured in `config.yml`) are not found during startup, and corresponding predictions will be disabled.
*   The data fetching script (`fetch_mp_data.py`) will warn if the API key is not properly configured (see Configuration section) and may fail or retrieve limited data.
*   Basic error messages are shown for CIF parsing issues or missing dataset files.

## Data Search Criteria

The data acquisition strategy relies on specific roles for each data source:

*   **SuperCon**: This dataset is the primary source for the target variable, which is the critical temperature (Tc) of superconducting materials.
*   **OQMD (Open Quantum Materials Database)**: OQMD is used to obtain complementary material properties (e.g., formation energy, band gap, crystal structure) for compositions identified in the SuperCon dataset. It also serves as a broader database for sourcing material properties and crystal structures for general analysis and model training.
*   **Materials Project (MP)**: The Materials Project API is an *optional* source for acquiring complementary material properties and crystal structures. It can be used similarly to OQMD to enrich the dataset or as an alternative source for such information.

## OracleNet GNN Model

This project includes OracleNet, a Graph Neural Network (GNN) model designed for predicting material properties. The GNN takes material structures represented as graphs (nodes being atoms, edges being bonds/connections) and learns to predict target properties.

### Data Preparation for GNN

Before training the GNN, graph data needs to be prepared:
- Raw material data (e.g., from Materials Project or OQMD containing CIF strings and target properties) is processed into graph structures.
- The script `scripts/prepare_gnn_data.py` handles this conversion. It takes raw data (e.g., `mp_raw_data.json`), converts structures to graphs using `pymatgen` and `torch_geometric`, and saves datasets (`train_graphs.pt`, `val_graphs.pt`, `test_graphs.pt`) in the `data/` directory.
  - Node features typically include atomic number and electronegativity.
  - Edge features may include interatomic distances.
  - Target properties (e.g., band gap, formation energy) are stored with each graph.

### Training the GNN Model

The OracleNet GNN model can be trained using the following script:

```bash
python scripts/train_gnn_model.py
```

- This script loads the preprocessed graph data from `data/train_graphs.pt` and `data/val_graphs.pt`.
- It instantiates the `OracleNetGNN` model (a GCN-based architecture defined in `models/gnn_oracle_net.py`).
- Training involves optimizing the model to predict a target property (e.g., band gap, specified by `gnn_target_index` in the configuration).
- The script uses an Adam optimizer and Mean Squared Error (MSE) loss.
- The model with the best validation loss is saved to `data/oracle_net_gnn.pth` (or as configured).
- Key training hyperparameters (learning rate, batch size, epochs, hidden channels, target index) can be configured in `config.yml` under the `gnn_settings` section.

### Evaluating the GNN Model

To evaluate the performance of the trained GNN model on the test set:

```bash
python scripts/evaluate_gnn_model.py
```

- This script loads the trained model from `data/oracle_net_gnn.pth` and the test data from `data/test_graphs.pt`.
- It calculates and reports metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- The GNN's performance is also compared against a random baseline predictor.
- Basic error analysis is performed to show the top N predictions with the highest errors, helping to identify areas where the model struggles.
- Configuration for evaluation (e.g., paths, `gnn_target_index`) is also managed via `config.yml` under `gnn_settings`.

### GNN Configuration

All settings related to the GNN model, training, evaluation, and dummy data generation (used if actual data files are missing) are located in `config.yml` under the `gnn_settings:` key. This includes file paths, learning parameters, model architecture details (like hidden channels), and target property selection.

## Technology Stack

This project leverages the following core technologies:

*   **Language**: Python 3.10+
*   **Machine Learning**:
    *   PyTorch (`torch`): Core deep learning framework.
    *   PyTorch Geometric (PyG) (`torch_geometric`): Library for deep learning on graphs and other irregular structures.
*   **Chemistry/Materials Science**:
    *   RDKit (`rdkit-pypi`): Toolkit for cheminformatics.
    *   Pymatgen (`pymatgen`): Python Materials Genomics library for materials analysis, including CIF and structure manipulation.
*   **Experiment Management (Planned)**:
    *   Weights & Biases (W&B) or MLflow: For tracking experiments, models, and datasets. (Not yet integrated)
*   **Data Version Control (Planned)**:
    *   DVC (Data Version Control): For managing large data files and ML models alongside Git. (Not yet integrated)
