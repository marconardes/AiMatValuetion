# Material Property Predictor Prototype

## Description

This project is a Tkinter GUI application designed as a prototype for predicting material properties from Crystallographic Information Files (CIF) and for managing a small dataset of materials properties. It allows users to:
*   Select a CIF file and predict properties using pre-trained machine learning models.
*   Manually enter material data and save it to a local CSV dataset.
*   Generate an initial dataset using the Materials Project API (requires an API key).
*   Train machine learning models based on the generated dataset.

## Core Components and Workflow

The project is structured into several key components:
*   **Data Acquisition Scripts (`fetch_mp_data.py`, `process_raw_data.py`):** For creating and preparing the dataset.
*   **Model Training Script (`train_model.py`):** For training ML models from the dataset.
*   **GUI Application (`material_predictor_gui.py`):** The main user interface.
*   **Configuration File (`config.yml`):** For managing all important settings and parameters.
*   **Utilities (`utils/`):** Contains shared modules for configuration loading (`config_loader.py`) and data schemas (`schema.py`).
*   **Tests (`tests/`):** Includes unit and integration tests to ensure code reliability.

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
        4.  Train models: `python train_model.py` (uses settings from `config.yml`)
        5.  Run the GUI: `python material_predictor_gui.py` (loads models and datasets as per `config.yml`)

## Configuration (`config.yml`)

Project settings are managed centrally in the `config.yml` file located in the root directory. This file allows you to customize various parameters without modifying the scripts directly.

**Key settings include:**
*   `mp_api_key`: **Your Materials Project API key. This is essential for fetching data using `fetch_mp_data.py`.**
*   File paths: Locations for raw data, processed datasets, and saved models (e.g., `fetch_data.output_filename`, `process_data.output_filename`, `train_model.dataset_filename`, paths in `train_model.models` and `train_model.preprocessors`).
*   Model parameters: Settings for training machine learning models, such as `train_model.test_size` and `train_model.n_estimators`.
*   GUI settings: Window title and dimensions under the `gui` section.

**Important:** Before running `fetch_mp_data.py` for the first time, you **must** update the `mp_api_key` field in `config.yml` with your personal Materials Project API key. If this key is not found or is set to the placeholder `"YOUR_MP_API_KEY"` in `config.yml`, the system will then check for the `MP_API_KEY` environment variable as a fallback.

## Running Tests

The project includes a suite of unit and integration tests located in the `tests/` directory. These tests are built using the `pytest` framework.

To run all tests, navigate to the root directory of the project in your terminal and execute:
```bash
pytest
```
This will discover and run all test files (e.g., `test_*.py`).
*   **Unit tests** verify the functionality of individual modules (e.g., configuration loading, data processing logic).
*   **Integration tests** check if different parts of the system work together correctly (e.g., the data processing pipeline from fetching to model training).

## Error Handling & Model Availability
*   The GUI will show warnings if model files (`.joblib`, paths configured in `config.yml`) are not found during startup, and corresponding predictions will be disabled.
*   The data fetching script (`fetch_mp_data.py`) will warn if the API key is not properly configured (see Configuration section) and may fail or retrieve limited data.
*   Basic error messages are shown for CIF parsing issues or missing dataset files.

[end of README.md]
