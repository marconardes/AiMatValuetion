# Material Property Predictor Prototype

## Description

This project is a Tkinter GUI application designed as a prototype for predicting material properties from Crystallographic Information Files (CIF) and for managing a small dataset of materials properties. It allows users to:
*   Select a CIF file and predict properties using pre-trained machine learning models.
*   Manually enter material data and save it to a local CSV dataset.
*   Generate an initial dataset using the Materials Project API (requires an API key).
*   Train machine learning models based on the generated dataset.

## Core Components and Workflow

1.  **Data Acquisition (Optional, for dataset generation):**
    *   Primarily designed for creating a dataset of Fe-based compounds using the Materials Project API.
    *   **`fetch_mp_data.py`**: This script queries the Materials Project API.
        *   **Requirement**: You **must** set an environment variable named `MP_API_KEY` with your valid Materials Project API key. You can obtain a key by registering at [materialsproject.org](https://materialsproject.org).
        *   It fetches raw data for materials containing Iron (Fe) and saves it to `mp_raw_data.json`.
    *   **`process_raw_data.py`**: This script processes `mp_raw_data.json`.
        *   It uses `pymatgen` to parse CIF strings and extract structural features.
        *   It combines these with API-sourced data and saves the result to `Fe_materials_dataset.csv`.
    *   **Placeholder Dataset**: A placeholder `Fe_materials_dataset.csv` is included in the repository. This allows the GUI and model training script to run for demonstration purposes even if you don't immediately fetch fresh data from the API.

2.  **Model Training (`train_model.py`):**
    *   This script loads the `Fe_materials_dataset.csv`.
    *   It trains several machine learning models to predict material properties:
        *   Band Gap (Regressor)
        *   Formation Energy per Atom (Regressor)
        *   Metallicity (Is Metal - Classifier)
        *   Density of States (DOS) at Fermi Level (Regressor, for metals only)
    *   The script performs basic preprocessing (imputation, scaling, one-hot encoding) and saves the trained models and preprocessors as `.joblib` files:
        *   `model_target_band_gap.joblib`
        *   `model_target_formation_energy.joblib`
        *   `model_target_is_metal.joblib`
        *   `model_dos_at_fermi.joblib`
        *   `preprocessor_main.joblib` (for general features)
        *   `preprocessor_dos_at_fermi.joblib` (specifically for DOS model features)
    *   **Usage**: `python train_model.py`

3.  **GUI Application (`material_predictor_gui.py`):**
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
    This includes `pymatgen`, `scikit-learn`, `pandas`, `numpy`, `mp-api`, and `joblib`.

4.  **Running the Application & Workflow:**
    *   **Option A: Use placeholder data and pre-trained models (if provided)**
        1.  The repository may include placeholder `Fe_materials_dataset.csv` and pre-trained `.joblib` files.
        2.  Run the GUI: `python material_predictor_gui.py`
        3.  Use the "Predict from CIF" tab with your own CIF files, or explore the "Manual Data Entry" tab.
    *   **Option B: Generate dataset and train models locally**
        1.  **Set API Key (Crucial for `fetch_mp_data.py`):**
            Set the `MP_API_KEY` environment variable:
            ```bash
            # Linux/macOS
            export MP_API_KEY="YOUR_ACTUAL_API_KEY"
            # Windows Command Prompt
            set MP_API_KEY="YOUR_ACTUAL_API_KEY"
            # Windows PowerShell
            $Env:MP_API_KEY="YOUR_ACTUAL_API_KEY"
            ```
        2.  Run data fetching: `python fetch_mp_data.py`
        3.  Process raw data: `python process_raw_data.py` (This creates/updates `Fe_materials_dataset.csv`)
        4.  Train models: `python train_model.py` (This creates the `.joblib` model files)
        5.  Run the GUI: `python material_predictor_gui.py`

## Error Handling & Model Availability
*   The GUI will show warnings if model files (`.joblib`) are not found during startup, and corresponding predictions will be disabled.
*   The data fetching script (`fetch_mp_data.py`) will warn if the `MP_API_KEY` is not set and may fail or retrieve limited data.
*   Basic error messages are shown for CIF parsing issues or missing dataset files.

[end of README.md]
