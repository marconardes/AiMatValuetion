# Material Property Predictor Prototype

## Description

This project is a Tkinter GUI application designed as a prototype for predicting material properties from Crystallographic Information Files (CIF). It allows users to select a CIF file, view some basic extracted material data, and see placeholder predictions for key electronic properties.

## Current Features

*   **CIF File Selection:** Users can browse and select a `.cif` file from their local system.
*   **Material Data Extraction:** Utilizes the `pymatgen` library to parse the selected CIF file and extracts:
    *   Chemical Formula (reduced)
    *   Density
    *   Cell Volume
*   **Placeholder Predictions:** Displays placeholder values for:
    *   Band Gap
    *   Density of States (DOS)
    *   Formation Energy
    (Note: These predictions are currently based on a hardcoded lookup using the material's chemical formula and are for demonstration purposes only.)
*   **Error Handling:** Basic error messages are shown if a CIF file cannot be parsed.

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
    Make sure you have Python 3.x installed.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python material_predictor_gui.py
    ```
    This will launch the Tkinter GUI. You can then select a CIF file and click "Predict". Example CIF files are not provided in the repository, so you will need to use your own.
