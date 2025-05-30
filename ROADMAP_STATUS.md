# AI Model Development Roadmap: Current Status

This document tracks the project's progress against the initial roadmap for building an AI model to predict material properties.

- `[x]` Implemented
- `[~]` Partially Implemented or Basic Version Exists
- `[ ]` Not Implemented

---

## 1. Define the Problem and Scope:

- `[x]` **What specific electronic property do you want to predict?** (Implemented: Band gap, density of states features, formation energy. Stability not directly, though formation energy is related.)
- `[~]` **For which class of materials?** (Annotation: Initial dataset focused on Fe-based compounds from Materials Project. The GUI tool itself is generic for any CIF.)
- `[ ]` **What is the desired level of accuracy and acceptable computational cost?** (Annotation: Accuracy is evaluated, but no specific target like "80%" was set or optimized for.)

## 2. Data Acquisition and Preparation:

- `[x]` **Data Source: Public Databases** (Annotation: Implemented fetching from Materials Project API via `fetch_mp_data.py`. AFLOW was discussed but not implemented as a direct source.)
- `[~]` **Data Source: Your Own Calculations** (Annotation: Direct DFT calculations by the agent are not feasible. However, the "Manual Data Entry" tab in the GUI allows users to input data from their own calculations.)
- `[~]` **Cleaning and Validation:** (Annotation: Basic NaN handling is done in the training script. No comprehensive data cleaning or validation methods are implemented yet.)
- `[~]` **Data Volume:** (Annotation: The framework fetches ~50 materials for demonstration. Scripts can be adapted for more, and manual entry is possible. The current dataset is not "large volume" in the context of deep learning.)

## 3. Feature Engineering (Material Descriptors):

- `[x]` **This is one of the most crucial steps. You need to convert the material information (chemical composition, crystal structure) into a numerical format that the AI ​​model can understand.** (Annotation: Implemented via `process_raw_data.py` using `pymatgen`.)
- `[x]` **Feature Types: Composition-Based** (Annotation: Includes reduced formula, number of elements, list of elements.)
- `[x]` **Feature Types: Structure-Based** (Annotation: Includes density, cell volume, volume per atom, space group number, crystal system, lattice parameters, number of sites.)
- `[x]` **Tools: Libraries such as pymatgen (integrated with the Materials Project) and Matminer are extremely useful for generating a wide variety of material descriptors.** (Annotation: `pymatgen` is used extensively. `Matminer` is not used.)

## 4. AI Model Selection and Training:

- `[x]` **Data Splitting: Separate your data into training, validation, and testing sets.** (Annotation: `train_test_split` from `scikit-learn` is used.)
- `[~]` **Algorithm Selection: Neural Networks (Deep Learning)** (Annotation: Classical Machine Learning (Random Forest) from `scikit-learn` is implemented. Neural Networks or Graph Neural Networks are not implemented.)
- `[~]` **Training:**
    - `[~]` Choose a loss function (e.g., Mean Squared Error for regression). (Annotation: Implicit for Random Forest.)
    - `[ ]` Choose an optimizer (e.g., Adam). (Annotation: Not applicable for Random Forest as implemented.)
    - `[ ]` Tune the model's hyperparameters (e.g., learning rate, number of layers/neurons in neural networks, tree depth in Random Forest). (Annotation: Default hyperparameters used for Random Forest; no tuning implemented.)

## 5. Model Evaluation:

- `[x]` **Metrics: For regression problems (such as predicting a band gap), use metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² (coefficient of determination).** (Annotation: MAE and R² are implemented for regression models. RMSE is not, but similar information is conveyed.)
- `[ ]` **Cross-Validation: An important technique to obtain a more robust estimate of the model's performance.** (Annotation: Not implemented.)
- `[~]` **Error Analysis: Understand where your model is failing. Does it have difficulty with certain types of materials or value ranges?** (Annotation: Basic evaluation metrics are printed. No detailed error analysis tools or reports are generated.)

## 6. Iteration and Refinement:

- `[~]` **Based on the evaluation, you may need to go back to previous steps:** (Annotation: The project provides scripts and a GUI that allow for data augmentation and retraining, supporting an iterative process. However, no automated iteration or refinement loops have been implemented or executed.)
    - `[~]` Collect more data or better quality data. (Annotation: Possible via API script modification or manual entry.)
    - `[ ]` Design new features. (Annotation: Current feature set is fixed for now.)
    *   `[ ]` Try different model architectures. (Annotation: Only Random Forest implemented.)
    *   `[ ]` Tune the hyperparameters better. (Annotation: Not implemented.)

## Common Tools and Languages:

- `[x]` **Python:** The dominant language for machine learning.
- `[x]` **Essential Python Libraries:**
    - `[x]` **scikit-learn:** For classical machine learning.
    - `[ ]` **TensorFlow or PyTorch:** For deep learning.
    - `[x]` **pymatgen:** For manipulating crystal structures and material data.
    - `[ ]` **Matminer:** For material featurization.
    - `[x]` **Pandas:** For manipulating tabular data.
    *   `[x]` **NumPy:** For numerical computation (via pandas/sklearn).
    *   `[ ]` **Matplotlib / Seaborn:** For data visualization. (Annotation: No specific data visualization features implemented in this project.)

## Adds a graphical interface for the creation of new materials.

- `[x]` **(Referring to GUI for inputting material candidates for prediction & manual data entry)** (Annotation: The Tkinter GUI includes a "Predict from CIF" tab and a "Manual Data Entry" tab, fulfilling this.)

---
## Project Enhancements

- `[x]` **GUI Code Refactoring:** The `material_predictor_gui.py` file was significantly refactored. Tab-specific functionalities ('Predict from CIF', 'Manual Data Entry') were moved into their own classes (`PredictionTab`, `ManualEntryTab`) for improved modularity, readability, and maintainability.

---
## Future Considerations & Potential Next Steps (Dicas)

Here are some potential areas for future development and improvement:

*   **Advanced Model Training:**
    *   Implement hyperparameter tuning (e.g., using `GridSearchCV` or `RandomizedSearchCV` from `scikit-learn`) for the existing Random Forest models to potentially improve their performance.
    *   Incorporate cross-validation during the model training process (`train_model.py`) for more robust evaluation metrics.
*   **Explore Advanced Models:**
    *   If larger and more diverse datasets become available, explore more advanced model architectures such as:
        *   Feedforward Neural Networks (FNNs) for tabular feature data.
        *   Graph Neural Networks (GNNs), like CGCNN, which can learn directly from crystal structures (would require significant changes to feature engineering and data representation).
*   **Deeper Evaluation & Analysis:**
    *   Develop more detailed error analysis tools or outputs. For instance, identify types of materials or feature ranges where models perform poorly.
    *   Implement functionality to plot feature distributions, target variable distributions, or prediction correlations (e.g., predicted vs. actual plots). This might involve integrating libraries like Matplotlib/Seaborn, potentially as a separate script or new GUI tab.
*   **Data Management & Scalability:**
    *   For larger datasets, consider moving away from CSVs to more robust storage solutions (e.g., SQLite database, Parquet files).
    *   If using very large datasets or complex models, explore tools for experiment tracking (e.g., MLflow, Weights & Biases).
*   **Code & Project Structure:**
    *   `[x]` Introduce a configuration file (e.g., YAML or JSON) to manage settings like model paths, file paths, or default parameters, instead of having them hardcoded in scripts.
    *   `[x]` Develop a suite of unit and integration tests to ensure code reliability and catch regressions as the project evolves.
    *   `[x]` Further modularize the code, for example, by moving utility functions or data schema definitions into separate modules.
*   **User Interface Enhancements:**
    *   Allow selection of different trained models if multiple versions or types are available.
    *   Provide more interactive feedback or visualizations within the GUI.
