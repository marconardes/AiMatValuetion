# AI Superconductor Discovery Roadmap: A 20-Step Plan

## Core Technology Stack

This project utilizes the following core technologies:

*   **Language**: Python 3.10+
*   **Machine Learning**:
    *   PyTorch (`torch`): Core deep learning framework. (Installed)
    *   PyTorch Geometric (PyG) (`torch_geometric`): Library for deep learning on graphs. (Installed)
*   **Chemistry/Materials Science**:
    *   RDKit (`rdkit-pypi`): Toolkit for cheminformatics. (Installed)
    *   Pymatgen (`pymatgen`): Library for materials analysis. (Installed)
*   **Experiment Management (Planned)**:
    *   Weights & Biases (W&B) or MLflow. (To be integrated)
*   **Data Version Control (Planned)**:
    *   DVC (Data Version Control). (To be integrated)

**Legend:**
- `[x]` Implemented
- `[~]` Partially Implemented or Basic Version Exists
- `[ ]` Not Implemented

---

This plan transforms strategic phases into an actionable workflow, with clear priorities from initial setup to experimental validation.

## Phase I: 🏗️ Building the Data Foundation (Priorities 1-6)

The project's success hinges entirely on the quality and organization of your data.

- `[~]` **(Priority 1/20) Development Environment Setup:**
    - `[x]` Set up a code repository (Git). (B)
    - `[~]` Install essential AI libraries (PyTorch, TensorFlow). (B)
    - `[~]` Install AI libraries for graphs (PyTorch Geometric or DGL). (B)
    - `[x]` Install chemistry/materials libraries (Pymatgen, RDKit). (B)

- `[~]` **(Priority 2/20) Identification and Access to Data Sources:**
    - `[~]` Access Materials Project (API, optional), SuperCon (local files), OQMD (API identified); ICSD pending. (I)
    - `[x]` Define search criteria (SuperCon for Tc, OQMD/MP for properties). (B)

- `[~]` **(Priority 3/20) Development of Data Extraction Scripts:**
    - `[x]` Scripts for Materials Project data extraction exist. (I)
    - `[x]` Develop script to process local SuperCon dataset (`raw.tsv`) for compositions and Tc. (I)
    - `[x]` Develop script to fetch complementary data from OQMD API for SuperCon compositions. (I)
    - `[x]` Process fetched OQMD data (`oqmd_data_raw.json`) to select/filter entries and extract features. (I)
    - `[x]` Store raw data in an organized format (e.g., local database or data lake). (B)

- `[x]` **(Priority 4/20) Data Cleaning and Normalization:** (I)
    - `[x]` Validate extracted data, handling missing values and inconsistencies. (I)
    - `[x]` Unify units and formats. For example, ensure all crystal structures are in a standard format like CIF files. (I)

- `[x]` **(Priority 5/20) Definition and Implementation of Graph Representation:** (A)
    - `[x]` Formally define how a crystal structure will be converted into a graph. (A)
        - `[x]` Nodes: Atoms (with features like atomic number, electronegativity). (I)
        - `[x]` Edges: Bonds or neighborhood (with features like distance). (I)
    - `[x]` Implement the Structure -> Graph conversion function. (A)

- `[x]` **(Priority 6/20) Dataset Preprocessing and Splitting:** (I)
    - `[x]` Process all cleaned data, converting them into graph objects. (A)
    - `[x]` Save this processed dataset for quick access. (B)
    - `[x]` Split the dataset into Training (70%), Validation (20%), and Test (10%) sets. (B)

## Phase II: 🤖 Development of the "OracleNet" Predictive Model (Priorities 7-10)

With the data ready, we build the tool that will guide our generator.

- `[ ]` **(Priority 7/20) Design and Implementation of the Predictive GNN Architecture:** (A)
    - `[ ]` Choose and implement a GNN architecture (e.g., SchNet, GAT, MEGNet) for OracleNet. (A)
    - `[ ]` The model should accept a graph as input and produce a numerical value (Tc) as output. (A)

- `[~]` **(Priority 8/20) Training the Predictive Model:** (A)
    - `[ ]` Write the training loop for OracleNet. (A)
    - `[~]` Train the model on the training set, using the validation set to tune hyperparameters (learning rate, layer size, etc.). (I)

- `[~]` **(Priority 9/20) Rigorous Evaluation of OracleNet:** (I)
    - `[~]` Measure the performance of the trained model on the test set (which the model has never seen). (I)
    - `[ ]` Important metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE). (B)
    - `[ ]` Critical checkpoint: OracleNet must have significantly better predictive power than a random baseline. If not, go back to Phase I or improve the architecture. (I)

- `[ ]` **(Priority 10/20) Error Analysis and Interpretability:** (I)
    - `[ ]` Analyze where OracleNet makes the most mistakes. Does it struggle with any specific family of materials? (I)
    - `[ ]` Use explainability techniques (XAI for GNNs) to understand which substructures the model considers important for superconductivity. (A)

## Phase III (Revisada): ✨ Development of the "Creator" Generative Model with VAE + LNN (Priorities 11-16)

Objective: Create a system that generates chemically valid and physically stable materials (using LNN), optimized for high Tc (using VAE and OracleNet).

- `[ ]` **(Priority 11/20) Design of the Hybrid Architecture (VAE + LNN):**
    - `[ ]` Design the main generative architecture (VAE-based):
        - `[ ]` Encoder (GNN): Compresses a material graph into a latent space vector.
        - `[ ]` Decoder (GNN): Generates a new material graph from a latent space vector.
    - `[ ]` Design the physical validation network:
        - `[ ]` Lagrangian Neural Network (LNN): A network trained to learn an approximation of the potential energy of an atomic configuration. It will receive a generated graph and evaluate its energy stability.

- `[ ]` **(Priority 12/20) Implementation of the Advanced Composite Loss Function:**
    - `[ ]` This is the logic that connects generation, property optimization, and physical realism. The VAE's loss will be a weighted sum of:
        - `[ ]` Reconstruction Loss: How well the VAE reconstructs input data.
        - `[ ]` KL Divergence Loss: Standard VAE latent space regularization.
        - `[ ]` Predictive Loss (OracleNet): Encourages generation of materials with high Tc predicted by OracleNet.
        - `[ ]` Physical Stability Loss (LNN): Penalizes the generator for creating structures that the LNN classifies as high energy (unstable or physically implausible). This is the crucial link to the LNN.

- `[ ]` **(Priority 13/20) Implementation of the Hybrid Training Loop:**
    - `[ ]` Write the script that trains the VAE system. LNN training can be done separately with simulation data (e.g., DFT) or jointly.
    - `[ ]` In the main VAE training loop:
        - `[ ]` Generate a "fake" graph with the Decoder.
        - `[ ]` Pass the graph through OracleNet to get the predictive loss.
        - `[ ]` Pass the same graph through the pre-trained LNN to get the stability loss.
        - `[ ]` Calculate the composite loss and update VAE weights.

- `[ ]` **(Priority 14/20) Training the Models:**
    - `[ ]` 1. Train the LNN: Train the network to predict the energy of atomic configurations from a database of known materials and their calculated energies.
    - `[ ]` 2. Train the VAE System: Run VAE training using the composite loss function, which now includes feedback from the already trained LNN. Monitor all loss components.

- `[ ]` **(Priority 15/20) Generation of the Batch of Physically Valid Candidates:**
    - `[ ]` Use the trained VAE Decoder to generate thousands of new structures.
    - `[ ]` By design, these structures have already been optimized during training to be candidates for high Tc and physical stability.

- `[ ]` **(Priority 16/20) Advanced Filtering and Ranking of Candidates:**
    - `[ ]` Create a final, more robust evaluation pipeline:
        - `[ ]` Final chemical validity check.
        - `[ ]` Re-run LNN to get an accurate energy stability score for each finalist candidate.
        - `[ ]` Run OracleNet to predict Tc for each.
        - `[ ]` Rank candidates using a combined criterion: highest predicted Tc AND lowest energy (highest stability).

## Phase IV: 🧪 Validation and Closing the Loop (Priorities 17-20)

Where AI meets the real world.

- `[ ]` **(Priority 17/20) Screening with Classical Computational Simulations:** (A)
    - `[ ]` Take the top ~100 from the ranked list. (B)
    - `[ ]` Perform more accurate, but slower, simulations (like DFT) to verify the stability and electronic properties of these candidates. (A)

- `[ ]` **(Priority 18/20) Selection of Final Candidates for Synthesis:** (I)
    - `[ ]` Based on AI results and computational screening, select a small number (1 to 5) of "champion" candidates for experimental validation. (I)

- `[ ]` **(Priority 19/20) Collaboration for Synthesis and Laboratory Testing:** (B)
    - `[ ]` This step requires collaboration with a materials physics or chemistry lab. (B)
    - `[ ]` Partners will attempt to synthesize the proposed materials and measure their actual properties, including Tc. (B)

- `[ ]` **(Priority 20/20) Closing the "Active Learning" Loop:** (I)
    - `[ ]` The most important step for long-term success. (B)
    - `[ ]` Take the experimental results (whether success or failure) from Phase 19. (B)
    - `[ ]` Add these new data points to your original database. (B)
    - `[ ]` Re-train OracleNet and, optionally, the GAN system with this new data. (A)
    - `[ ]` Repeat the cycle from Phase III/IV. (B)
