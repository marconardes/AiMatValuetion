# AI Superconductor Discovery Roadmap: A 20-Step Plan

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

- `[~]` **(Priority 6/20) Dataset Preprocessing and Splitting:** (I)
    - `[ ]` Process all cleaned data, converting them into graph objects. (A)
    - `[ ]` Save this processed dataset for quick access. (B)
    - `[~]` Split the dataset into Training (70%), Validation (15%), and Test (15%) sets. (B)

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

## Phase III: ✨ Development of the "Creator" Generative Model (Priorities 11-16)

Now, the most innovative part: creating new materials.

- `[ ]` **(Priority 11/20) Design of the GAN Architecture for Graphs:** (A)
    - `[ ]` Design the two main networks: (A)
        - `[ ]` Generator: A GNN that takes noise and generates a new material graph. (A)
        - `[ ]` Discriminator: A GNN that takes a graph and classifies it as real or fake. (A)

- `[ ]` **(Priority 12/20) Implementation of the Composite Loss Function:** (A)
    - `[ ]` This is the core logic. The Generator's loss function will be a weighted sum of: (A)
        - `[ ]` Adversarial Loss: How well it fools the Discriminator. (A)
        - `[ ]` Predictive Loss: How high the Tc predicted by OracleNet is for the generated material (the goal is to maximize this). (A)
        - `[ ]` (Optional) Regularization terms to ensure chemical validity. (I)

- `[ ]` **(Priority 13/20) Implementation of the GAN Training Loop:** (A)
    - `[ ]` Write the script that alternates between training the Discriminator (with real and fake data) and the Generator (using the composite loss). This cycle is more complex than in Phase II. (A)

- `[ ]` **(Priority 14/20) Training the Complete GAN System:** (A)
    - `[ ]` Run the GAN training. This step is computationally intensive and may require powerful GPUs. (A)
    - `[ ]` Monitor the Generator and Discriminator losses to ensure training is stable. (I)

- `[ ]` **(Priority 15/20) Generation of the Initial Batch of Candidates:** (I)
    - `[ ]` Use the trained Generator to create a large number (thousands) of new molecular structures that do not exist in the database. (I)

- `[ ]` **(Priority 16/20) Filtering and Ranking of Generated Candidates:** (I)
    - `[ ]` Create a pipeline to evaluate the generated candidates: (I)
        - `[ ]` Check basic chemical validity. (B)
        - `[ ]` Run OracleNet to predict the Tc of each one. (I)
        - `[ ]` Rank candidates from highest Tc to lowest. (B)

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
