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
    - `[x]` Set up a code repository (Git).
    - `[~]` Install essential AI libraries (PyTorch, TensorFlow).
    - `[~]` Install AI libraries for graphs (PyTorch Geometric or DGL).
    - `[x]` Install chemistry/materials libraries (Pymatgen, RDKit).

- `[~]` **(Priority 2/20) Identification and Access to Data Sources:**
    - `[~]` Access Materials Project (API), SuperCon (local files), OQMD (API identified); ICSD pending.
    - `[~]` Define search criteria (SuperCon for Tc, OQMD/MP for properties).

- `[~]` **(Priority 3/20) Development of Data Extraction Scripts:**
    - `[x]` Scripts for Materials Project data extraction exist.
    - `[x]` Develop script to process local SuperCon dataset (`raw.tsv`) for compositions and Tc.
    - `[ ]` Develop script to fetch complementary data from OQMD API for SuperCon compositions.
    - `[x]` Store raw data in an organized format (e.g., local database or data lake).

- `[~]` **(Priority 4/20) Data Cleaning and Normalization:**
    - `[~]` Validate extracted data, handling missing values and inconsistencies.
    - `[~]` Unify units and formats. For example, ensure all crystal structures are in a standard format like CIF files.

- `[ ]` **(Priority 5/20) Definition and Implementation of Graph Representation:**
    - `[ ]` Formally define how a crystal structure will be converted into a graph.
        - `[ ]` Nodes: Atoms (with features like atomic number, electronegativity).
        - `[ ]` Edges: Bonds or neighborhood (with features like distance).
    - `[ ]` Implement the Structure -> Graph conversion function.

- `[~]` **(Priority 6/20) Dataset Preprocessing and Splitting:**
    - `[ ]` Process all cleaned data, converting them into graph objects.
    - `[ ]` Save this processed dataset for quick access.
    - `[~]` Split the dataset into Training (70%), Validation (15%), and Test (15%) sets.

## Phase II: 🤖 Development of the "OracleNet" Predictive Model (Priorities 7-10)

With the data ready, we build the tool that will guide our generator.

- `[ ]` **(Priority 7/20) Design and Implementation of the Predictive GNN Architecture:**
    - `[ ]` Choose and implement a GNN architecture (e.g., SchNet, GAT, MEGNet) for OracleNet.
    - `[ ]` The model should accept a graph as input and produce a numerical value (Tc) as output.

- `[~]` **(Priority 8/20) Training the Predictive Model:**
    - `[ ]` Write the training loop for OracleNet.
    - `[~]` Train the model on the training set, using the validation set to tune hyperparameters (learning rate, layer size, etc.).

- `[~]` **(Priority 9/20) Rigorous Evaluation of OracleNet:**
    - `[~]` Measure the performance of the trained model on the test set (which the model has never seen).
    - `[ ]` Important metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
    - `[ ]` Critical checkpoint: OracleNet must have significantly better predictive power than a random baseline. If not, go back to Phase I or improve the architecture.

- `[ ]` **(Priority 10/20) Error Analysis and Interpretability:**
    - `[ ]` Analyze where OracleNet makes the most mistakes. Does it struggle with any specific family of materials?
    - `[ ]` Use explainability techniques (XAI for GNNs) to understand which substructures the model considers important for superconductivity.

## Phase III: ✨ Development of the "Creator" Generative Model (Priorities 11-16)

Now, the most innovative part: creating new materials.

- `[ ]` **(Priority 11/20) Design of the GAN Architecture for Graphs:**
    - `[ ]` Design the two main networks:
        - `[ ]` Generator: A GNN that takes noise and generates a new material graph.
        - `[ ]` Discriminator: A GNN that takes a graph and classifies it as real or fake.

- `[ ]` **(Priority 12/20) Implementation of the Composite Loss Function:**
    - `[ ]` This is the core logic. The Generator's loss function will be a weighted sum of:
        - `[ ]` Adversarial Loss: How well it fools the Discriminator.
        - `[ ]` Predictive Loss: How high the Tc predicted by OracleNet is for the generated material (the goal is to maximize this).
        - `[ ]` (Optional) Regularization terms to ensure chemical validity.

- `[ ]` **(Priority 13/20) Implementation of the GAN Training Loop:**
    - `[ ]` Write the script that alternates between training the Discriminator (with real and fake data) and the Generator (using the composite loss). This cycle is more complex than in Phase II.

- `[ ]` **(Priority 14/20) Training the Complete GAN System:**
    - `[ ]` Run the GAN training. This step is computationally intensive and may require powerful GPUs.
    - `[ ]` Monitor the Generator and Discriminator losses to ensure training is stable.

- `[ ]` **(Priority 15/20) Generation of the Initial Batch of Candidates:**
    - `[ ]` Use the trained Generator to create a large number (thousands) of new molecular structures that do not exist in the database.

- `[ ]` **(Priority 16/20) Filtering and Ranking of Generated Candidates:**
    - `[ ]` Create a pipeline to evaluate the generated candidates:
        - `[ ]` Check basic chemical validity.
        - `[ ]` Run OracleNet to predict the Tc of each one.
        - `[ ]` Rank candidates from highest Tc to lowest.

## Phase IV: 🧪 Validation and Closing the Loop (Priorities 17-20)

Where AI meets the real world.

- `[ ]` **(Priority 17/20) Screening with Classical Computational Simulations:**
    - `[ ]` Take the top ~100 from the ranked list.
    - `[ ]` Perform more accurate, but slower, simulations (like DFT) to verify the stability and electronic properties of these candidates.

- `[ ]` **(Priority 18/20) Selection of Final Candidates for Synthesis:**
    - `[ ]` Based on AI results and computational screening, select a small number (1 to 5) of "champion" candidates for experimental validation.

- `[ ]` **(Priority 19/20) Collaboration for Synthesis and Laboratory Testing:**
    - `[ ]` This step requires collaboration with a materials physics or chemistry lab.
    - `[ ]` Partners will attempt to synthesize the proposed materials and measure their actual properties, including Tc.

- `[ ]` **(Priority 20/20) Closing the "Active Learning" Loop:**
    - `[ ]` The most important step for long-term success.
    - `[ ]` Take the experimental results (whether success or failure) from Phase 19.
    - `[ ]` Add these new data points to your original database.
    - `[ ]` Re-train OracleNet and, optionally, the GAN system with this new data. The system will become smarter with each iteration.
    - `[ ]` Repeat the cycle from Phase III/IV.
