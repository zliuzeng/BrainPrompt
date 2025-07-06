# BrainPrompt
This is the official PyTorch implementation of BrainPrompt from the paper " MICCAI 2025：BrainPrompt: Domain Adaptation with Prompt Learning for Multi-site Brain Network Analysis" accepted by 28th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI'25).
![image](https://github.com/user-attachments/assets/8ee65712-fc58-495e-802d-f9765413e860)

 Project Structure and Pipeline Overview
This project follows a four-step pipeline for domain adaptation on multi-site brain network data:

🔹 Step 1: Data Download and Preprocessing
ABIDE_download.py
Downloads raw ABIDE dataset from the public repository.

data_process.py
Preprocesses the data, including time series extraction, functional connectivity construction, and atlas alignment.

🔹 Step 2: Pretraining on Source Domains
baseline_train.py
Trains a baseline model on aggregated source domains for initialization.

baseline_model.py
Contains the architecture of the baseline network used for initial training.

🔹 Step 3: Source Domain Learning and Similarity Computation
source_train.py
Trains the prompt-based model on selected source domains.

source_model.py
Defines the prompt-based model structure used in source domain learning.

similarity1_16.py, similarity2_16.py
Computes the similarity between different domains (e.g., feature distance) to guide prompt transfer.

🔹 Step 4: Target Domain Adaptation
target_train.py
Performs adaptation in the target domain using few-shot prompt tuning.

target_model.py
Defines the model architecture used for target-domain adaptation.

🔹 Entry Point and Configuration
main.py
Main entry point to run training and evaluation across different stages.

setting.py
Stores all configurable parameters, including training hyperparameters, dataset paths, and experiment settings.
