# BrainPrompt
This is the official PyTorch implementation of BrainPrompt from the paper " MICCAI 2025：BrainPrompt: Domain Adaptation with Prompt Learning for Multi-site Brain Network Analysis" accepted by 28th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI'25).
![image](https://github.com/user-attachments/assets/8ee65712-fc58-495e-802d-f9765413e860)

###  **Project Structure and Pipeline Overview**

This project follows a four-step pipeline for domain adaptation on multi-site brain network data:

#### 🔹 Step 1: Data Download and Preprocessing

Downloads raw ABIDE dataset from the public repository. Preprocesses the data, including time series extraction and functional connectivity construction.

data_process.py ABIDE_download.py

#### 🔹 Step 2: Pretraining on Source Domains

Trains a baseline model on aggregated source domains for initialization. Contains the architecture of the baseline network used for initial training.

baseline_train.py  baseline_model.py

#### 🔹 Step 3: Source prompts Learning

**3.1 Training Source Mask Prompts**

source_train1.py source_model1.py

**3.2 Computing Domain Similarity Scores and Integrating Mask Prompts**

similarity1_16.py/similarity2_16.py KnowledgeDistillation1.py/KnowledgeDistillation2.py

**3.3 Training Source Domain Embeddings**

source_train2.py source_model2.py

#### 🔹 Step 4: Target Domain Adaptation

Performs adaptation in the target domain using few-shot prompt tuning. Defines the model architecture used for target-domain adaptation.

target_train.py target_model.py

### 🔹 Entry Point and Configuration

Main entry point to run training and evaluation across different stages.Stores all configurable parameters, including training hyperparameters, dataset paths, and experiment settings.

main.py  setting.py
