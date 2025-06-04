# Tree Root Prediction Project - Setup Guide
===============================================

This repository contains the course project for the **Machine Learning** course at **Universitat Politècnica de Catalunya (UPC)**, completed as part of the **Big Data Management and Analytics (BDMA)** Erasmus Mundus Master's program.

## Project Overview

The objective of this project is to apply machine learning techniques to a real-world graph-based NLP problem: **predicting the root node of a syntactic dependency tree**. These trees are extracted from a multilingual parallel corpus comprising 21 languages.

The task is framed as a **binary classification** problem at the **node level**—for each node in the tree, predict whether it is the root or not. The central hypothesis is that **graph centrality measures** can help identify the root node based on its structural importance.

## Instructions to set up and run the project.

1. Clone 
------------------------------
```bash
# Clone the repository
git clone https://github.com/saracherif123/tree-root-prediction.git
cd tree-root-prediction
```

2. Create a virtual environment
------------------------------
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows (PowerShell or cmd)
```

3. Install requirements
------------------------------
```bash
pip install -r requirements.txt
```

4. run the model from the command line:
------------------------------
```bash
python3 notebooks/model.ipynb
```
