# tree-root-prediction

This repository contains the course project for the **Machine Learning** course at **Universitat Politècnica de Catalunya (UPC)**, completed as part of the **Big Data Management and Analytics (BDMA)** Erasmus Mundus Master's program.

## Project Overview

The objective of this project is to apply machine learning techniques to a real-world graph-based NLP problem: **predicting the root node of a syntactic dependency tree**. These trees are extracted from a multilingual parallel corpus comprising 21 languages.

The task is framed as a **binary classification** problem at the **node level**—for each node in the tree, predict whether it is the root or not. The central hypothesis is that **graph centrality measures** can help identify the root node based on its structural importance.


Tree Root Prediction Project - Setup Guide
=========================================

This guide provides step-by-step instructions to set up and run the tree root prediction project.

1. Clone and Run
---------------
```bash
# Clone the repository
git clone https://github.com/saracherif123/tree-root-prediction.git
cd tree-root-prediction

# Install requirements
pip install -r requirements.txt

# Run the model
python3 notebooks/model.ipynb
```

2. Run the model directly from the command line, use:
```bash
python3 notebooks/model.ipynb
```
