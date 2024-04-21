# Multi-Label Email Classification System

This repository contains the code for a multi-label email classification system. The system is designed to classify emails based on multiple dependent variables (labels) such as Type 2, Type 3, and Type 4. It provides a modular and extensible architecture that allows for easy modification and addition of preprocessing steps, machine learning models, and evaluation metrics.

## Features

- Separation of concerns (SoC) architecture with components for preprocessing, embeddings, modeling, etc.
- Supports two design decisions for multi-label classification:
  1. Chained Multi-outputs Approach: Trains a single model instance on chained labels (e.g., Type 2, Type 2+3, Type 2+3+4)
  2. Hierarchical Modeling Approach: Trains multiple model instances on filtered data based on the classes of preceding labels
- Encapsulates input data using a `Data` class for consistent access across models
- Implements multiple machine learning models with a consistent interface for training, prediction, and evaluation
- Provides a main controller (`main.py`) for orchestrating the preprocessing, embedding, modeling, and evaluation steps

## Repository Structure

- `main.py`: Main controller script for running the email classification system
- `preprocess.py`: Contains functions for data preprocessing, including de-duplication, noise removal, and translation
- `embeddings.py`: Implements functions for generating embeddings from text data (e.g., TF-IDF)
- `modelling/`: Directory containing modules related to modeling
  - `modelling.py`: Defines functions for model training, prediction, and evaluation
  - `data_model.py`: Implements the `Data` class for encapsulating input data
- `model/`: Directory containing implementations of various machine learning models
  - `base.py`: Defines the abstract base class for all models
  - `randomforest.py`: Implements the Random Forest model
  - `sgd.py`: Implements the Stochastic Gradient Descent (SGD) model
  - `adaboost.py`: Implements the AdaBoost model
  - `voting.py`: Implements the Voting Classifier model
  - `hist_gb.py`: Implements the Histogram-based Gradient Boosting model
  - `random_trees_ensembling.py`: Implements the Random Trees Embedding model
- `data/`: Directory containing the input data files
  - `AppGallery.csv`: Input data file for the AppGallery domain
  - `Purchasing.csv`: Input data file for the Purchasing domain
- `Config.py`: Configuration file for storing constants and settings

## Usage

1. Install the required dependencies (TODO: Create `requirements.txt` file).
2. Place the input data files (`AppGallery.csv` and `Purchasing.csv`) in the `data/` directory.
3. Modify the `Config.py` file to adjust any configuration settings if needed.
4. Run the `main.py` script to execute the email classification system.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
