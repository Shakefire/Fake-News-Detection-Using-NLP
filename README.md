# Fake News Detection Using NLP

## Project Description
This project implements a Fake News Detection system using Natural Language Processing (NLP) techniques. It leverages various machine learning models to classify news articles as fake or real based on their textual content. The system includes data loading, preprocessing, model training, prediction, and visualization components.

## Features
- Data loading and preprocessing for multiple datasets.
- Training of different machine learning models including Logistic Regression, Naive Bayes, and Random Forest.
- Model evaluation and history tracking.
- Prediction system for classifying new news articles.
- Visualization of results and model performance.
- Supports multiple datasets such as GossipCop and PolitiFact.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Shakefire/Fake-News-Detection-Using-NLP.git
   cd Fake-News-Detection-Using-NLP/Fake News Detection Using NLP
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
Run the training script to train the models on the datasets:
```bash
python training_app.py
```
This will preprocess the data, train the models, and save the trained models and their histories in the `models/` directory.

### Making Predictions
Use the prediction system to classify new news articles:
```bash
python prediction_system.py
```
This script loads the trained models and predicts the class of input news articles.

### Visualization
To visualize model performance and results, run:
```bash
python visualization.py
```

## Data
The project uses the following datasets located in the `data/` directory:
- GossipCop Fake News (`gossipcop_fake.csv`)
- GossipCop Real News (`gossipcop_real.csv`)
- PolitiFact Fake News (`politifact_fake.csv`)
- PolitiFact Real News (`politifact_real.csv`)

## Models
Trained models are saved in the `models/` directory with timestamps. Models include:
- Logistic Regression
- Naive Bayes
- Random Forest

Model history files track training metrics for analysis.

## Folder Structure
```
.
├── app.py                  # Main application script
├── training_app.py         # Script for training models
├── prediction_system.py    # Script for making predictions
├── data/                   # Dataset files
├── models/                 # Saved trained models and histories
├── predictions/            # Prediction history and outputs
├── utils.py                # Utility functions
├── data_loader.py          # Data loading functions
├── data_processor.py       # Data preprocessing functions
├── model_trainer.py        # Model training functions
├── model.py                # Model definitions
├── visualization.py        # Visualization scripts
├── pyproject.toml          # Project configuration
└── README.md               # Project documentation
```

## Requirements
- Python 3.8+
- Required Python packages listed in `requirements.txt`

## License
This project is licensed under the MIT License.

## Contact
For questions or support, please contact Shakefire.
