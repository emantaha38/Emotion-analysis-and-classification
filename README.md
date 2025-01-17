# Emotion Recognition using Deep Learning

A deep learning-based emotion recognition system that classifies text into different emotional categories using LSTM networks and provides explanations using LIME.

## Project Overview

This project implements a bidirectional LSTM model to recognize emotions in text data. It includes comprehensive text preprocessing, model training, and model interpretation using LIME (Local Interpretable Model-agnostic Explanations).

## Features

- Text preprocessing pipeline with lemmatization and cleaning
- Bidirectional LSTM model for emotion classification
- Model interpretation using LIME
- Advanced visualization tools for analysis
- Batch processing capabilities
- Misclassification analysis

## Requirements

```
python >= 3.8
tensorflow >= 2.0
nltk
numpy
pandas
seaborn
matplotlib
lime
scikit-learn
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-recognition.git
cd emotion-recognition
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Project Structure

```
emotion-recognition/
├── data/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── lime_analyzer.py
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
└── README.md
```

## Usage

### Training the Model

```python
from src.model import load_data, normalize_text, prepare_data, create_sequences, build_model

# Load and preprocess data
df_train, df_val, df_test = load_data('train.txt', 'val.txt', 'test.txt')
df_train = normalize_text(df_train)
df_val = normalize_text(df_val)
df_test = normalize_text(df_test)

# Prepare data
(X_train, y_train), (X_val, y_val), (X_test, y_test), le = prepare_data(df_train, df_val, df_test)
X_train, X_val, X_test, tokenizer = create_sequences(X_train, X_val, X_test)

# Build and train model
model = build_model(vocab_size, X_train.shape[1])
history = train_model(model, X_train, y_train, X_val, y_val)
```

### Using LIME for Model Interpretation

```python
from src.lime_analyzer import EmotionLIMEAnalyzer

# Initialize analyzer
analyzer = EmotionLIMEAnalyzer(model, tokenizer, le)

# Analyze single text
result = analyzer.analyze_single_text("I'm feeling very happy today!")

# Batch analysis
texts = ["I'm excited!", "This is frustrating", "I feel sad"]
emotions = ["joy", "anger", "sadness"]
summary_df, avg_importance = analyzer.analyze_batch(texts, emotions)
analyzer.plot_batch_analysis(summary_df, avg_importance)
```

## Model Architecture

The emotion recognition model uses a Bidirectional LSTM architecture:
- Embedding layer (100 dimensions)
- Bidirectional LSTM layers (64 and 32 units)
- Dense layers for classification
- Dropout for regularization

## Performance

The model achieves the following metrics on the test set:
- Accuracy: X%
- F1-Score: X%
- Detailed performance metrics for each emotion category

## LIME Analysis Features

The project includes comprehensive LIME analysis tools:
- Single text analysis with visualization
- Batch analysis capabilities
- Feature importance visualization
- Misclassification analysis
- Confidence distribution plots

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/emotion-recognition](https://github.com/yourusername/emotion-recognition)

## Acknowledgments

- NLTK team for text processing tools
- LIME authors for interpretation framework
- Dataset source/creators (add specific references)
