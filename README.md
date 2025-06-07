# 🚫 AT&T Spam Detection - Deep Learning

## 📋 Project Overview

This project presents an automated spam detection solution for AT&T, using deep learning techniques to protect users from constant exposure to unwanted messages.

**Goal:**
To create a spam detector capable of automatically flagging unwanted messages as they arrive, based solely on the content of the SMS. 

## 🎯 Features

- Real-time automatic spam detection
- LSTM vs. BERT model comparison
- Detailed performance visualizations
- Comprehensive metrics evaluation
- Simple and intuitive interface
  
## 🔧 Technologies used

### Implemented models
- LSTM (Long Short-Term Memory)
- DistilBERT (Transfer learning)
## 📊 Project structure
```
├── main.ipynb # Main Notebook
├── img/
│ └── ATT-Logo.png # AT&T Logo
├── data/ # Message Dataset
└── README.md # Documentation
```

## 🚀 Technologies and Libraries
- `torch`
- `torchvision`
- `torchaudio`
- `transformers`
- `tensorflow`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `tqdm`
  
## 📈 Processing Pipeline

### Main Components
- **Data Preprocessing** for LSTM and BERT
- **Custom LSTM Architecture Implementation**
- **BERT Optimization** with DistilBERT
- **Visualization Features** without Subgraphs
- **Comprehensive Evaluation Metrics**

## 📈 Performance Metrics

### 🏆 Recommended model: BERT

- **Best overall accuracy** (98.21%)
- **Higher F1 score** (0.9301)
- **High recall** for spam detection
- **Balanced performance** between precision and recall
  
## 🎨 Visualizations

The The project includes several visualizations:
- Confusion matrices for each model
- Comparison graphs of metrics
- Performance trends during training
  
### Main features

## 📋 Detailed features

### Intelligent preprocessing

- Automatic text cleaning
- Special character handling
- Sequence normalization
  
### Robust models

- LSTM: Temporal dependency capture
- BERT: Advanced contextual understanding

### Evaluation
- Accuracy, Precision, Recall, F1 score
- Detailed confusion matrices
- Visual comparisons
  
## 🎯 Use cases

### Production deployment

- Integration with AT&T systems
- Real-time message filtering
- Automatic user protection
  
### Potential applications

- Email filtering
- Content moderation
- Communication security
  
## 🔮 Future improvements

- **Newer Models** (GPT, T5)
- **Multilingual Detection**
- **Real-Time Adaptation**
- **Interactive Web Interface**


## 📊 Model Performance Analysis

### Advantages of BERT:

- Higher Overall Accuracy (98.21% vs. 97.22%)
- Better F1 Score (0.9301 vs. 0.8848)
- Significantly Higher Recall (89.26% vs. 79.87%)
- Better Detection of Positive Cases
  
### Advantages of LSTM:

- Slightly Higher Accuracy (99.17% vs. 97.08%)
- Fewer false positives for spam prediction
  
### Recommendation:
Choose **BERT** for optimal overall performance, especially if detecting as many spam cases as possible is critical.

---
## 👤 Author
**Andriana's Project**

🔗 GitHub: [https://github.com/Andrianiniaina/0-DeepLearning-Project]
