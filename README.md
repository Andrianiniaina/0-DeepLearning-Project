# 🚫 AT&T Spam Detection - Deep Learning

![AT&T Logo](img/ATT-Logo.png)

## 📋 Project Overview

This project presents an automated spam detection solution for AT&T, using deep learning techniques to protect users from constant exposure to unwanted messages.

**Goal:** Create a spam detector capable of automatically flagging spam messages as soon as they arrive, based solely on the text message content.

## 🎯 Features

- **Automatic spam detection** in real-time
- **Model comparison** LSTM vs BERT
- **Detailed visualizations** of performance
- **Comprehensive metrics** evaluation
- **Simple and intuitive** interface

## 🔧 Technologies Used

### Main Libraries
- **TensorFlow/Keras** - LSTM models
- **PyTorch** - Deep learning infrastructure
- **Transformers (Hugging Face)** - BERT models
- **Scikit-learn** - Metrics and preprocessing
- **Pandas/NumPy** - Data manipulation

### Implemented Models
- **LSTM** (Long Short-Term Memory)
- **DistilBERT** (Transfer Learning)

## 📊 Project Structure

```
├── main.ipynb                 # Main notebook
├── img/
│   └── ATT-Logo.png          # AT&T Logo
├── data/                     # Message dataset
└── README.md                 # Documentation
```

## 🚀 Installation and Setup

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install tensorflow
pip install scikit-learn
pip install pandas numpy
pip install matplotlib seaborn
pip install tqdm
```

### GPU/CPU Configuration
The code automatically detects GPU availability:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 📈 Processing Pipeline

### 1. 📊 Data Loading and Exploration
- Import message dataset
- Exploratory data analysis
- Descriptive statistics

### 2. ⚙️ Preprocessing
- **For LSTM:** Tokenization, padding, sequences
- **For BERT:** BERT tokenization, attention masks
- Text cleaning and normalization

### 3. 🤖 Modeling
- **LSTM:** Custom recurrent architecture
- **BERT:** Fine-tuning pre-trained DistilBERT

### 4. 🎯 Training
- Training both models
- Cross-validation
- Hyperparameter optimization

### 5. 📊 Evaluation
- Performance metrics
- Confusion matrices
- Model comparison

## 📈 Performance Metrics

### Comparative Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| LSTM  | 0.9722   | 0.9917    | 0.7987 | 0.8848   |
| BERT  | 0.9821   | 0.9708    | 0.8926 | 0.9301   |

### 🏆 Recommended Model: BERT
- **Best overall accuracy** (98.21%)
- **Superior F1-Score** (0.9301)
- **High recall** for spam detection
- **Balanced performance** between precision and recall

## 🎨 Visualizations

The project includes several visualizations:
- **Confusion matrices** for each model
- **Comparative charts** of metrics
- **Performance evolution** during training

## 🔧 Usage

### Complete Execution
```python
if __name__ == "__main__":
    main()
```

### Main Functions
```python
# Results visualization
visualize_results(lstm_trainer, results)

# Final summary
print_final_summary(results)
```

## 📋 Detailed Features

### Intelligent Preprocessing
- Automatic text cleaning
- Special character handling
- Sequence normalization

### Robust Models
- **LSTM:** Captures temporal dependencies
- **BERT:** Advanced contextual understanding

### Comprehensive Evaluation
- Accuracy, Precision, Recall, F1-Score
- Detailed confusion matrices
- Visual comparisons

## 🎯 Use Cases

### Production Deployment
- Integration into AT&T systems
- Real-time message filtering
- Automatic user protection

### Potential Applications
- Email filtering
- Content moderation
- Communication security

## 🔮 Future Improvements

- **Newer models** (GPT, T5)
- **Multilingual detection**
- **Real-time adaptation**
- **Interactive web interface**

## 📞 Support

For any questions or issues:
- Check GPU/CPU configuration
- Ensure all dependencies are installed
- Review training logs

## 🏷️ Versions

- **Transformers:** 4.52.4
- **TensorFlow:** Compatible with installed version
- **PyTorch:** Stable version recommended

## 📚 Code Structure

### Main Components
- **Data preprocessing** for both LSTM and BERT
- **Custom LSTM architecture** implementation
- **BERT fine-tuning** with DistilBERT
- **Visualization functions** without subplots
- **Comprehensive evaluation** metrics

### Key Functions
```python
visualize_results(lstm_trainer, results)  # Generate visualizations
print_final_summary(results)              # Display final comparison
```

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies** from requirements
3. **Run the main notebook** `main.ipynb`
4. **Review results** and visualizations
5. **Deploy the best model** (BERT recommended)

## 📊 Model Performance Analysis

### BERT Advantages:
- Higher overall accuracy (98.21% vs 97.22%)
- Better F1-Score (0.9301 vs 0.8848)
- Significantly higher recall (89.26% vs 79.87%)
- Better at detecting positive cases

### LSTM Advantages:
- Slightly higher precision (99.17% vs 97.08%)
- Fewer false positives when predicting spam

### Recommendation:
Choose **BERT** for optimal overall performance, especially if detecting maximum spam cases is critical.

---

## 👤 Author

Project by **Andriana**  
🔗 GitHub: [https://github.com/Andrianiniaina/0-DeepLearning-Project]
