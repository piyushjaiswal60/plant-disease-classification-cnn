# 🌿 Plant Disease Classification using CNN

A deep learning project for classifying plant leaf diseases using the PlantVillage dataset.

The project compares a lightweight Custom CNN with MobileNetV2 transfer learning and provides an interactive Streamlit application for real-time image classification.

---

## 🚀 Features

- Plant disease classification across 38 classes
- Custom CNN built from scratch
- MobileNetV2 transfer learning with ImageNet weights
- Model performance comparison
- Classification report
- Confusion matrix
- Top-3 predictions
- Confidence scores
- Interactive Streamlit application
- Docker deployment support

---

## 📊 Dataset

The project uses the **PlantVillage dataset**.

- Total images: 54,305
- Number of classes: 38
- Training images: 37,998
- Validation images: 8,145
- Test images: 8,162

Dataset split:

- 70% Training
- 15% Validation
- 15% Testing

A fixed random seed of 42 was used for reproducibility.

---

## 🧠 Models

### Custom CNN

A lightweight CNN developed from scratch using:

- Conv2D
- Batch Normalization
- MaxPooling
- Global Average Pooling
- Dropout
- Dense classification layer

Total parameters:

**115,558**

Model size:

**1.40 MB**

Test accuracy:

**93.40%**

---

### MobileNetV2

MobileNetV2 pretrained on ImageNet was used for transfer learning.

The pretrained base was frozen and a custom classification head was added for the 38 PlantVillage classes.

Total parameters:

**2,306,662**

Model size:

**9.74 MB**

Test accuracy:

**96.18%**

---

## 📈 Model Comparison

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Test Accuracy | 93.40% | **96.18%** |
| Macro F1 | ~0.91 | **0.9486** |
| Weighted F1 | ~0.93 | **0.9620** |
| Parameters | **115,558** | 2,306,662 |
| Model Size | **1.40 MB** | 9.74 MB |

### Selected Model

MobileNetV2 was selected as the higher-performing model because it achieved better test accuracy and F1-score.

However, the Streamlit application allows predictions from **both models side-by-side**, making it possible to observe cases where the models agree or disagree.

---

## 🖥️ Streamlit Application

The application allows users to:

1. Upload a plant leaf image
2. Preview the image
3. Run both models
4. View the predicted disease
5. View confidence scores
6. View the top 3 predictions from each model
7. Compare Custom CNN and MobileNetV2 predictions

Run the application:

```bash
streamlit run app.py