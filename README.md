# Detecting Quishing Attacks with Machine Learning Techniques Through QR Code Analysis

This repository contains the dataset and code associated with our paper accepted at the **8th International Conference on Optimization and Learning (2025) in Dubai**. Our research introduces a novel approach for detecting QR code-based phishing attacks (**“Quishing”**) by analyzing QR code structure and pixel patterns without extracting the embedded content.

## **📜 Abstract**

The rise of QR code-based phishing (*“Quishing”*) poses a growing cybersecurity threat, as attackers increasingly exploit QR codes to bypass traditional phishing defenses. Existing detection methods predominantly focus on URL analysis, requiring payload extraction, which may inadvertently expose users to malicious content. Moreover, QR codes encode various data types beyond URLs (e.g., Wi-Fi credentials, payment information), making URL-based detection insufficient for broader security concerns.

To address these gaps, we propose the **first framework** for quishing detection that directly analyzes **QR code structure and pixel patterns**. We generated a dataset of phishing and benign QR codes and used it to train and evaluate multiple machine learning models, including **Logistic Regression, Decision Trees, Random Forest, Naïve Bayes, LightGBM, and XGBoost**. Our best-performing model (**XGBoost**) achieves an **AUC of 0.9106**, demonstrating the feasibility of QR-centric detection.

Through feature importance analysis, we identify key visual indicators of malicious intent and refine our feature set by removing non-informative pixels, improving performance to an **AUC of 0.9133** with a reduced feature space. Our findings reveal that the structural features of QR codes correlate strongly with phishing risk. This work establishes a foundation for **quishing mitigation** and highlights the potential of **direct QR analysis** as a critical layer in modern phishing defenses.

---

## **📂 Dataset**

The dataset consists of **9,987 QR codes** stored as numpy arrays, each with dimensions **(69x69)**, along with corresponding labels indicating whether the QR code is phishing or benign.

- `qr_codes_29.pickle` → Contains the QR code images as a **numpy array** (`shape: (9987, 69, 69)`).
- `qr_codes_29_labels.pickle` → Contains the labels (`shape: (9987,)`), where:
  - **1 = Phishing QR Code**
  - **0 = Benign QR Code**

---

## **⚙️ How to Use**

### **1️⃣ Install Dependencies**
Ensure you have the required Python libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### **2️⃣ Load the Dataset**
```python
import pickle

# Load QR code images
with open('qr_codes_29.pickle', 'rb') as f:
    qr_codes = pickle.load(f)

# Load labels
with open('qr_codes_29_labels.pickle', 'rb') as f:
    labels = pickle.load(f)

print(type(qr_codes), qr_codes.shape)
print(type(labels), labels.shape)
```

### **3️⃣ Visualize Sample QR Codes**
```python
import numpy as np
import matplotlib.pyplot as plt

# Select 10 random samples
indices = np.random.choice(len(qr_codes), 10, replace=False)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(qr_codes[indices[i]], cmap="gray")
    ax.set_title(f"Label: {labels[indices[i]]}")
    ax.axis("off")

plt.tight_layout()
plt.show()
```

## **📝 Citation**
If you use this dataset or code in your research, please cite our work:

```bibtex
@inproceedings{trad2025quishing,
  author    = {Fouad Trad, Ali Chehab},
  title     = {Detecting Quishing Attacks with Machine Learning Techniques Through QR Code Analysis},
  booktitle = {Proceedings of the 8th International Conference on Optimization and Learning},
  year      = {2025},
  location  = {Dubai, UAE}
}

```
