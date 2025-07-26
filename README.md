# FashionMNIST Attempt

A PyTorch-based implementation for classifying the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  
The project demonstrates a clean and modular approach to building deep learning classification pipelines by 
separating core PyTorch functionalities into a custom package called **`torcus`**, which serves as a **PyTorch 
Wrapper** for streamlined training and evaluation.

---

## 🚀 Features

- **Custom PyTorch Wrapper (`torcus`)**  
  - `CModel` class to wrap PyTorch models for classification tasks.  
  - Handles model initialization, training loops and basic evaluation metrics.
  - `CMetrics` class to go through the `CModel` class to compute metrics like confusion matrices, ROC curves, etc.  
  - Simplifies Jupyter notebooks by moving all boilerplate code into a reusable module.
  
- **Fashion-MNIST Classification**  
  - Example notebook `FashionMNIST.ipynb` showcasing training and evaluation.  
  - Preconfigured for **Accuracy**, **Confusion Matrix**, and **ROC metrics** (via `torchmetrics`).

- **Model Summary & Visualization**  
  - Uses `torchinfo` for model summaries.  
  - Includes Seaborn and Matplotlib for metric visualization.

---

## 🗂 Project Structure

```
FashionMNIST-Attempt/
│
├── FashionMNIST.ipynb          # Main notebook using the torcus wrapper
│
└── torcus/
    ├── classification.py       # PyTorch wrapper (CModel class)
    ├── requirements.txt        # Dependencies
    └── __init__.py
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mishrasumitranjan/FashionMNIST-Attempt.git
   cd FashionMNIST-Attempt
   ```

2. **Install dependencies:**
   ```bash
   pip install -r torcus/requirements.txt
   ```

---

## 🧩 Usage

1. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook FashionMNIST.ipynb
   ```

2. **Training a Model (Example):**
   Inside the notebook, a model can be initialized using the `CModel` class:
   ```python
   from torcus.classification import CModel
   from torch import nn
   
   # Define a simple CNN model
   model = nn.Sequential(
       nn.Conv2d(1, 32, 3, stride=1, padding=1),
       nn.ReLU(),
       nn.MaxPool2d(2),
       nn.Flatten(),
       nn.Linear(32 * 14 * 14, 10)
   )

   # Wrap the model
   cm = CModel(model, num_classes=10, input_size=[1, 1, 28, 28])
   cm.summary()   # Get model summary
   cm.load_data(train_loader, test_loader)
   cm.fit(epochs=10)
   ```

---

## 📦 `torcus` Wrapper Details

The **`CModel` class** in `torcus/classification.py` is designed to:
- Provide **train** and **evaluate** methods for classification models.
- Compute metrics like **loss** and **accuracy** automatically.
- Provide a `summary()` method powered by `torchinfo` for quick architecture inspection.

The **`CMetrics` class** in `torcus/classification.py` is designed to:
- Compute metrics like loss, accuracy, and confusion matrix.
- Provide `plot_confusion_matrix()` and `plot_roc_curve()` methods for visualizing the confusion matrix and ROC curve.
- Plot images alongside their predicted and truth labels for comparison.

This design ensures the notebook remains **minimal**, focusing only on:
- Dataset loading (e.g., Fashion-MNIST with `torchvision`).
- Model definition.
- Calling the `CModel` wrapper methods for training and evaluation.
- Calling the `CMetrics` class methods for metrics visualization.

---

## 📊 Results

**Training/Validation Accuracy vs. Loss Curve:**

![Accuracy vs. Loss](.github/assets/Accuracy%20vs.%20Loss.png)

**Classification Report:**  
The model achieved an accuracy of **89%**.

**Confusion Matrix:**

![Confusion Matrix](.github/assets/Confusion%20Matrix.png)

**ROC Curve:**

![ROC Curve](.github/assets/ROC%20Curve.png)

**Label Comparison:**

![Label Comparison](.github/assets/Label%20Comparison.png)

---

## 🔮 Future Improvements

- Expand `torcus` into a more general-purpose PyTorch utility library.
- Add data augmentation pipelines.
- Implement learning rate schedulers.

---

## 📜 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.