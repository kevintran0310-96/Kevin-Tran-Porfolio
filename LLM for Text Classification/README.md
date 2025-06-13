# 🚀 LLM-Powered Illicit Content Detection

### A deep-dive comparison of LLMs (Llama 3, Gemma) vs. traditional ML for enhancing online safety.

This project tackles the critical challenge of identifying and classifying illicit content on online marketplaces. I developed and benchmarked a suite of models to automatically detect harmful listings, comparing the power of modern Large Language Models (LLMs) against established machine learning techniques.

---

*Quoc Khoa Tran (Kevin) | [Portfolio](https://your-portfolio-link.com](https://github.com/kevintran0310-96/Kevin-Tran-Porfolio/)) | [LinkedIn](https://www.linkedin.com/in/kevintran0310/) | [khoatran031096@gmail.com](mailto:khoatran031096@gmail.com)*

---

### 🌟 Key Features

* **Advanced LLM Fine-Tuning**: Implemented and fine-tuned multiple state-of-the-art models including **Meta Llama 3.2** and **Google Gemma**.
* **Efficient Training Techniques**: Leveraged **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA (Low-Rank Adaptation)** and **4-bit Quantization (QLoRA)** to train large models on limited hardware.
* **Comprehensive Benchmarking**: Conducted a rigorous comparative analysis against baseline models like **BERT**, **Support Vector Machines (SVM)**, and **Naive Bayes**.
* **Dual Classification Tasks**:
    1.  **Binary Classification**: Identifying content as `illicit` or `non-illicit`.
    2.  **Multi-Class Classification**: Categorizing content into **40 distinct illicit types**.
* **Robust Evaluation**: Addressed significant class imbalance using weighted loss functions to ensure fair and accurate model performance metrics.

### 🛠️ Tech Stack

* **Core Libraries**: Python, PyTorch, Hugging Face (Transformers, PEFT, Datasets, Accelerate), Scikit-learn
* **Data Handling**: Pandas, NumPy
* **Experiment Tracking**: Weights & Biases (WandB)
* **Text Processing**: NLTK
* **Development Environment**: Jupyter Notebook

### 📊 Performance Highlights

The results show a clear, task-dependent advantage for different model architectures.

#### **Binary Classification (Illicit vs. Non-Illicit)**

While LLMs perform well, a well-tuned **SVM proves to be a powerful and efficient baseline**, slightly outperforming Llama 3.2.

| Model       | Accuracy | F1 (Macro) | Precision (Macro) | Recall (Macro) |
| :---------- | :------: | :--------: | :---------------: | :------------: |
| **SVM** | **0.90** | **0.81** | **0.88** | **0.77** |
| **Llama 3.2** |   0.89   |    0.80    |       0.87        |      0.77      |
| **Naive Bayes**|   0.86   |    0.73    |       0.82        |      0.69      |
| **BERT** |   0.84   |    0.58    |       0.92        |      0.57      |
| **Gemma 3** |   0.83   |    0.70    |       0.73        |      0.68      |

#### **Multi-Class Classification (40 Illicit Categories)**

In the more complex multi-class task, **Llama 3.2's advanced semantic understanding allows it to significantly outperform all other models.**

| Model       | Accuracy | F1 (Macro) | Precision (Weighted) | Recall (Weighted) |
| :---------- | :------: | :--------: | :------------------: | :---------------: |
| **Llama 3.2** | **0.74** | **0.61** |      **0.75** |     **0.74** |
| **SVM** |   0.72   |    0.44    |         0.78         |       0.72        |
| **Gemma 3** |   0.68   |    0.54    |         0.68         |       0.68        |
| **BERT** |   0.68   |    0.34    |         0.60         |       0.68        |
| **Naive Bayes**|   0.54   |    0.16    |         0.53         |       0.54        |

### 💡 My Contribution & Methodology

As the sole author of this project, I was responsible for the end-to-end research and development cycle. My key contributions include:

* **System Design**: Architecting the entire comparison framework, from data preprocessing to model evaluation.
* **Data Preprocessing**: Designing the text cleaning and feature engineering pipeline for the multilingual DUTA10K dataset.
* **Model Implementation**:
    * Building and tuning the TF-IDF pipelines for the SVM and Naive Bayes models.
    * Implementing the fine-tuning process for BERT, Llama 3.2, and Gemma. This involved a deep dive into the Hugging Face ecosystem to apply **QLoRA (Quantized Low-Rank Adaptation)**, strategically selecting quantization parameters and LoRA target modules to balance performance with resource constraints.
* **Analysis**: Systematically evaluating model outputs, generating performance metrics, and drawing actionable conclusions from the results.

All code in this repository is my original work, created for this research.

### 📂 Repository Structure

The project is organized into modular Jupyter notebooks for easy reproducibility of each experiment.

```bash
├── Using LLM To Detect Illicit Content On Online Marketplaces.pdf        # The full research paper
├── classicmodels_binaryclassification.ipynb    # SVM & Naive Bayes (Binary)
├── classicmodels_multiclassification.ipynb   # SVM & Naive Bayes (Multi-class)
├── BERT_binaryclassification.ipynb           # BERT (Binary)
├── BERT_multiclassification.ipynb          # BERT (Multi-class)
├── llama3.2_binaryclassification.ipynb       # Llama 3.2 (Binary)
├── llama3.2_multiclassification.ipynb      # Llama 3.2 (Multi-class)
├── Gemma3-4b_binaryclassification.ipynb    # Gemma 3 (Binary)
├── Gemma3-4b_multiclassification.ipynb   # Gemma 3 (Multi-class)
└── README.md
```
### ⚙️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install dependencies:**
    It's best to use a virtual environment.
    ```bash
    pip install -U torch transformers datasets peft bitsandbytes scikit-learn pandas numpy nltk evaluate wandb
    ```

3.  **Download NLTK data:**
    Run the following in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

4.  **Launch the notebooks**: Run the Jupyter notebook for the experiment you wish to reproduce.
    * **Important**: The LLM notebooks (`llama3.2` and `Gemma3-4b`) require a CUDA-enabled GPU.

---
