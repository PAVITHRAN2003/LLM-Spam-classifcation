# LLM-Spam-classifcation

# LLM-Spam-Classification

## Project Description
This project explores the application of pre-trained Large Language Models (LLMs) for the task of email spam classification. It specifically utilizes and compares the performance of **DistilBERT** and **ALBERT** models on the Enron Spam Dataset, focusing on evaluation metrics such as accuracy, precision, recall, F1-score, and inference speed.

## Features
* Implementation of text classification using DistilBERT and ALBERT models.
* Data preparation and preprocessing for the Enron Spam Dataset.
* Comparative analysis of model performance in terms of classification metrics and inference speed.
* Demonstrates how to fine-tune transformer models for binary text classification.

## Dataset Used
* **Enron Spam Dataset**: This dataset contains email messages labeled as either "spam" or "ham" (non-spam) and is used for training and evaluating the classification models.
    * Dataset link: [https://huggingface.co/datasets/SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)

## Models Used
The following pre-trained LLMs from the Hugging Face Transformers library are utilized for the spam classification task:
* **DistilBERT**: A distilled version of BERT, known for being smaller, faster, and lighter while retaining much of BERT's performance.
* **ALBERT**: A Lite BERT for Self-supervised Learning of Language Representations, which uses parameter-reduction techniques to significantly lower memory consumption and increase training speed.

## Performance Comparison
The analysis conducted within this project indicates that DistilBERT generally offers a superior balance of top-tier accuracy and speed compared to ALBERT for this spam classification task.

Key observations include:
* **Speed**: DistilBERT demonstrated significantly faster evaluation times (approx. 14.6 seconds at 137 samples/sec) compared to ALBERT (approx. 32.8 seconds at 61 samples/sec). This dramatic throughput advantage makes DistilBERT particularly well-suited for applications where low latency and high volume processing are critical.
* **Accuracy**: While ALBERT still offers strong performance in a smaller model footprint, for a balance of top-tier accuracy and speed, DistilBERT was found to be the clear frontrunner.

## Setup and Installation
To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/LLM-Spam-Classification.git](https://github.com/your-username/LLM-Spam-Classification.git)
    cd LLM-Spam-Classification
    ```
2.  **Install dependencies:**
    The project requires the following Python libraries. You can install them using pip:
    ```bash
    pip install transformers datasets accelerate evaluate scikit-learn
    ```
    *(Note: Ensure you have a suitable environment, preferably with GPU support, for efficient model training and inference.)*

## Usage
The core logic for this spam classification project is contained within the Jupyter Notebook:
* `a2_bonus_llm_TEAMMATE1_ TEAMMATE2.ipynb`

To run the project:
1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "a2_bonus_llm_TEAMMATE1_ TEAMMATE2.ipynb"
    ```
2.  Execute the cells sequentially to load the dataset, preprocess data, fine-tune the models, evaluate their performance, and view the comparison.
