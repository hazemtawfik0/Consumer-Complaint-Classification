Consumer Complaint Classification

A text-classification project for routing consumer complaints into the correct financial category.

The project began with a controlled comparison between recurrent neural networks and a pretrained Transformer. After the Transformer produced the strongest pilot results, it was retrained on a larger 60,000-record dataset and packaged in a local Gradio application.

Project overview

The model classifies a written complaint into one of five categories:

Credit reporting

Debt collection

Mortgages and loans

Credit card

Retail banking

The repository includes the training workflow, evaluation results, and a local web interface for testing new complaints.

Model development

Pilot comparison

Four models were trained and evaluated on the same 15,999-record sample:

Model

Accuracy

Macro F1

Transformer

79.67%

79.08%

SimpleRNN

77.21%

74.62%

LSTM

76.17%

74.61%

GRU

74.38%

70.32%

The Transformer achieved the best overall performance and was selected for the final stage.

Final model

The final DistilBERT classifier was trained on a stratified dataset of 60,000 complaints:

Training: 42,000 records

Validation: 9,000 records

Test: 9,000 records

Final test results:

Metric

Score

Accuracy

83.1%

Macro F1

82.3%

Macro F1 was used as the main selection metric because the class distribution is imbalanced.

Pipeline

The training pipeline includes:

Dataset loading and exploration

Missing-value and duplicate removal

Text normalization

Stratified train, validation, and test splits

Label encoding

Class weighting

DistilBERT fine-tuning

Accuracy, precision, recall, and F1 evaluation

Confusion matrix and error analysis

Model export and local deployment

Local application

The project includes a Gradio interface that runs the trained model locally.

The interface shows:

Predicted complaint category

Confidence score

Probability distribution across all categories

Inference details

Example complaints

Repository structure

Consumer-Complaint-Classification/
├── app.py
├── requirements.txt
├── setup_windows.bat
├── run_app.bat
├── verify_setup.py
├── experiment_config.json
├── final_transformer_metrics.csv
├── final_60k_class_distribution.csv
├── label_encoder.joblib
├── README.md
└── notebooks/

Large generated files are not stored in the normal Git history:

.venv/

checkpoints/

fine_tuned_transformer/

model weight files

large ZIP archives

Download the trained model

The trained model is distributed separately because the weight file is larger than GitHub's regular file limit.

Download the latest deployment package from:

https://github.com/hazemtawfik0/Consumer-Complaint-Classification/releases/latest

After downloading, extract the model into the project root so the folder structure contains:

Consumer-Complaint-Classification/
└── fine_tuned_transformer/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt

Run the application on Windows

1. Install Python

Python 3.11 64-bit is recommended.

Check the installation:

py -3.11 --version

2. Create the environment and install packages

From the project folder:

.\setup_windows.bat

The script creates a local virtual environment and installs the required packages.

3. Verify the setup

.\.venv\Scripts\python.exe verify_setup.py

4. Start the application

.\run_app.bat

The interface will open at:

http://127.0.0.1:7860

Keep the terminal open while the application is running.

Run manually

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py

Main libraries

Python

PyTorch

Hugging Face Transformers

scikit-learn

pandas

Gradio

Notes

The confidence value shown by the application is the model's softmax output. It should be treated as a model score rather than a guarantee.

The pilot comparison and the final 60,000-record experiment serve different purposes:

The pilot experiment provides the fair comparison between all four architectures.

The final experiment scales the selected Transformer to a larger dataset.

Future improvements

Possible next steps include:

Probability calibration

Hyperparameter tuning

Longer input sequences

Testing larger Transformer models

Adding model monitoring and data-drift checks

Deploying the application through Docker or Hugging Face Spaces
