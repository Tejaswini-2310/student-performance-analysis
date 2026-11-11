# Student Performance Prediction and Analysis Using Machine Learning

## Project Overview

This project focuses on analyzing and predicting student academic performance using Machine Learning techniques. It investigates the factors that influence student performance and compares different machine learning models for identifying students who are at risk of failing.

The project uses the UCI Student Performance Dataset, which contains academic, demographic, family and social information about students.

The experiments are conducted on two datasets:
- Mathematics (Math)
- Portuguese Language (Portuguese)

## Objectives

- Analyze student academic performance using machine learning.
- Predict whether a student is likely to pass or fail.
- Compare the performance of different machine learning models.
- Evaluate models using Accuracy, Precision, Recall, F1-score and ROC-AUC.
- Identify important factors influencing student academic performance.
- Study the effect of different feature-selection scenarios.
- Analyze the stability of feature importance across repeated experiments.

## Dataset

**Dataset:** UCI Student Performance Dataset

The dataset includes two subjects:

| Dataset | Number of Students |
|---|---:|
| Mathematics | 395 |
| Portuguese | 649 |
| Total | 1,044 |

The datasets contain information related to:
- Student demographics
- Family background
- Study time
- Absences
- Previous grades
- Family support
- School-related activities

### Target Variable

The target variable is created using the final grade (G3):

- Fail (1): G3 less than 10
- Pass (0): G3 greater than or equal to 10

Failure is treated as the positive class so that Recall measures the proportion of failing students correctly identified by the model.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SciPy
- VS Code
- Git and GitHub

## Machine Learning Methodology

The project follows these steps:

1. Dataset loading and preprocessing.
2. Feature preparation and encoding.
3. Model training.
4. Repeated Stratified K-Fold Cross-Validation.
5. Performance evaluation.
6. Feature importance analysis using permutation importance.
7. Model comparison across experimental scenarios.
8. Visualization of experimental results.

## Evaluation Metrics

The following evaluation metrics are used:

- **Accuracy:** Overall percentage of correct predictions.
- **Precision:** Percentage of predicted failures that are actual failures.
- **Recall:** Percentage of actual failing students correctly identified.
- **F1-score:** Harmonic mean of Precision and Recall.
- **ROC-AUC:** Ability of the model to distinguish between pass and fail classes.

## Project Structure

```text
student-performance-analysis/
│
├── code.py
│
├── data/
│   ├── student-mat.csv
│   └── student-por.csv
│
├── results/
│   ├── change_A_to_C.csv
│   ├── fig1_f1_by_scenario.png
│   ├── fig2_top_factors_Math.png
│   ├── fig2_top_factors_Portuguese.png
│   ├── fig3_recall_by_scenario.png
│   ├── permutation_rank_summary_Math.csv
│   ├── permutation_rank_summary_Portuguese.csv
│   ├── permutation_ranks_raw.csv
│   ├── permutation_stability.csv
│   ├── raw_scores.csv
│   ├── summary_mean_std.csv
│   └── summary_means.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Tejaswini-2310/student-performance-analysis.git
```

### 2. Navigate to the project directory

```bash
cd student-performance-analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the UCI Student Performance Dataset and place the following files inside the `data/` directory:

- student-mat.csv
- student-por.csv

The CSV files use semicolon (`;`) as the separator.

### 5. Run the project

```bash
python code.py
```

### 6. View the results

The generated CSV files and graphs will be saved in the `results/` directory.

## Experimental Results

The program successfully processes both datasets.

| Dataset | Students | Failure Rate |
|---|---:|---:|
| Mathematics | 395 | 32.9% |
| Portuguese | 649 | 15.4% |

The project generates visualizations to analyze:

1. F1-score comparison across experimental scenarios.
2. Important factors influencing Mathematics performance.
3. Important factors influencing Portuguese performance.
4. Recall comparison across experimental scenarios.
5. Stability of feature importance rankings.

Detailed experimental results are available in the `results/` directory.

## Key Outcomes

- Analyzed academic performance across two student datasets.
- Implemented a machine learning-based pass/fail prediction experiment.
- Compared model performance using multiple evaluation metrics.
- Investigated important academic and social factors using permutation importance.
- Examined the stability of feature rankings across repeated experiments.

## Future Enhancements

- Develop a web-based student performance prediction application.
- Integrate real-time student performance monitoring.
- Introduce explainable AI techniques for better model interpretation.
- Extend the analysis to additional academic datasets.
- Develop an early-warning system to identify students at risk of failure.

## Author

**Tejaswini Reddy**

GitHub: https://github.com/Tejaswini-2310

---

*This project was developed as a Machine Learning research and experimental analysis project.*
