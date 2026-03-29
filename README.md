# Task 1: Mental Health in Technology-related Jobs

**Course: Machine Learning: Unsupervised Learning and Feature Engineering (DLBDSMLUSL01)**  
**IU International University of Applied Sciences**

---

## Overview

This project analyses the OSMI Mental Health in Tech Survey 2016 dataset using 
unsupervised machine learning techniques. The goal is to categorise survey 
participants into meaningful groups based on their responses and provide 
visualisations that reduce the complexity of the data while preserving its 
main characteristics.

The analysis was conducted to support the HR department in identifying 
points of leverage for a pre-emptive mental health programme targeted at 
technology-oriented employees.

---

## Dataset

**Source:** OSMI Mental Health in Tech Survey 2016  
**Available at:** https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016  

**Respondents:** 1,428 (after cleaning)  
**Original features:** 63  
**Features after preprocessing:** 56  

The dataset contains responses from technology-oriented professionals across 
53 countries, covering topics such as employer mental health support, personal 
mental health history, workplace attitudes, and demographic information.

---

## Repository Structure
```
iu-case-study-ml-unsupervised-learning-feature-engineering-task1
│
├── iu_task1_ml_unsupervised_learning.ipynb   -> Google Colab Notebook
├── README.md
│
└── figures/                              -> All visualisations from the Google Colab Notebook
    ├── 1_missing_values.png
    ├── 2_age_distribution.png
    ├── 3_employment_distribution.png
    ├── 4_country_distribution.png
    ├── 5_correlation_heatmap.png
    ├── 5.1_correlation_heatmap_legend.png
    ├── 6_pca_variance.png
    ├── 7_pca_2d.png
    ├── 8_mds_2d.png
    ├── 9_elbow_method.png
    ├── 10_silhouette_scores.png
    ├── 11_kmeans_pca.png
    ├── 12_algorithm_comparison.png
    ├── 13_gmm_hierarchical_comparison.png
    ├── 14_cluster_profile_heatmap.png
    ├── 15_diagnosis_distribution.png
    ├── 16_work_position_distribution.png
    └── 17_work_interference.png
```

---

## Approach

### Data Preprocessing
- Columns with over 79% missing values were dropped
- Employer-related columns missing due to self-employment were filled 
  with sentinel values (MNAR treatment)
- Gender standardised from 70 free-text variations to 4 categories
- Work position simplified from 264 combinations to primary roles
- Three diagnosis columns consolidated into one feature
- Ordinal encoding for ordered categories, one-hot encoding for nominal

### Feature Selection
- Zero-variance features removed (6 columns)
- Highly correlated features removed, threshold > 0.85 (6 columns)
- Final feature set: 56 features

### Dimensionality Reduction
- PCA applied as primary method — 38 components retain 90% of variance
- MDS applied for comparison — produced no meaningful structure
- PCA 2D projection used for cluster visualisation

### Clustering

Three algorithms were tested; K-Means, Gaussian Mixture Models, and 
Hierarchical Clustering - all with k=3. K-Means was selected as the final 
algorithm due to its balanced and interpretable cluster distribution. GMM 
achieved a marginally higher silhouette score but assigned 73.6% of 
respondents to a single cluster, making it unsuitable for meaningful 
interpretation.

---

## Results

Three clusters were identified:

**Cluster 0 — Employed, High Mental Health Burden (683 respondents, 47.8%)**  
High rates of current (78%) and past (88.5%) mental health disorders. 
Professionally diagnosed in 81.7% of cases. Significant work interference 
even when receiving treatment. Most affected group in the dataset.

**Cluster 1 — Employed, Low Mental Health Burden (459 respondents, 32.1%)**  
Over 81% report no diagnosis. Near-zero work interference. Low family 
history of mental illness. Represents the mentally healthy employed segment.

**Cluster 2 — Self-Employed, Moderate Mental Health Burden (286 respondents, 20.0%)**  
Entirely self-employed, average age 36.88. Over half report a current mental 
health disorder. No access to employer-provided mental health resources, 
making their situation structurally distinct from the employed clusters.

---

## How to Run

### Step 1 — Open the Notebook

Go to https://colab.research.google.com and upload:
```
iu_task1_ml_unsupervised_learning.ipynb
```

### Step 2 — Set Up Kaggle API

In Kaggle, go to Settings → API → Create New Token.  
Add the token to Colab Secrets with the name `KAGGLE_API_TOKEN`.  
The notebook will download the dataset automatically.

### Step 3 — Run All Cells

Go to Runtime → Run All.  
The full pipeline runs from data loading to cluster visualisation.

---

## Requirements

The notebook runs on Google Colab with no local installation required.  
All libraries used are available in the standard Colab environment.

Key libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## Author

Siddharthsinh Rathod  
Course: Machine Learning: Unsupervised Learning and Feature Engineering (DLBDSMLUSL01)  
IU International University of Applied Sciences
