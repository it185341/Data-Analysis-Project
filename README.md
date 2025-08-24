# PROJECT: Data Analysis Using Knowledge Discovery Algorithms

## Part A – Classification

The spam dataset originates from a collection of email messages. The features recorded for each message are shown in the table below:

| Feature   | Description |
|-----------|-------------|
| cap_ave   | Average length of consecutive capital letters |
| remove    | Percentage occurrence of the word “remove” (1: percentage = 0%, 2: percentage > 0%) |
| 000       | Percentage occurrence of the number “000” (1: percentage ≤ 0.3%, 2: percentage > 0.3%) |
| money     | Percentage occurrence of the word “money” (1: percentage ≤ 0.1%, 2: percentage > 0.1%) |
| free      | Percentage occurrence of the word “free” (1: percentage ≤ 0.2%, 2: percentage > 0.2%) |
| our       | Percentage occurrence of the word “our” (1: percentage ≤ 0.1%, 2: percentage > 0.1%) |
| char_$    | Percentage occurrence of the symbol “$” (1: percentage ≤ 0.1%, 2: percentage > 0.1%) |
| char_!    | Percentage occurrence of the symbol “!” (1: percentage ≤ 0.1%, 2: 0.1% < percentage ≤ 0.4%, 3: percentage > 0.4%) |
| class     | Message category (email, spam) |

**Objective:** Develop a mechanism to classify messages as either “email” or “spam”.  
You are required to implement a comparative study of different classification algorithms. Specifically, you need to apply and compare the following algorithms:

- Decision Trees (C4.5 - J48)  
- k-Nearest Neighbors (kNN - IBk)  
- Naive Bayes  
- Random Forest  

You may perform any preprocessing you consider useful (e.g., oversampling - SMOTE, attribute selection). You can use any validation method (cross-validation, train-test split). **Important:** if you perform oversampling, the test set must not contain synthetic instances.

---

## Part B – Clustering

The Student Performance dataset includes students’ grades in various subjects. The objective is to analyze this dataset using clustering algorithms. Specifically, you should apply:

- k-means clustering combined with the **elbow method** to determine the appropriate number of clusters.  
- Hierarchical clustering (agglomerative) and present a **dendrogram**. Based on the dendrogram, choose the appropriate number of clusters.  
- DBSCAN clustering. Choose suitable parameter values based on the **k-distance graph**.

For each case, comment on the parameter choices and the resulting clusters.  
**Note:** Only 3 features are numeric. If you want to use categorical features, you must convert them to numeric values meaningfully.

---

## Part C – Association Rule Mining

Apply the **Apriori algorithm** on the Student Performance dataset to discover association rules. Evaluate the rules based on the **Lift measure**. Present the rules discovered by the algorithm and comment on at least 10 rules that you find interesting.

**Important:** Apriori treats all data as categorical. Therefore, you must either ignore numeric features or discretize them.

---

## Datasets

The datasets are provided in **CSV formats**. In this GitHub project, they are located in the folder datasets

## Deliverables

- A technical report including presentation and commentary/interpretation of results. Screenshots, experimental measurements, parameter values, Python code snippets, or WEKA screenshots should be included.  
- The preprocessed datasets.  
- Python files.  

**Notes:**  
- You may use Python (scikit-learn) or WEKA (Python/scikit-learn is recommended).  
- Preprocessing and parameter selection are up to you.  
- This is an individual project; no in-class presentation is required.
