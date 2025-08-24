
import pandas as pd

# PART A
# Load the dataset
spam_df = pd.read_csv("spam.csv")
# Display first 5 rows
print(spam_df.head())
# Show unique values in 'class' column
print(spam_df['class'].unique())
print(spam_df['class'].value_counts())
# Fix typo 'emai' → 'email'
spam_df['class'] = spam_df['class'].replace('emai', 'email')
print(spam_df['class'].unique())
print(spam_df['class'].value_counts())
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = spam_df.drop('class', axis=1)
y = spam_df['class']

# Label Encoding (convert email/spam into 0/1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Check which class corresponds to 0 and which to 1
print(le.classes_)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Check dataset sizes
print("Train size:", len(X_train))
print("Test size:", len(X_test))

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# List of models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "k-NN": KNeighborsClassifier(n_neighbors=5)
}

# Training and evaluation
for name, model in models.items():
    print(f"\n=======================\nResults for {name}:")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Print performance report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# PART B
import pandas as pd

# Load dataset
df_students = pd.read_csv('StudentsPerformance.csv')

# Show first rows
print(df_students.head())

# Show data types and detect nulls
print(df_students.info())

# Show basic descriptions of numeric columns
print(df_students.describe())

#ELBOW METHOD k-means
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Keep only numeric attributes
X = df_students[['math score', 'reading score', 'writing score']]

# Elbow method
sse = []  # Sum of Squared Errors
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Plot Elbow graph
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of Clusters(k)')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method for Finding Optimal k')
plt.grid(True)
plt.show()

# Apply k-means with k = 2
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

# Add new column with cluster labels to the dataframe
df_students['cluster_kmeans'] = clusters

# Show number of students per cluster
print(df_students['cluster_kmeans'].value_counts() )

# Show averages
print(df_students.groupby('cluster_kmeans')[['math score', 'reading score', 'writing score']].mean())

# HIERARCHICAL CLUSTERING
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Plot dendrogram
linked = linkage(X, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('Dendrogram - Hierarchical Clustering')
plt.xlabel('Students (or small clusters)')
plt.ylabel('Merging distance')
plt.grid(True)
plt.show()

# Apply hierarchical clustering
agglo = AgglomerativeClustering(n_clusters=2, linkage='ward')
clusters_agglo = agglo.fit_predict(X)
df_students['cluster_agglo'] = clusters_agglo

# Show averages
print(df_students.groupby('cluster_agglo')[['math score', 'reading score', 'writing score']].mean())

# DBSCAN
from sklearn.cluster import DBSCAN
# Create k-distance
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Nearest Neighbors for k = 4 (min_samples = 4)
neigh = NearestNeighbors(n_neighbors=4)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

# Sort distances and plot
k_distances = np.sort(distances[:, 3])
plt.figure(figsize=(10, 5))
plt.plot(k_distances)
plt.title('k-distance graph (k = 4)')
plt.xlabel('Data Index')
plt.ylabel('Distance to 4th neighbor')
plt.grid(True)
plt.show()

# Print basic distance statistics
print(f"Min distance to {k}neighbor: {k_distances.min():.3f}")
print(f"Mean distance to {k}neighbor: {k_distances.mean():.3f}")
print(f"Max distance to {k}neighbor: {k_distances.max():.3f}")

# Select eps based on the graph
eps_value = 7

# Apply DBSCAN
dbscan = DBSCAN(eps=eps_value, min_samples=k)
db_clusters = dbscan.fit_predict(X)

# Add results to dataframe
df_students['cluster_dbscan'] = db_clusters

# Number of points per cluster (-1 = noise)
print("\nNumber of points per cluster (cluster):")
print(df_students['cluster_dbscan'].value_counts())

# Average scores per cluster (excluding noise)
print("\nAverage scores per cluster (without noise):")
print(df_students[df_students['cluster_dbscan'] != -1].groupby('cluster_dbscan')[['math score', 'reading score', 'writing score']].mean())

# PART B
import pandas as pd

# Load dataset
df_students = pd.read_csv('StudentsPerformance.csv')

# Show first rows
print(df_students.head())

# Show data types and detect nulls
print(df_students.info())

# Show basic descriptions of numeric columns
print(df_students.describe())

#ELBOW METHOD k-means
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Keep only numeric attributes
X = df_students[['math score', 'reading score', 'writing score']]

# Elbow method
sse = []  # Sum of Squared Errors
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Plot Elbow graph
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of Clusters(k)')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method for Finding Optimal k')
plt.grid(True)
plt.show()

# Apply k-means with k = 2
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

# Add new column with cluster labels to the dataframe
df_students['cluster_kmeans'] = clusters

# Show number of students per cluster
print(df_students['cluster_kmeans'].value_counts() )

# Show averages
print(df_students.groupby('cluster_kmeans')[['math score', 'reading score', 'writing score']].mean())

# HIERARCHICAL CLUSTERING
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Plot dendrogram
linked = linkage(X, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='level', p=5)
plt.title('Dendrogram - Hierarchical Clustering')
plt.xlabel('Students (or small clusters)')
plt.ylabel('Merging distance')
plt.grid(True)
plt.show()

# Apply hierarchical clustering
agglo = AgglomerativeClustering(n_clusters=2, linkage='ward')
clusters_agglo = agglo.fit_predict(X)
df_students['cluster_agglo'] = clusters_agglo

# Show averages
print(df_students.groupby('cluster_agglo')[['math score', 'reading score', 'writing score']].mean())

# DBSCAN
from sklearn.cluster import DBSCAN
# Create k-distance
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Nearest Neighbors for k = 4 (min_samples = 4)
neigh = NearestNeighbors(n_neighbors=4)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

# Sort distances and plot
k_distances = np.sort(distances[:, 3])
plt.figure(figsize=(10, 5))
plt.plot(k_distances)
plt.title('k-distance graph (k = 4)')
plt.xlabel('Data Index')
plt.ylabel('Distance to 4th neighbor')
plt.grid(True)
plt.show()

# Print basic distance statistics
print(f"Min distance to {k}neighbor: {k_distances.min():.3f}")
print(f"Mean distance to {k}neighbor: {k_distances.mean():.3f}")
print(f"Max distance to {k}neighbor: {k_distances.max():.3f}")

# Select eps based on the graph
eps_value = 7

# Apply DBSCAN
dbscan = DBSCAN(eps=eps_value, min_samples=k)
db_clusters = dbscan.fit_predict(X)

# Add results to dataframe
df_students['cluster_dbscan'] = db_clusters

# Number of points per cluster (-1 = noise)
print("\nNumber of points per cluster (cluster):")
print(df_students['cluster_dbscan'].value_counts())

# Average scores per cluster (excluding noise)
print("\nAverage scores per cluster (without noise):")
print(df_students[df_students['cluster_dbscan'] != -1].groupby('cluster_dbscan')[['math score', 'reading score', 'writing score']].mean())

# PART C
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Copy the original dataframe
df_discretized = df_students.copy()

# Define labels for discretization
score_labels = ['low', 'medium', 'high']

# Discretize the scores into 3 equal parts using qcut
df_discretized['math_level'] = pd.qcut(df_discretized['math score'], q=3, labels=score_labels)
df_discretized['reading_level'] = pd.qcut(df_discretized['reading score'], q=3, labels=score_labels)
df_discretized['writing_level'] = pd.qcut(df_discretized['writing score'], q=3, labels=score_labels)

# Select only categorical columns for analysis
categorical_data = df_discretized[['gender', 'race/ethnicity', 'parental level of education',
                                   'lunch', 'test preparation course',
                                   'math_level', 'reading_level', 'writing_level']]

# Convert rows to list of lists (transactions)
transactions = categorical_data.astype(str).values.tolist()

# Convert to one-hot encoded format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_apriori = pd.DataFrame(te_ary, columns=te.columns_)

# Extract frequent itemsets with minimum support of 5%
frequent_itemsets = apriori(df_apriori, min_support=0.05, use_colnames=True)

# Generate association rules with lift >= 1.0
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

# Sort rules by lift in descending order
rules = rules.sort_values(by='lift', ascending=False)

# Convert frozensets to strings for better readability
rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# Pandas display settings for better print
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 100)

# Display all rules
print(rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].to_string())

# Display top 10 rules
print("\nTop 10 rules:\n")
print(rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].head(10).to_string())
