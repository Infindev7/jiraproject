# # Statistics Practical — Question Bank Solutions (Experiments 2–9)
# All 15 questions from `Stats_Practical_Exam_QB.pdf`, solved and executed.
# Methods follow your Experiment 2–9 notebooks (same constructions, same parameters: random_state=42, test_size=0.2, stratify, n_estimators=200, n_init=10, n_repeats=10, alpha=1.0). Datasets come from scikit-learn only, as the question bank specifies — loaded with `pd.DataFrame(data, columns=feature_names)`, the same DataFrame-first style as your notebooks.
# The only exceptions are methods the question itself forces (flagged FORCED EXCEPTION in the Answer lines): there is no experiment-native version of those, because your notebooks never covered them.

# ## Q1. Iris Dataset — Exploratory Data Analysis
# *Experiment 2*
#
# Using the Iris dataset, calculate the mean, median and standard deviation of the four numerical features and generate a box plot for one selected feature.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris: features into a DataFrame, target kept aside
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Descriptive statistics, Experiment 2 pattern
stats = pd.DataFrame({"Mean": X.mean(), "Median": X.median(), "Std": X.std()})
print(stats.round(2))

# Box plot for one selected feature: sepal length
sns.boxplot(y=X["sepal length (cm)"])
plt.title("Box Plot - Sepal Length (cm)")
plt.tight_layout()
plt.show()

# **Answer:** Mean sepal length 5.84, median 5.80, std 0.83. Same DataFrame construction as your Experiment 2 notebook, restricted to the three asked statistics.

# ## Q2. Diabetes Dataset — Distribution Analysis
# *Experiment 2*
#
# Using load_diabetes() from scikit-learn, calculate the descriptive statistics of the BMI feature and generate a histogram to examine its distribution.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
bmi = df["bmi"]

# Descriptive statistics of the BMI feature
print("Mean:", round(bmi.mean(), 2))
print("Median:", round(bmi.median(), 2))
print("Std:", round(bmi.std(), 2))

# Histogram, Experiment 2 pattern
plt.hist(bmi, bins=25)
plt.title("Distribution of BMI")
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# **Answer:** Mean ≈ 0.00, median −0.01, std 0.05. NOTE: sklearn's diabetes features are already standardized, so these values are correct, not a mistake.

# ## Q3. Iris Dataset — Correlation Analysis
# *Experiment 3*
#
# Calculate the Pearson correlation matrix of the numerical features in the Iris dataset and visualize it using a heatmap. Identify the feature pair with the strongest absolute correlation.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Pearson correlation matrix
corr = X.corr()
print(corr.round(2))

# Heatmap, Experiment 3 pattern
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix - Iris Features")
plt.tight_layout()
plt.show()

# Strongest absolute correlation, Experiment 3 pattern
s = corr.abs().unstack().sort_values(ascending=False)
print(s[s < 1].head(2))

# **Answer:** Strongest pair: petal length vs petal width, r = 0.963 (shown twice, once in each order, exactly as your Experiment 3 notebook prints it).

# ## Q4. Distance Measures and Scaling
# *Experiment 3*
#
# Create a small dataset containing at least four observations and two numerical features with different scales. Apply standardization and calculate the Euclidean distance between two observations before and after standardization.

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Small dataset: income (large scale) vs rating (small scale)
X = np.array([[1000, 2.0],
              [1500, 3.0],
              [800, 1.5],
              [1200, 2.5]])

# Euclidean distance between observation 0 and 1, BEFORE scaling
d_raw = euclidean_distances([X[0]], [X[1]])[0][0]
print(f"Before scaling: {d_raw:.3f}")

# Standardize, Experiment 3 pattern, then distance again
Xs = StandardScaler().fit_transform(X)
d_std = euclidean_distances([Xs[0]], [Xs[1]])[0][0]
print(f"After scaling:  {d_std:.3f}")

# **Answer:** Before scaling 500.001 (income dominates completely); after scaling 2.634. Standardization makes the distance meaningful.

# ## Q5. One-Sample t-Test
# *Experiment 4*
#
# Consider the observations 52, 48, 51, 49, 54, 50, 53, 47. Perform a one-sample t-test to determine whether the population mean differs significantly from 50 at the 5% significance level.

import numpy as np
from scipy import stats

data = np.array([52, 48, 51, 49, 54, 50, 53, 47])
print(f"Sample mean = {data.mean():.3f}, n = {len(data)}")

# H0: population mean = 50. One sample vs a fixed value -> one-sample test
t_stat, p_val = stats.ttest_1samp(data, 50)
print(f"t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
print("Decision:", "Reject H0" if p_val < 0.05 else "Fail to reject H0")

# **Answer:** t = 0.577, p = 0.5818 > 0.05 → fail to reject H0. FORCED EXCEPTION: your Experiment 4 used a two-sample test (ttest_ind); this question gives one sample against 50, so the one-sample version from the same scipy.stats family is the only correct method.

# ## Q6. Bootstrap Confidence Interval
# *Experiment 5*
#
# Using the dataset 12, 15, 14, 10, 18, 16, 13, 17, 11, 14, implement bootstrap resampling to estimate a 95% confidence interval for the sample mean.

import numpy as np

data = np.array([12, 15, 14, 10, 18, 16, 13, 17, 11, 14], dtype=float)
rng = np.random.default_rng(42)

# 2000 resamples WITH replacement, Experiment 5 pattern
boots = np.array([rng.choice(data, size=len(data), replace=True).mean()
                  for _ in range(2000)])

# 95% CI = middle 95% of bootstrap means
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"Sample mean = {data.mean():.2f}")
print(f"Bootstrap 95% CI: ({lo:.2f}, {hi:.2f})")

# **Answer:** Sample mean 14.00, bootstrap 95% CI (12.50, 15.50). No normality assumption needed.

# ## Q7. Diabetes Regression
# *Experiment 6*
#
# Using load_diabetes() from scikit-learn, build a linear regression model using the BMI feature to predict the target. Evaluate the model using Mean Absolute Error (MAE) and R2.

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
X = df[["bmi"]].values
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print(f"MAE = {mean_absolute_error(y_test, pred):.2f}")
print(f"R2  = {r2_score(y_test, pred):.3f}")

# **Answer:** MAE 52.26, R² 0.233. FORCED EXCEPTION: your Experiment 6 used RMSE; this question explicitly asks MAE, so mean_absolute_error replaces it — same model, same split, only the metric changes.

# ## Q8. Regression Model Comparison
# *Experiment 6*
#
# Using the Diabetes dataset, train Linear Regression and Ridge Regression models using the same train-test split. Compare their test-set R2 scores.

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
X = df.values
y = diabetes.target

# SAME split for both models, so the comparison is fair
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model loop, Experiment 6 pattern
for name, model in [("LinearRegression", LinearRegression()),
                    ("Ridge", Ridge(alpha=1.0))]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name} test R2 = {r2_score(y_test, pred):.3f}")

# **Answer:** Linear test R² 0.453 beats Ridge test R² 0.419 on this split. (Your Experiment 6 notebook itself loops over models this way.)

# ## Q9. Iris Classification — Decision Tree
# *Experiment 7*
#
# Using the Iris dataset, train a Decision Tree classifier to predict the species. Evaluate its performance using a confusion matrix, accuracy, precision, recall and F1-score.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names).values
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Accuracy  = {accuracy_score(y_test, pred):.3f}")
print(f"Precision = {precision_score(y_test, pred, average='weighted'):.3f}")
print(f"Recall    = {recall_score(y_test, pred, average='weighted'):.3f}")
print(f"F1-score  = {f1_score(y_test, pred, average='weighted'):.3f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))

# **Answer:** Accuracy 0.933, precision/recall/F1 0.933. Confusion matrix [[10,0,0],[0,9,1],[0,1,9]]. FORCED EXCEPTIONS: the question names Decision Tree (your Experiment 7 used Logistic Regression and Random Forest), and 3-class Iris forces average='weighted' — without it the metric calls error, since your notebook's binary code had no averaging to do.

# ## Q10. Classification Performance Measures — Random Forest
# *Experiment 7*
#
# Using the Iris dataset, train a Random Forest classifier to predict the species. Evaluate its performance using a confusion matrix, accuracy, precision, recall and F1-score.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names).values
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Accuracy  = {accuracy_score(y_test, pred):.3f}")
print(f"Precision = {precision_score(y_test, pred, average='weighted'):.3f}")
print(f"Recall    = {recall_score(y_test, pred, average='weighted'):.3f}")
print(f"F1-score  = {f1_score(y_test, pred, average='weighted'):.3f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))

# **Answer:** Accuracy 0.900, precision 0.902, recall 0.900, F1 0.900. Confusion matrix [[10,0,0],[0,9,1],[0,2,8]]. Same Experiment 7 pattern (n_estimators=200, stratify); average='weighted' is forced by the 3 classes, as in Q9.

# ## Q11. K-Means Clustering
# *Experiment 8*
#
# Using the numerical features of the Iris dataset, apply K-Means clustering with three clusters. Calculate the Silhouette Score to evaluate the clustering result.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names).values

# Scale first, Experiment 8 pattern: K-Means is distance-based
Xs = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xs)

print(f"Silhouette Score = {silhouette_score(Xs, km.labels_):.3f}")

# **Answer:** Silhouette Score 0.460. FORCED EXCEPTION: your Experiment 8 evaluated clustering with the elbow method and a crosstab; this question explicitly asks the Silhouette Score, so silhouette_score replaces the elbow check — scaling and KMeans steps are unchanged from your notebook.

# ## Q12. PCA Visualization
# *Experiment 8*
#
# Apply PCA to the standardized numerical features of the Iris dataset and reduce them to two principal components. Generate a two-dimensional scatter plot of the transformed observations, using the actual species labels to distinguish the points.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names).values
y = iris.target

Xs = StandardScaler().fit_transform(X)
Z = PCA(n_components=2, random_state=42).fit_transform(Xs)

# Scatter of the 2 components, coloured by species label, Experiment 8 pattern
plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis")
plt.colorbar()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Iris Dataset (2 components)")
plt.tight_layout()
plt.show()

# **Answer:** Setosa separates cleanly; versicolor and virginica overlap. Colour scale: 0 = setosa, 1 = versicolor, 2 = virginica.

# ## Q13. Cross-Validation
# *Experiment 9*
#
# Using the Breast Cancer Wisconsin dataset available through scikit-learn, train a Logistic Regression classifier and evaluate it using 5-fold cross-validation. Report the individual fold scores and mean accuracy.

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

bc = load_breast_cancer()
X = pd.DataFrame(bc.data, columns=bc.feature_names).values
y = bc.target

scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
print("Fold scores:", np.round(scores, 3))
print(f"Mean accuracy = {scores.mean():.3f}")

# **Answer:** Fold scores [0.930, 0.939, 0.974, 0.947, 0.965], mean accuracy 0.951. Same cross_val_score pattern as your Experiment 9; max_iter=1000 is your Experiment 7's Logistic Regression setting.

# ## Q14. Feature Importance
# *Experiment 9*
#
# Using the Breast Cancer Wisconsin dataset, train a Decision Tree classifier and display a bar chart of its top five feature importances.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

bc = load_breast_cancer()
X = pd.DataFrame(bc.data, columns=bc.feature_names).values
y = bc.target
names = bc.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

# A tree stores its own importances; top 5 as a bar chart
imp = pd.Series(model.feature_importances_, index=names)
top5 = imp.sort_values(ascending=False).head(5)
print(top5.round(3))
top5.plot(kind="barh")
plt.title("Top 5 Feature Importances - Decision Tree")
plt.tight_layout()
plt.show()

# **Answer:** Top feature by far: worst radius (0.698), then worst concave points (0.116), worst texture (0.059), texture error (0.053), worst concavity (0.016). FORCED EXCEPTION: your Experiment 9 measured importance by permutation (that's Q15); for a single tree, the question's 'feature importances' are the tree's own feature_importances_ attribute.

# ## Q15. Model Explainability using Permutation Importance
# *Experiment 9*
#
# Using the Breast Cancer Wisconsin dataset, train a Random Forest classifier and calculate permutation feature importance on the test set. Display the top five features according to their importance.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

bc = load_breast_cancer()
X = pd.DataFrame(bc.data, columns=bc.feature_names).values
y = bc.target
names = bc.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Shuffle each feature in turn; importance = accuracy drop, Experiment 9 pattern
pi = permutation_importance(model, X_test, y_test,
                            n_repeats=10, random_state=42)
imp = pd.Series(pi.importances_mean, index=names)
top5 = imp.sort_values(ascending=False).head(5)
print(top5.round(3))
top5.plot(kind="barh")
plt.title("Top 5 Permutation Importances - Random Forest")
plt.tight_layout()
plt.show()

# **Answer:** Top 5: mean area, compactness error, worst perimeter, worst area, mean concave points (all ≈ 0.008–0.009 — the forest spreads importance, unlike the single tree in Q14).