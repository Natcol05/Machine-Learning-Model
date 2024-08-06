<p align="center"><b>Predicting and Addressing Customer Churn: The case of Model Fitness</b></p>

The aim of this project is to design a machine learning model for a gym named Model Fitness to predict churn probabilities based on key features such as location distance, contract length, and attendance to group classes. 
For the model, logistic regression was used as it demonstrated better performance in accuracy, precision, and recall compared to the Random Forest Classifier. Additionally, a k-means model was applied to classify clients into five clusters: Frequent Spenders, High Risk, Loyal Long Term, Occasional Visitors, and Remote Visitors.

These are some of the libraries used in this project:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
```
