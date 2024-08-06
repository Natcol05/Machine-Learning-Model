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
We had 14 characteristics and 4.000 observations. Churn is our target variable, and the other 13 are characteristics. It's important to add that we ddin't have missing values and all of them are numercial. 

We analyzed the correlation between the target variable and the characteristics separately for people who had canceled and those who hadn't, as you can see in the heatmap:

<p align="center">
  <img src="https://github.com/Natcol05/Machine-Learning-Model/blob/c7d3bdc37bff06ff0ecf992f7b1f7ed56d3fae9d/Graphics/Correlation_calceled_group.png" alt="Sample Image">
</p>

It is necessary to highlight that gender wasn't analyzed because we can't determine which numbers correspond to female and male, making it impossible to generate comments about it.

Following that, we can make some observations based on the data distribution:

* Clients who do not live near the gym have a higher chance of canceling their plan.
* Clients who work for an associated company have lower cancellation rates. However, the proportion of clients with this characteristic is only slightly higher than those without it, meaning the size of both groups is not significantly different.
* Clients who joined the gym through a promotional offer from a friend have lower cancellation chances, but this population isn't the largest.
* Clients with contracts shorter than 3 months have a higher likelihood of canceling their plan.
* There are more clients who do not participate in group classes, and this group has a higher cancellation rate.
* The age distribution does not provide particularly useful information.
* Cancellation rates increase for clients who are close to the end of their contract (within 2 months).
* The higher the number of visits by a client, the lower the chance of cancellation.

Regarding the correlation in the canceled group, 'month to end contract' and 'contract period' have a high correlation (0.98). Similarly, 'promo_friends' and 'partner' show a correlation of 0.38, while 'near_location' and 'promo_friends' have a correlation of 0.20.

However, in the case of the non-canceled group, we observe that 'partner' and 'promo_friends' have a higher correlation (0.46). Additionally, new correlations are found between 'partner' and 'contract period' (0.29), 'contract period' and 'promo_friends' (0.22), 'month to end contract' and 'partner' (0.28), and 'month to end contract' and 'promo_friends' (0.21).

Afther this we classified the clients in 5 clusters:

<p align="center">
  <img src="https://github.com/Natcol05/Machine-Learning-Model/blob/c7d3bdc37bff06ff0ecf992f7b1f7ed56d3fae9d/Graphics/Clusters.png" alt="Sample Image">
</p>

* **Remote Visitors:** clients who aren't near the location and have a moderate churn rate.
* **Occasional Visitors:** Members who don't provide phone numbers and have a moderate churn rate.
* **High Risk:** Members that have the highest churn rate, are near the location, have shorter contract periods, and lower class attendance.
* **Frequent Spenders:** Members who have high additional charges, frequent class attendance, and a low churn rate.
* **Loyal Long-Term:** Clients who are more committed (longer contract periods, higher lifetime, frequent class attendance) and have the lowest churn rate.

**General Conclusions:**

Regarding the exploratory data analysis, we did not encounter significant issues. In general, there were no missing data, and the dataset comprised numerical information. The categorical variables, such as phone, gender, and distance, were classified as 0 or 1, with 0 meaning "No" and 1 meaning "Yes."
Gender was not considered in the analysis because we lacked sufficient information to determine which number corresponded to female and male.
Now, regarding the conclusions, most clients are highly committed, meaning they have long contract periods, a high lifetime value, and frequent attendance, coupled with a low churn rate. This is a very positive indicator for the gym.
However, we have a significant group of clients who live far away from the gym. Although being near the location doesn’t necessarily guarantee lower churn rates, motivating people who live far away can be useful in reducing cancellations. In this case, a good marketing strategy is essential for proper outreach. Perhaps creating a promotion scheme based on the distance from the gym could be attractive. This promotion could consider the cost of public transportation to determine the discount amount.
Another strategy that could work is offering promotions and benefits for people who sign long-term contracts from the beginning. For instance, offering a lower fee for the first year or access to exclusive gym services for those with a year-long contract could be effective.
Finally, there is a high correlation between attendance at group activities and low churn rates. Therefore, effective marketing strategies are also crucial here. Additionally, considering the schedules of people who visit the gym and the timings of the classes could be useful. It might be helpful to analyze the number of people who visit the gym during hours when no classes are offered and see if adjustments can be made.




