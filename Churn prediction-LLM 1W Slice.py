from google.colab import drive
drive.mount('/content/drive')
import urllib
from pathlib import Path
import sqlalchemy
import pandas as pd
pd.options.display.max_rows = 15
pd.options.display.max_columns = 500
from tqdm import tqdm
tqdm.pandas()
import numpy as np
import time
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_palette(sns.color_palette())
from math import inf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, linear_model
import scipy.stats as stat
from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from IPython.display import display as IPD
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
driver_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data112023.csv')

driver_data_dt = ['week', 'activated_date', 'ftt_time']
for i in driver_data_dt:
    driver_data[i] = pd.to_datetime(driver_data[i])
del driver_data['activated_date']
del driver_data['ftt_time']

driver_data['vehicle_model'] = driver_data['vehicle_model'].fillna('None')
driver_data = driver_data.fillna(0)
driver_data # 761102 rows
driver_date = driver_data.groupby('driver_id')['week'].min().reset_index()
driver_date.columns = ['driver_id', 'first_week']
driver_sparse = driver_data[['driver_id', 'ops_city_id', 'device_type',
                             'vehicle_brand', 'vehicle_model']].drop_duplicates()
driver_sparse = driver_data[['week']].drop_duplicates().merge(driver_sparse,
                                                              how = 'cross')
driver_data = driver_sparse.merge(driver_data, how = 'left',
                                  on = list(driver_sparse.columns))
del driver_sparse
driver_data = driver_data.merge(driver_date, how = 'left',
                                on = 'driver_id').query('week >= first_week')
del driver_date
del driver_data['first_week']
driver_data['segment'] = driver_data['segment'].fillna('Parttime')
driver_data = driver_data.fillna(0).reset_index(drop = True)
driver_data # 986830 rows
driver_data = driver_data.sort_values(['week', 'driver_id']).reset_index(drop = True)
driver_data_self = driver_data[['week', 'driver_id', 'not_churn']]
driver_data_self = driver_data_self.merge(driver_data_self, how = 'left', on = 'driver_id', suffixes = ['', '_1w'])
driver_data_self = driver_data_self[driver_data_self.week_1w == driver_data_self.week + pd.Timedelta(days = 7)].reset_index(drop = True)
del driver_data_self['not_churn']
del driver_data_self['week_1w']
driver_data_self
driver_data_self = driver_data[['week', 'driver_id', 'not_churn']].merge(driver_data_self, how = 'left', on = ['week', 'driver_id'])
# Since latest week have no future information, so we drop it
latest_week = driver_data_self['week'].max()
driver_data_self = driver_data_self[driver_data_self['week'] != latest_week].reset_index(drop = True)
# Now any null value of column 'not_churn_1w' means that the driver disappear (not online) -> churn
driver_data_self['not_churn_1w'] = driver_data_self['not_churn_1w'].fillna(0)
driver_data_self[driver_data_self['not_churn'] == 1]['not_churn_1w'].mean()
driver_data = driver_data[driver_data['week'] != latest_week].reset_index(drop = True)
driver_data['not_churn_1w'] = driver_data_self['not_churn_1w']
driver_data
# Split Train - Test
latest_week = driver_data['week'].max()
driver_data_test = driver_data[driver_data['week'] == latest_week].reset_index(drop = True)
driver_data_train = driver_data[driver_data['week'] != latest_week].reset_index(drop = True)
IPD(driver_data_train)
IPD(driver_data_test)
driver_data.describe()
driver_data.columns.values
driver_data.dtypes
X = driver_data_train[['ops_city_id', 'device_type', 'sum_sh', 'completed_trip',
       'total_dispatched_request', 'Valid_dispatch', 'accepted',
       'cancelled', 'rider_cancel', 'driver_cancel', 'total_fare',
       'gmv_aftertax', 'Trips_without_surge', 'Trips_with_surge',
       'GB_without_surge', 'Surge_Organic_GB', 'Surge_Contri_GB',
       'total_ETA', 'total_ATA', 'trip_time_hour', 'en_route_hour',
       'distance_travelled', 'driver_trip_amount', 'weekly_amount',
       'loyalty_amount', 'not_churn']].values
k = []
for i in range (1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    k.append(kmeans.inertia_)
    
plt.plot(range(1, 11), k)
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("k")
plt.show()
kmeans = KMeans(n_clusters = 3, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
driver_data_train['Cluster'] = kmeans.fit_predict(X)
driver_data_train
driver_data_train['Cluster'].unique()
train_0 = driver_data_train[driver_data_train['Cluster'] == 0].reset_index(drop = True)
train_0
def woe_discrete(df_input, variable_name):
    df = df_input[[variable_name, 'not_churn_1w']]
    df = pd.concat([df.groupby(variable_name, as_index = False)['not_churn_1w'].count(),
    df.groupby(variable_name, as_index = False)['not_churn_1w'].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [variable_name, 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE']).reset_index(drop = True)
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    if (df['IV'][0] == -inf) or (df['IV'][0] == inf):
        extreme = set(df[(df['WoE'] > -inf) & (df['WoE'] < inf)][variable_name])
        df_input_new = df_input[df_input[variable_name].isin(extreme)]
        df['IV_adjusted'] = woe_discrete(df_input_new, variable_name)['IV'][0]
    else:
        df['IV_adjusted'] = df['IV']
    return df

def woe_continuous(df_input, variable_name, class_count = 50):
    df = df_input
    fine_classing_variable = variable_name + '_fine_classing'
    df[fine_classing_variable] = pd.cut(df[variable_name], class_count)
    df = woe_discrete(df, fine_classing_variable)
    df = df.sort_values([fine_classing_variable])
    df = df.reset_index(drop = True)
    return df

def woe_continuous_best_class_count(df_input, variable_name, class_count_bounds = [3, 20]):
    candidate_class_counts = list(range(class_count_bounds[0], class_count_bounds[1]))
    IV_adjusted_list = [woe_continuous(df_input, variable_name, class_count)['IV_adjusted'][0] for class_count in candidate_class_counts]
    best_cut = candidate_class_counts[IV_adjusted_list.index(max(IV_adjusted_list))]
    return [best_cut, woe_continuous(df_input, variable_name, best_cut)]

def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 0, figsize = (10, 5)):
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize = figsize)
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    var_name = df_WoE.columns[0]
    var_IV = df_WoE['IV_adjusted'][0]
    plt.ylabel('Weight of Evidence')
    plt.title('Weight of Evidence by ' + var_name + ' (IV = ' + str(round(var_IV, 5)) + ')')
    plt.xticks(rotation = rotation_of_x_axis_labels)
    plt.show()
discrete_cols = ['ops_city_id', 'device_type', 'vehicle_model',
                  'vehicle_brand', 'segment']
df_discrete = pd.DataFrame(discrete_cols).rename(columns = {0:'DISCRETE_COL'})
df_discrete['DISTINCT'], df_discrete['IV'], df_discrete['IV_ADJUSTED'] = '', '', ''
for i in range(len(df_discrete)):
    col = df_discrete['DISCRETE_COL'][i]
    results = woe_discrete(train_0, col)
    df_discrete['DISTINCT'][i] = len(results)
    df_discrete['IV'][i] = results['IV'][0]
    df_discrete['IV_ADJUSTED'][i] = results['IV_adjusted'][0]
    print(col, 'done!')
df_discrete = df_discrete.sort_values(['IV_ADJUSTED', 'IV', 'DISTINCT'], ascending = [False, True, True]).reset_index(drop = True)
df_discrete
selected_discrete_cols = ['segment']
plot_by_woe(woe_discrete(train_0, 'segment'))
woe_discrete(train_0, 'segment')
train_0.columns
pd.options.display.max_rows = None
continuous_cols = ['sum_sh', 'completed_trip',
                   'total_dispatched_request', 'Valid_dispatch',
                   'accepted', 'cancelled', 'rider_cancel', 'driver_cancel',
                   'total_fare', 'gmv_aftertax', 'Trips_without_surge',
                   'Trips_with_surge', 'GB_without_surge', 'Surge_Organic_GB',
                   'Surge_Contri_GB', 'total_ETA', 'total_ATA', 'trip_time_hour',
                   'en_route_hour', 'distance_travelled', 'driver_trip_amount',
                   'weekly_amount', 'loyalty_amount']

df_continuous = pd.DataFrame(continuous_cols).rename(columns = {0:'CONTINUOUS_COL'})
df_continuous['BEST_CUT'], df_continuous['IV'], df_continuous['IV_ADJUSTED'] = '', '', ''
for i in range(len(df_continuous)):
    col = df_continuous['CONTINUOUS_COL'][i]
    results = woe_continuous_best_class_count(train_0, col)
    df_continuous['BEST_CUT'][i] = results[0]
    df_continuous['IV'][i] = results[1]['IV'][0]
    df_continuous['IV_ADJUSTED'][i] = results[1]['IV_adjusted'][0]
    print(col, 'done!')
df_continuous = df_continuous.sort_values(['IV_ADJUSTED', 'IV', 'BEST_CUT'], ascending = [False, True, True]).reset_index(drop = True)
df_continuous
woe_continuous_best_class_count(train_0, 'driver_cancel')[1]
plot_by_woe(woe_continuous_best_class_count(train_0, 'completed_trip')[1], 90)
selected_continuous_cols = ['completed_trip', 'total_fare', 'total_ETA', 'distance_travelled' ]
selected_continuous_cols = [x + '_discrete' for x in selected_continuous_cols]
selected_continuous_cols
def completed_trip_discrete(completed_trip):
    if completed_trip <= 7:
        return '0 - 7'
    if completed_trip <= 14:
        return '7 - 14'
    if completed_trip <= 28:
        return '14 - 28'
    if completed_trip <= 50:
        return '28 - 50'
    if completed_trip <= 100:
        return '50 - 100'
    return '>100'

train_0['completed_trip_discrete'] = train_0['completed_trip'].apply(lambda x: completed_trip_discrete(x))
driver_data_test['completed_trip_discrete'] = driver_data_test['completed_trip'].apply(lambda x: completed_trip_discrete(x))
woe_discrete(train_0, 'completed_trip_discrete')
def total_fare_discrete(total_fare):
    if total_fare <= 100000:
        return '< 100k'
    if total_fare <= 1000000:
        return '100k - 1 mil'
    if total_fare <= 5000000:
        return '1 - 5 mil'
    return '> 5 mil'

train_0['total_fare_discrete'] = train_0['total_fare'].apply(lambda x: total_fare_discrete(x))
driver_data_test['total_fare_discrete'] = driver_data_test['total_fare'].apply(lambda x: total_fare_discrete(x))
woe_discrete(train_0, 'total_fare_discrete')
def total_ETA_discrete(total_ETA):
    if total_ETA <= 1:
        return '< 1 min'
    if total_ETA <= 60:
        return '1 - 60 mins'
    if total_ETA <= 480:
        return '60 - 480 mins'
    return '> 480'

train_0['total_ETA_discrete'] = train_0['total_ETA'].apply(lambda x: total_ETA_discrete(x))
driver_data_test['total_ETA_discrete'] = driver_data_test['total_ETA'].apply(lambda x: total_ETA_discrete(x))
woe_discrete(train_0, 'total_ETA_discrete')
def distance_travelled_discrete(distance_travelled):
    if distance_travelled < 1:
        return '< 1 km'
    if distance_travelled <= 10:
        return '1 - 10km'
    if distance_travelled <= 50:
        return '10 - 50km'
    if distance_travelled <= 100:
        return '50 - 100km'
    if distance_travelled <= 1000:
        return '100 - 1000km'
    return '> 1000km'
train_0['distance_travelled_discrete'] = train_0['distance_travelled'].apply(lambda x: distance_travelled_discrete(x))
driver_data_test['distance_travelled_discrete'] = driver_data_test['distance_travelled'].apply(lambda x: distance_travelled_discrete(x))
woe_discrete(train_0, 'distance_travelled_discrete')
selected_cols = []
selected_cols.extend(selected_discrete_cols)
selected_cols.extend(selected_continuous_cols)
selected_cols
pd.options.display.max_rows = 15
list_dummies = [pd.get_dummies(train_0[col], prefix = col, prefix_sep = ':').iloc[:, :-1] for col in selected_cols]
train_data_X = pd.concat(list_dummies, axis = 1)
train_data_X
train_data_Y = train_0['not_churn_1w']
train_data_Y
# Class to display p-values for logistic regression in sklearn.

class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.z_scores = z_scores
        self.p_values = p_values
        self.sigma_estimates = sigma_estimates
        self.F_ij = F_ij
pd.options.display.max_rows = None
reg_0 = LogisticRegression_with_p_values()
reg_0.fit(train_data_X, train_data_Y)
summary_table = pd.DataFrame(columns = ['Feature name'], data = train_data_X.columns.values)
summary_table['Coefficients'] = np.transpose(reg_0.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg_0.intercept_[0]]
summary_table = summary_table.sort_index()
p_values = reg_0.p_values
p_values = np.append(np.nan, np.array(p_values))
summary_table['p_values'] = p_values
summary_table
summary_table[summary_table['p_values'] < 0.05]
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_train_proba = reg_0.model.predict_proba(train_data_X)[: ][: , 1]
Y_hat_train_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([train_data_Y, pd.DataFrame(Y_hat_test_proba)], axis = 1)
df_actual_predicted.columns = ['Y_train', 'Y_hat_train_proba']
tr = 0.5
df_actual_predicted['Y_hat_train'] = np.where(df_actual_predicted['Y_hat_train_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_train'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_train'], df_actual_predicted['Y_hat_train']))
print(classification_report(df_actual_predicted['Y_train'], df_actual_predicted['Y_hat_train']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_train'], df_actual_predicted['Y_hat_train_proba'])
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC = roc_auc_score(df_actual_predicted['Y_train'], df_actual_predicted['Y_hat_train_proba'])
AUROC
df_actual_predicted = df_actual_predicted.sort_values('Y_hat_train_proba').reset_index(drop = True)
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Y_train'].cumsum()
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Good']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / (df_actual_predicted.shape[0])
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / df_actual_predicted['Y_train'].sum()
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / (df_actual_predicted.shape[0] - df_actual_predicted['Y_train'].sum())
df_actual_predicted
# Plot Gini
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'], (1 - df_actual_predicted['Y_hat_train']).cumsum() / sum(1 - df_actual_predicted['Y_hat_train']), linestyle = '--', color = 'k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
# Here we calculate Gini from AUROC.
Gini = AUROC * 2 - 1
Gini
# Plot KS
plt.plot(df_actual_predicted['Y_hat_train_proba'], df_actual_predicted['Cumulative Perc Bad'], color = 'r')
plt.plot(df_actual_predicted['Y_hat_train_proba'], df_actual_predicted['Cumulative Perc Good'], color = 'b')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted['Cumulative Perc Bad'] - df_actual_predicted['Cumulative Perc Good'])
KS
for x in list(np.linspace(0, 1, 11)):
    print('Threshold:', x)
    df_actual_predicted['Y_hat_train'] = np.where(df_actual_predicted['Y_hat_train_proba'] > x, 1, 0)
    print(classification_report(df_actual_predicted['Y_train'], df_actual_predicted['Y_hat_train']))
pd.options.display.max_rows = 15
list_dummies = [pd.get_dummies(driver_data_test[col], prefix = col, prefix_sep = ':').iloc[:, :-1] for col in selected_cols]
test_data_X = pd.concat(list_dummies, axis = 1)
test_data_Y = driver_data_test['not_churn_1w']
IPD(test_data_X)
IPD(test_data_Y)
test_data_X.columns
train_data_X.columns
for col in [x for x in test_data_X.columns if x not in train_data_X.columns]:
    del test_data_X[col]
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = reg_0.model.predict_proba(test_data_X)[: ][: , 1]
Y_hat_test_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([test_data_Y, pd.DataFrame(Y_hat_test_proba)], axis = 1)
df_actual_predicted.columns = ['Y_test', 'Y_hat_test_proba']
tr = 0.5
df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_test'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC = roc_auc_score(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
AUROC
df_actual_predicted = df_actual_predicted.sort_values('Y_hat_test_proba').reset_index(drop = True)
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Y_test'].cumsum()
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Good']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / (df_actual_predicted.shape[0])
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / df_actual_predicted['Y_test'].sum()
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / (df_actual_predicted.shape[0] - df_actual_predicted['Y_test'].sum())
df_actual_predicted
# Plot Gini
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'], (1 - df_actual_predicted['Y_hat_test']).cumsum() / sum(1 - df_actual_predicted['Y_hat_test']), linestyle = '--', color = 'k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
# Here we calculate Gini from AUROC.
Gini = AUROC * 2 - 1
Gini
# Plot KS
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Bad'], color = 'r')
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Good'], color = 'b')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted['Cumulative Perc Bad'] - df_actual_predicted['Cumulative Perc Good'])
KS
for x in list(np.linspace(0, 1, 11)):
    print('Threshold:', x)
    df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > x, 1, 0)
    print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Population Stability Index
train_churn_perc = sum(1 - train_data_Y) / len(train_data_Y)
test_churn_perc = sum(1 - test_data_Y) / len(test_data_Y)
PSI_churn = (test_churn_perc - train_churn_perc) * np.log(test_churn_perc / train_churn_perc)
PSI_not_churn = (train_churn_perc - test_churn_perc) * np.log((1 - test_churn_perc) / (1 - train_churn_perc))
[PSI_churn, PSI_not_churn]
rf_model_0 = RandomForestClassifier(n_estimators = 100)
rf_model_0.fit(train_data_X, train_data_Y)

feature_imp = pd.Series(rf_model_0.feature_importances_,
                        index = train_data_X.columns)
feature_imp = feature_imp.sort_values(ascending = False)
feature_imp
sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = rf_model_0.predict_proba(train_data_X)[: ][: , 1]
Y_hat_test_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([train_data_Y, pd.DataFrame(Y_hat_test_proba)], axis = 1)
df_actual_predicted.columns = ['Y_test', 'Y_hat_test_proba']
tr = 0.5
df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_test'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC = roc_auc_score(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
AUROC
df_actual_predicted = df_actual_predicted.sort_values('Y_hat_test_proba').reset_index(drop = True)
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Y_test'].cumsum()
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Good']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / (df_actual_predicted.shape[0])
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / df_actual_predicted['Y_test'].sum()
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / (df_actual_predicted.shape[0] - df_actual_predicted['Y_test'].sum())
df_actual_predicted
# Plot Gini
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'], (1 - df_actual_predicted['Y_hat_test']).cumsum() / sum(1 - df_actual_predicted['Y_hat_test']), linestyle = '--', color = 'k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
# Here we calculate Gini from AUROC.
Gini = AUROC * 2 - 1
Gini
# Plot KS
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Bad'], color = 'r')
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Good'], color = 'b')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted['Cumulative Perc Bad'] - df_actual_predicted['Cumulative Perc Good'])
KS
for x in list(np.linspace(0, 1, 11)):
    print('Threshold:', x)
    df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > x, 1, 0)
    print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = rf_model_0.predict_proba(test_data_X)[: ][: , 1]
Y_hat_test_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([test_data_Y, pd.DataFrame(Y_hat_test_proba)], axis = 1)
df_actual_predicted.columns = ['Y_test', 'Y_hat_test_proba']
tr = 0.5
df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_test'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC = roc_auc_score(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
AUROC
df_actual_predicted = df_actual_predicted.sort_values('Y_hat_test_proba').reset_index(drop = True)
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Y_test'].cumsum()
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Good']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / (df_actual_predicted.shape[0])
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / df_actual_predicted['Y_test'].sum()
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / (df_actual_predicted.shape[0] - df_actual_predicted['Y_test'].sum())
df_actual_predicted
# Plot Gini
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'], (1 - df_actual_predicted['Y_hat_test']).cumsum() / sum(1 - df_actual_predicted['Y_hat_test']), linestyle = '--', color = 'k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
# Here we calculate Gini from AUROC.
Gini = AUROC * 2 - 1
Gini
# Plot KS
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Bad'], color = 'r')
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Good'], color = 'b')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted['Cumulative Perc Bad'] - df_actual_predicted['Cumulative Perc Good'])
KS
for x in list(np.linspace(0, 1, 11)):
    print('Threshold:', x)
    df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > x, 1, 0)
    print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
train_1 = driver_data_train[driver_data_train['Cluster'] == 1].reset_index(drop = True)
train_1
discrete_cols = ['ops_city_id', 'device_type', 'vehicle_model',
                  'vehicle_brand', 'segment']
df_discrete = pd.DataFrame(discrete_cols).rename(columns = {0:'DISCRETE_COL'})
df_discrete['DISTINCT'], df_discrete['IV'], df_discrete['IV_ADJUSTED'] = '', '', ''
for i in range(len(df_discrete)):
    col = df_discrete['DISCRETE_COL'][i]
    results = woe_discrete(train_1, col)
    df_discrete['DISTINCT'][i] = len(results)
    df_discrete['IV'][i] = results['IV'][0]
    df_discrete['IV_ADJUSTED'][i] = results['IV_adjusted'][0]
    print(col, 'done!')
df_discrete = df_discrete.sort_values(['IV_ADJUSTED', 'IV', 'DISTINCT'], ascending = [False, True, True]).reset_index(drop = True)
df_discrete
selected_discrete_cols = []
pd.options.display.max_rows = None
continuous_cols = ['sum_sh', 'completed_trip',
                   'total_dispatched_request', 'Valid_dispatch',
                   'accepted', 'cancelled', 'rider_cancel', 'driver_cancel',
                   'total_fare', 'gmv_aftertax', 'Trips_without_surge',
                   'Trips_with_surge', 'GB_without_surge', 'Surge_Organic_GB',
                   'Surge_Contri_GB', 'total_ETA', 'total_ATA', 'trip_time_hour',
                   'en_route_hour', 'distance_travelled', 'driver_trip_amount',
                   'weekly_amount', 'loyalty_amount']

df_continuous = pd.DataFrame(continuous_cols).rename(columns = {0:'CONTINUOUS_COL'})
df_continuous['BEST_CUT'], df_continuous['IV'], df_continuous['IV_ADJUSTED'] = '', '', ''
for i in range(len(df_continuous)):
    col = df_continuous['CONTINUOUS_COL'][i]
    results = woe_continuous_best_class_count(train_1, col)
    df_continuous['BEST_CUT'][i] = results[0]
    df_continuous['IV'][i] = results[1]['IV'][0]
    df_continuous['IV_ADJUSTED'][i] = results[1]['IV_adjusted'][0]
    print(col, 'done!')
df_continuous = df_continuous.sort_values(['IV_ADJUSTED', 'IV', 'BEST_CUT'], ascending = [False, True, True]).reset_index(drop = True)
df_continuous
selected_continuous_cols = ['completed_trip', 'total_fare', 'total_ETA', 'distance_travelled' ]
selected_continuous_cols = [x + '_discrete' for x in selected_continuous_cols]
selected_continuous_cols
def completed_trip_discrete(completed_trip):
    if completed_trip <= 7:
        return '0 - 7'
    if completed_trip <= 14:
        return '7 - 14'
    if completed_trip <= 28:
        return '14 - 28'
    if completed_trip <= 50:
        return '28 - 50'
    if completed_trip <= 100:
        return '50 - 100'
    return '>100'

train_1['completed_trip_discrete'] = train_1['completed_trip'].apply(lambda x: completed_trip_discrete(x))
driver_data_test['completed_trip_discrete'] = driver_data_test['completed_trip'].apply(lambda x: completed_trip_discrete(x))
woe_discrete(train_1, 'completed_trip_discrete')
def total_fare_discrete(total_fare):
    if total_fare <= 100000:
        return '< 100k'
    if total_fare <= 1000000:
        return '100k - 1 mil'
    if total_fare <= 5000000:
        return '1 - 5 mil'
    return '> 5 mil'

train_1['total_fare_discrete'] = train_1['total_fare'].apply(lambda x: total_fare_discrete(x))
driver_data_test['total_fare_discrete'] = driver_data_test['total_fare'].apply(lambda x: total_fare_discrete(x))
woe_discrete(train_1, 'total_fare_discrete')
def total_ETA_discrete(total_ETA):
    if total_ETA <= 1:
        return '< 1 min'
    if total_ETA <= 60:
        return '1 - 60 mins'
    if total_ETA <= 480:
        return '60 - 480 mins'
    return '> 480'

train_1['total_ETA_discrete'] = train_1['total_ETA'].apply(lambda x: total_ETA_discrete(x))
driver_data_test['total_ETA_discrete'] = driver_data_test['total_ETA'].apply(lambda x: total_ETA_discrete(x))
woe_discrete(train_1, 'total_ETA_discrete')
def distance_travelled_discrete(distance_travelled):
    if distance_travelled < 1:
        return '< 1 km'
    if distance_travelled <= 10:
        return '1 - 10km'
    if distance_travelled <= 50:
        return '10 - 50km'
    if distance_travelled <= 100:
        return '50 - 100km'
    if distance_travelled <= 1000:
        return '100 - 1000km'
    return '> 1000km'
train_1['distance_travelled_discrete'] = train_1['distance_travelled'].apply(lambda x: distance_travelled_discrete(x))
driver_data_test['distance_travelled_discrete'] = driver_data_test['distance_travelled'].apply(lambda x: distance_travelled_discrete(x))
woe_discrete(train_1, 'distance_travelled_discrete')
selected_cols = []
selected_cols.extend(selected_discrete_cols)
selected_cols.extend(selected_continuous_cols)
selected_cols
pd.options.display.max_rows = 15
list_dummies = [pd.get_dummies(train_1[col], prefix = col, prefix_sep = ':').iloc[:, :-1] for col in selected_cols]
train_data_X = pd.concat(list_dummies, axis = 1)
train_data_X
train_data_Y = train_1['not_churn_1w']
train_data_Y
pd.options.display.max_rows = None
reg_0 = LogisticRegression_with_p_values()
reg_0.fit(train_data_X, train_data_Y)
summary_table = pd.DataFrame(columns = ['Feature name'], data = train_data_X.columns.values)
summary_table['Coefficients'] = np.transpose(reg_0.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg_0.intercept_[0]]
summary_table = summary_table.sort_index()
p_values = reg_0.p_values
p_values = np.append(np.nan, np.array(p_values))
summary_table['p_values'] = p_values
summary_table
summary_table[summary_table['p_values'] < 0.05]
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = reg_0.model.predict_proba(train_data_X)[: ][: , 1]
Y_hat_test_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([train_data_Y, pd.DataFrame(Y_hat_test_proba)], axis = 1)
df_actual_predicted.columns = ['Y_test', 'Y_hat_test_proba']
tr = 0.5
df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_test'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC = roc_auc_score(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
AUROC
df_actual_predicted = df_actual_predicted.sort_values('Y_hat_test_proba').reset_index(drop = True)
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Y_test'].cumsum()
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Good']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / (df_actual_predicted.shape[0])
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / df_actual_predicted['Y_test'].sum()
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / (df_actual_predicted.shape[0] - df_actual_predicted['Y_test'].sum())
df_actual_predicted
# Plot Gini
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'], (1 - df_actual_predicted['Y_hat_test']).cumsum() / sum(1 - df_actual_predicted['Y_hat_test']), linestyle = '--', color = 'k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
# Here we calculate Gini from AUROC.
Gini = AUROC * 2 - 1
Gini
# Plot KS
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Bad'], color = 'r')
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Good'], color = 'b')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted['Cumulative Perc Bad'] - df_actual_predicted['Cumulative Perc Good'])
KS
for x in list(np.linspace(0, 1, 11)):
    print('Threshold:', x)
    df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > x, 1, 0)
    print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
pd.options.display.max_rows = 15
list_dummies = [pd.get_dummies(driver_data_test[col], prefix = col, prefix_sep = ':').iloc[:, :-1] for col in selected_cols]
test_data_X = pd.concat(list_dummies, axis = 1)
test_data_Y = driver_data_test['not_churn_1w']
IPD(test_data_X)
IPD(test_data_Y)
for col in [x for x in test_data_X.columns if x not in train_data_X.columns]:
    del test_data_X[col]
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = reg_0.model.predict_proba(test_data_X)[: ][: , 1]
Y_hat_test_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([test_data_Y, pd.DataFrame(Y_hat_test_proba)], axis = 1)
df_actual_predicted.columns = ['Y_test', 'Y_hat_test_proba']
tr = 0.5
df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_test'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC = roc_auc_score(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
AUROC
df_actual_predicted = df_actual_predicted.sort_values('Y_hat_test_proba').reset_index(drop = True)
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Y_test'].cumsum()
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Good']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / (df_actual_predicted.shape[0])
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / df_actual_predicted['Y_test'].sum()
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / (df_actual_predicted.shape[0] - df_actual_predicted['Y_test'].sum())
df_actual_predicted
# Plot Gini
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'], (1 - df_actual_predicted['Y_hat_test']).cumsum() / sum(1 - df_actual_predicted['Y_hat_test']), linestyle = '--', color = 'k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
# Here we calculate Gini from AUROC.
Gini = AUROC * 2 - 1
Gini
# Plot KS
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Bad'], color = 'r')
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Good'], color = 'b')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted['Cumulative Perc Bad'] - df_actual_predicted['Cumulative Perc Good'])
KS
for x in list(np.linspace(0, 1, 11)):
    print('Threshold:', x)
    df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > x, 1, 0)
    print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Population Stability Index
train_churn_perc = sum(1 - train_data_Y) / len(train_data_Y)
test_churn_perc = sum(1 - test_data_Y) / len(test_data_Y)
PSI_churn = (test_churn_perc - train_churn_perc) * np.log(test_churn_perc / train_churn_perc)
PSI_not_churn = (train_churn_perc - test_churn_perc) * np.log((1 - test_churn_perc) / (1 - train_churn_perc))
[PSI_churn, PSI_not_churn]
rf_model_1 = RandomForestClassifier(n_estimators = 100)
rf_model_1.fit(train_data_X, train_data_Y)

feature_imp = pd.Series(rf_model_1.feature_importances_,
                        index = train_data_X.columns)
feature_imp = feature_imp.sort_values(ascending = False)
feature_imp
sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = rf_model_1.predict_proba(train_data_X)[: ][: , 1]
Y_hat_test_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([train_data_Y, pd.DataFrame(Y_hat_test_proba)], axis = 1)
df_actual_predicted.columns = ['Y_test', 'Y_hat_test_proba']
tr = 0.5
df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_test'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC = roc_auc_score(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
AUROC
df_actual_predicted = df_actual_predicted.sort_values('Y_hat_test_proba').reset_index(drop = True)
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Y_test'].cumsum()
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Good']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / (df_actual_predicted.shape[0])
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / df_actual_predicted['Y_test'].sum()
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / (df_actual_predicted.shape[0] - df_actual_predicted['Y_test'].sum())
df_actual_predicted
# Plot Gini
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'], (1 - df_actual_predicted['Y_hat_test']).cumsum() / sum(1 - df_actual_predicted['Y_hat_test']), linestyle = '--', color = 'k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
# Here we calculate Gini from AUROC.
Gini = AUROC * 2 - 1
Gini
# Plot KS
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Bad'], color = 'r')
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Good'], color = 'b')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted['Cumulative Perc Bad'] - df_actual_predicted['Cumulative Perc Good'])
KS
for x in list(np.linspace(0, 1, 11)):
    print('Threshold:', x)
    df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > x, 1, 0)
    print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = rf_model_1.predict_proba(test_data_X)[: ][: , 1]
Y_hat_test_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([test_data_Y, pd.DataFrame(Y_hat_test_proba)], axis = 1)
df_actual_predicted.columns = ['Y_test', 'Y_hat_test_proba']
tr = 0.5
df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_test'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC = roc_auc_score(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
AUROC
df_actual_predicted = df_actual_predicted.sort_values('Y_hat_test_proba').reset_index(drop = True)
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Y_test'].cumsum()
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Good']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / (df_actual_predicted.shape[0])
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / df_actual_predicted['Y_test'].sum()
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / (df_actual_predicted.shape[0] - df_actual_predicted['Y_test'].sum())
df_actual_predicted
# Plot Gini
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'], (1 - df_actual_predicted['Y_hat_test']).cumsum() / sum(1 - df_actual_predicted['Y_hat_test']), linestyle = '--', color = 'k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
# Here we calculate Gini from AUROC.
Gini = AUROC * 2 - 1
Gini
# Plot KS
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Bad'], color = 'r')
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Good'], color = 'b')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted['Cumulative Perc Bad'] - df_actual_predicted['Cumulative Perc Good'])
KS
for x in list(np.linspace(0, 1, 11)):
    print('Threshold:', x)
    df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > x, 1, 0)
    print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
train_2 = driver_data_train[driver_data_train['Cluster'] == 2].reset_index(drop = True)
train_2
discrete_cols = ['ops_city_id', 'device_type', 'vehicle_model',
                  'vehicle_brand', 'segment']
df_discrete = pd.DataFrame(discrete_cols).rename(columns = {0:'DISCRETE_COL'})
df_discrete['DISTINCT'], df_discrete['IV'], df_discrete['IV_ADJUSTED'] = '', '', ''
for i in range(len(df_discrete)):
    col = df_discrete['DISCRETE_COL'][i]
    results = woe_discrete(train_2, col)
    df_discrete['DISTINCT'][i] = len(results)
    df_discrete['IV'][i] = results['IV'][0]
    df_discrete['IV_ADJUSTED'][i] = results['IV_adjusted'][0]
    print(col, 'done!')
df_discrete = df_discrete.sort_values(['IV_ADJUSTED', 'IV', 'DISTINCT'], ascending = [False, True, True]).reset_index(drop = True)
df_discrete
selected_discrete_cols = ['segment']
pd.options.display.max_rows = None
continuous_cols = ['sum_sh', 'completed_trip',
                   'total_dispatched_request', 'Valid_dispatch',
                   'accepted', 'cancelled', 'rider_cancel', 'driver_cancel',
                   'total_fare', 'gmv_aftertax', 'Trips_without_surge',
                   'Trips_with_surge', 'GB_without_surge', 'Surge_Organic_GB',
                   'Surge_Contri_GB', 'total_ETA', 'total_ATA', 'trip_time_hour',
                   'en_route_hour', 'distance_travelled', 'driver_trip_amount',
                   'weekly_amount', 'loyalty_amount']

df_continuous = pd.DataFrame(continuous_cols).rename(columns = {0:'CONTINUOUS_COL'})
df_continuous['BEST_CUT'], df_continuous['IV'], df_continuous['IV_ADJUSTED'] = '', '', ''
for i in range(len(df_continuous)):
    col = df_continuous['CONTINUOUS_COL'][i]
    results = woe_continuous_best_class_count(train_2, col)
    df_continuous['BEST_CUT'][i] = results[0]
    df_continuous['IV'][i] = results[1]['IV'][0]
    df_continuous['IV_ADJUSTED'][i] = results[1]['IV_adjusted'][0]
    print(col, 'done!')
df_continuous = df_continuous.sort_values(['IV_ADJUSTED', 'IV', 'BEST_CUT'], ascending = [False, True, True]).reset_index(drop = True)
df_continuous
selected_continuous_cols = ['completed_trip', 'total_fare', 'total_ETA', 'distance_travelled' ]
selected_continuous_cols = [x + '_discrete' for x in selected_continuous_cols]
selected_continuous_cols
def completed_trip_discrete(completed_trip):
    if completed_trip <= 7:
        return '0 - 7'
    if completed_trip <= 14:
        return '7 - 14'
    if completed_trip <= 28:
        return '14 - 28'
    if completed_trip <= 50:
        return '28 - 50'
    if completed_trip <= 100:
        return '50 - 100'
    return '>100'

train_2['completed_trip_discrete'] = train_2['completed_trip'].apply(lambda x: completed_trip_discrete(x))
driver_data_test['completed_trip_discrete'] = driver_data_test['completed_trip'].apply(lambda x: completed_trip_discrete(x))
woe_discrete(train_2, 'completed_trip_discrete')
def total_fare_discrete(total_fare):
    if total_fare <= 100000:
        return '< 100k'
    if total_fare <= 1000000:
        return '100k - 1 mil'
    if total_fare <= 5000000:
        return '1 - 5 mil'
    return '> 5 mil'

train_2['total_fare_discrete'] = train_2['total_fare'].apply(lambda x: total_fare_discrete(x))
driver_data_test['total_fare_discrete'] = driver_data_test['total_fare'].apply(lambda x: total_fare_discrete(x))
woe_discrete(train_2, 'total_fare_discrete')
def total_ETA_discrete(total_ETA):
    if total_ETA <= 1:
        return '< 1 min'
    if total_ETA <= 60:
        return '1 - 60 mins'
    if total_ETA <= 480:
        return '60 - 480 mins'
    return '> 480'

train_2['total_ETA_discrete'] = train_2['total_ETA'].apply(lambda x: total_ETA_discrete(x))
driver_data_test['total_ETA_discrete'] = driver_data_test['total_ETA'].apply(lambda x: total_ETA_discrete(x))
woe_discrete(train_2, 'total_ETA_discrete')
def distance_travelled_discrete(distance_travelled):
    if distance_travelled < 1:
        return '< 1 km'
    if distance_travelled <= 10:
        return '1 - 10km'
    if distance_travelled <= 50:
        return '10 - 50km'
    if distance_travelled <= 100:
        return '50 - 100km'
    if distance_travelled <= 1000:
        return '100 - 1000km'
    return '> 1000km'
train_2['distance_travelled_discrete'] = train_2['distance_travelled'].apply(lambda x: distance_travelled_discrete(x))
driver_data_test['distance_travelled_discrete'] = driver_data_test['distance_travelled'].apply(lambda x: distance_travelled_discrete(x))
woe_discrete(train_2, 'distance_travelled_discrete')
selected_cols = []
selected_cols.extend(selected_discrete_cols)
selected_cols.extend(selected_continuous_cols)
selected_cols
pd.options.display.max_rows = 15
list_dummies = [pd.get_dummies(train_2[col], prefix = col, prefix_sep = ':').iloc[:, :-1] for col in selected_cols]
train_data_X = pd.concat(list_dummies, axis = 1)
train_data_X
train_data_Y = train_2['not_churn_1w']
train_data_Y
pd.options.display.max_rows = None
reg_0 = LogisticRegression_with_p_values()
reg_0.fit(train_data_X, train_data_Y)
summary_table = pd.DataFrame(columns = ['Feature name'], data = train_data_X.columns.values)
summary_table['Coefficients'] = np.transpose(reg_0.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg_0.intercept_[0]]
summary_table = summary_table.sort_index()
p_values = reg_0.p_values
p_values = np.append(np.nan, np.array(p_values))
summary_table['p_values'] = p_values
summary_table
summary_table[summary_table['p_values'] < 0.05]
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = reg_0.model.predict_proba(train_data_X)[: ][: , 1]
Y_hat_test_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([train_data_Y, pd.DataFrame(Y_hat_test_proba)], axis = 1)
df_actual_predicted.columns = ['Y_test', 'Y_hat_test_proba']
tr = 0.5
df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_test'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC = roc_auc_score(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
AUROC
df_actual_predicted = df_actual_predicted.sort_values('Y_hat_test_proba').reset_index(drop = True)
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Y_test'].cumsum()
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Good']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / (df_actual_predicted.shape[0])
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / df_actual_predicted['Y_test'].sum()
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / (df_actual_predicted.shape[0] - df_actual_predicted['Y_test'].sum())
df_actual_predicted
# Plot Gini
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'], (1 - df_actual_predicted['Y_hat_test']).cumsum() / sum(1 - df_actual_predicted['Y_hat_test']), linestyle = '--', color = 'k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
# Here we calculate Gini from AUROC.
Gini = AUROC * 2 - 1
Gini
# Plot KS
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Bad'], color = 'r')
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Good'], color = 'b')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted['Cumulative Perc Bad'] - df_actual_predicted['Cumulative Perc Good'])
KS
for x in list(np.linspace(0, 1, 11)):
    print('Threshold:', x)
    df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > x, 1, 0)
    print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
pd.options.display.max_rows = 15
list_dummies = [pd.get_dummies(driver_data_test[col], prefix = col, prefix_sep = ':').iloc[:, :-1] for col in selected_cols]
test_data_X = pd.concat(list_dummies, axis = 1)
test_data_Y = driver_data_test['not_churn_1w']
IPD(test_data_X)
IPD(test_data_Y)
for col in [x for x in test_data_X.columns if x not in train_data_X.columns]:
    del test_data_X[col]
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = reg_0.model.predict_proba(test_data_X)[: ][: , 1]
Y_hat_test_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([test_data_Y, pd.DataFrame(Y_hat_test_proba)], axis = 1)
df_actual_predicted.columns = ['Y_test', 'Y_hat_test_proba']
tr = 0.9
df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_test'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC = roc_auc_score(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
AUROC
df_actual_predicted = df_actual_predicted.sort_values('Y_hat_test_proba').reset_index(drop = True)
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Y_test'].cumsum()
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Good']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / (df_actual_predicted.shape[0])
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / df_actual_predicted['Y_test'].sum()
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / (df_actual_predicted.shape[0] - df_actual_predicted['Y_test'].sum())
df_actual_predicted
# Plot Gini
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'], (1 - df_actual_predicted['Y_hat_test']).cumsum() / sum(1 - df_actual_predicted['Y_hat_test']), linestyle = '--', color = 'k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
# Here we calculate Gini from AUROC.
Gini = AUROC * 2 - 1
Gini
# Plot KS
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Bad'], color = 'r')
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Good'], color = 'b')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted['Cumulative Perc Bad'] - df_actual_predicted['Cumulative Perc Good'])
KS
for x in list(np.linspace(0, 1, 11)):
    print('Threshold:', x)
    df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > x, 1, 0)
    print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Population Stability Index
train_churn_perc = sum(1 - train_data_Y) / len(train_data_Y)
test_churn_perc = sum(1 - test_data_Y) / len(test_data_Y)
PSI_churn = (test_churn_perc - train_churn_perc) * np.log(test_churn_perc / train_churn_perc)
PSI_not_churn = (train_churn_perc - test_churn_perc) * np.log((1 - test_churn_perc) / (1 - train_churn_perc))
[PSI_churn, PSI_not_churn]
rf_model_2 = RandomForestClassifier(n_estimators = 100)
rf_model_2.fit(train_data_X, train_data_Y)

feature_imp = pd.Series(rf_model_2.feature_importances_,
                        index = train_data_X.columns)
feature_imp = feature_imp.sort_values(ascending = False)
feature_imp
sns.barplot(x = feature_imp, y = feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = rf_model_2.predict_proba(train_data_X)[: ][: , 1]
Y_hat_test_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([train_data_Y, pd.DataFrame(Y_hat_test_proba)], axis = 1)
df_actual_predicted.columns = ['Y_test', 'Y_hat_test_proba']
tr = 0.5
df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_test'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC = roc_auc_score(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
AUROC
df_actual_predicted = df_actual_predicted.sort_values('Y_hat_test_proba').reset_index(drop = True)
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Y_test'].cumsum()
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Good']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / (df_actual_predicted.shape[0])
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / df_actual_predicted['Y_test'].sum()
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / (df_actual_predicted.shape[0] - df_actual_predicted['Y_test'].sum())
df_actual_predicted
# Plot Gini
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'], (1 - df_actual_predicted['Y_hat_test']).cumsum() / sum(1 - df_actual_predicted['Y_hat_test']), linestyle = '--', color = 'k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
# Here we calculate Gini from AUROC.
Gini = AUROC * 2 - 1
Gini
# Plot KS
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Bad'], color = 'r')
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Good'], color = 'b')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted['Cumulative Perc Bad'] - df_actual_predicted['Cumulative Perc Good'])
KS
for x in list(np.linspace(0, 1, 11)):
    print('Threshold:', x)
    df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > x, 1, 0)
    print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = rf_model_2.predict_proba(test_data_X)[: ][: , 1]
Y_hat_test_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([test_data_Y, pd.DataFrame(Y_hat_test_proba)], axis = 1)
df_actual_predicted.columns = ['Y_test', 'Y_hat_test_proba']
tr = 0.7
df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_test'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC = roc_auc_score(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test_proba'])
AUROC
df_actual_predicted = df_actual_predicted.sort_values('Y_hat_test_proba').reset_index(drop = True)
df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Y_test'].cumsum()
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Good']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / (df_actual_predicted.shape[0])
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / df_actual_predicted['Y_test'].sum()
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / (df_actual_predicted.shape[0] - df_actual_predicted['Y_test'].sum())
df_actual_predicted
# Plot Gini
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'], df_actual_predicted['Cumulative Perc Population'], linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'], (1 - df_actual_predicted['Y_hat_test']).cumsum() / sum(1 - df_actual_predicted['Y_hat_test']), linestyle = '--', color = 'k')
plt.xlabel('Cumulative % Population')
plt.ylabel('Cumulative % Bad')
plt.title('Gini')
# Here we calculate Gini from AUROC.
Gini = AUROC * 2 - 1
Gini
# Plot KS
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Bad'], color = 'r')
plt.plot(df_actual_predicted['Y_hat_test_proba'], df_actual_predicted['Cumulative Perc Good'], color = 'b')
plt.xlabel('Estimated Probability for being Good')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov')
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS = max(df_actual_predicted['Cumulative Perc Bad'] - df_actual_predicted['Cumulative Perc Good'])
KS
for x in list(np.linspace(0, 1, 11)):
    print('Threshold:', x)
    df_actual_predicted['Y_hat_test'] = np.where(df_actual_predicted['Y_hat_test_proba'] > x, 1, 0)
    print(classification_report(df_actual_predicted['Y_test'], df_actual_predicted['Y_hat_test']))
