from google.colab import drive
drive.mount('/content/drive')
import urllib
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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, linear_model
import scipy.stats as stat
from sklearn.metrics import roc_curve, roc_auc_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from IPython.display import display as IPD
!pip install duckdb
import duckdb
con = duckdb.connect(database = ':memory:')
import plotly.graph_objects as pgo
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
driver_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data112023.csv')

driver_data_dt = ['week', 'activated_date', 'ftt_time']
for i in driver_data_dt:
    driver_data[i] = pd.to_datetime(driver_data[i])

driver_data['vehicle_model'] = driver_data['vehicle_model'].fillna('None')
driver_data # 761102 rows
query = '''
WITH
t1 AS (SELECT DISTINCT week FROM driver_data),

t2 AS
(SELECT DISTINCT
    driver_id, activated_date, ops_city_id, device_type,
    ftt_time, vehicle_brand, vehicle_model
FROM driver_data),

t3 AS (SELECT * FROM t1 CROSS JOIN t2),

t4 AS
(SELECT
    t3.*, t.sum_sh,	t.segment, t.completed_trip, t.total_dispatched_request,
    t.Valid_dispatch, t.accepted, t.cancelled, t.rider_cancel, t.driver_cancel,
    t.total_fare, t.gmv_aftertax, t.Trips_without_surge, t.Trips_with_surge,
    t.GB_without_surge, t.Surge_Organic_GB, t.Surge_Contri_GB, t.total_ETA,
    t.total_ATA, t.trip_time_hour, t.en_route_hour, t.distance_travelled,
    t.driver_trip_amount, t.weekly_amount, t.loyalty_amount, t.not_churn
FROM t3 LEFT JOIN driver_data AS t
ON t3.week = t.week AND t3.driver_id = t.driver_id),

t5 AS
(SELECT driver_id, MIN(week) AS first_week FROM driver_data
GROUP BY driver_id)

SELECT t4.* FROM t4 LEFT JOIN t5 ON t4.driver_id = t5.driver_id
WHERE t4.week >= t5.first_week
ORDER BY t4.week, t4.driver_id
'''
driver_data = con.execute(query).df()
driver_data['segment'] = driver_data['segment'].fillna('Parttime')
driver_data = driver_data.fillna(0).reset_index(drop = True)
driver_data # 986830 rows
query = '''
SELECT n.*, f.not_churn AS not_churn_1w
FROM driver_data AS n LEFT JOIN driver_data AS f
ON n.driver_id = f.driver_id
WHERE DATEDIFF('DAY', n.week, f.week) = 7
ORDER BY n.week, n.driver_id
'''
driver_data = con.execute(query).df()
driver_data
print('Through-the-Cycle Churn Rate:',
      1 - driver_data[driver_data['not_churn'] == 1]['not_churn_1w'].mean())
# Split Train - Test
latest_week = driver_data['week'].max()
driver_data_test = driver_data[driver_data['week'] == latest_week].reset_index(drop = True)
driver_data_train = driver_data[driver_data['week'] != latest_week].reset_index(drop = True)
IPD(driver_data_train)
IPD(driver_data_test)
driver_data.describe()
driver_data.columns.values
driver_data.dtypes
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
    results = woe_discrete(driver_data_train, col)
    df_discrete['DISTINCT'][i] = len(results)
    df_discrete['IV'][i] = results['IV'][0]
    df_discrete['IV_ADJUSTED'][i] = results['IV_adjusted'][0]
    print(col, 'done!')
df_discrete = df_discrete.sort_values(['IV_ADJUSTED', 'IV', 'DISTINCT'], ascending = [False, True, True]).reset_index(drop = True)
df_discrete
selected_discrete_cols = ['segment']
plot_by_woe(woe_discrete(driver_data_train, 'segment'))
woe_discrete(driver_data_train, 'segment')
driver_data_train.columns
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
    results = woe_continuous_best_class_count(driver_data_train, col)
    df_continuous['BEST_CUT'][i] = results[0]
    df_continuous['IV'][i] = results[1]['IV'][0]
    df_continuous['IV_ADJUSTED'][i] = results[1]['IV_adjusted'][0]
    print(col, 'done!')
df_continuous = df_continuous.sort_values(['IV_ADJUSTED', 'IV', 'BEST_CUT'], ascending = [False, True, True]).reset_index(drop = True)
df_continuous
woe_continuous_best_class_count(driver_data, 'driver_cancel')[1]
plot_by_woe(woe_continuous_best_class_count(driver_data_train, 'completed_trip')[1], 90)
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

driver_data_train['completed_trip_discrete'] = driver_data_train['completed_trip'].apply(lambda x: completed_trip_discrete(x))
driver_data_test['completed_trip_discrete'] = driver_data_test['completed_trip'].apply(lambda x: completed_trip_discrete(x))
woe_discrete(driver_data_train, 'completed_trip_discrete')
def total_fare_discrete(total_fare):
    if total_fare <= 100000:
        return '< 100k'
    if total_fare <= 1000000:
        return '100k - 1 mil'
    if total_fare <= 5000000:
        return '1 - 5 mil'
    return '> 5 mil'

driver_data_train['total_fare_discrete'] = driver_data_train['total_fare'].apply(lambda x: total_fare_discrete(x))
driver_data_test['total_fare_discrete'] = driver_data_test['total_fare'].apply(lambda x: total_fare_discrete(x))
woe_discrete(driver_data_train, 'total_fare_discrete')
def total_ETA_discrete(total_ETA):
    if total_ETA <= 1:
        return '< 1 min'
    if total_ETA <= 60:
        return '1 - 60 mins'
    if total_ETA <= 480:
        return '60 - 480 mins'
    return '> 480'

driver_data_train['total_ETA_discrete'] = driver_data_train['total_ETA'].apply(lambda x: total_ETA_discrete(x))
driver_data_test['total_ETA_discrete'] = driver_data_test['total_ETA'].apply(lambda x: total_ETA_discrete(x))
woe_discrete(driver_data_train, 'total_ETA_discrete')
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
driver_data_train['distance_travelled_discrete'] = driver_data_train['distance_travelled'].apply(lambda x: distance_travelled_discrete(x))
driver_data_test['distance_travelled_discrete'] = driver_data_test['distance_travelled'].apply(lambda x: distance_travelled_discrete(x))
woe_discrete(driver_data_train, 'distance_travelled_discrete')
selected_cols = []
selected_cols.extend(selected_discrete_cols)
selected_cols.extend(selected_continuous_cols)
selected_cols
pd.options.display.max_rows = 15
list_dummies = [pd.get_dummies(driver_data_train[col], prefix = col, prefix_sep = ':').iloc[:, :-1] for col in selected_cols]
train_data_X = pd.concat(list_dummies, axis = 1)
train_data_X
train_data_Y = driver_data_train['not_churn_1w']
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
reg = LogisticRegression_with_p_values()
reg.fit(train_data_X, train_data_Y)
summary_table = pd.DataFrame(columns = ['Feature name'], data = train_data_X.columns.values)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
p_values = reg.p_values
p_values = np.append(np.nan, np.array(p_values))
summary_table['p_values'] = p_values
summary_table
summary_table[summary_table['p_values'] < 0.05]
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_train_proba = reg.model.predict_proba(train_data_X)[: ][: , 1]
Y_hat_train_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([train_data_Y, pd.DataFrame(Y_hat_train_proba)], axis = 1)
df_actual_predicted.columns = ['Y_train', 'Y_hat_train_proba']
tr = 0.6
df_actual_predicted['Y_hat_train'] = np.where(df_actual_predicted['Y_hat_train_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_train'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_train'], df_actual_predicted['Y_hat_train']))
print(classification_report(df_actual_predicted['Y_train'], df_actual_predicted['Y_hat_train']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_train'],
                                 df_actual_predicted['Y_hat_train_proba'])
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
plt.plot(df_actual_predicted['Cumulative Perc Population'],
         df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'],
         df_actual_predicted['Cumulative Perc Population'],
         linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'],
         (1 - df_actual_predicted['Y_hat_train']).cumsum() / sum(1 - df_actual_predicted['Y_hat_train']),
         linestyle = '--', color = 'k')
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
result_dev = driver_data_train[['week', 'not_churn', 'not_churn_1w']]
result_dev['predict'] = np.where(Y_hat_train_proba > tr, 1, 0)
result_dev = result_dev.query('not_churn == 1').reset_index(drop = True)
del result_dev['not_churn']
pd.options.display.max_rows = 15
list_dummies = [pd.get_dummies(driver_data_test[col], prefix = col, prefix_sep = ':').iloc[:, :-1] for col in selected_cols]
test_data_X = pd.concat(list_dummies, axis = 1)
test_data_Y = driver_data_test['not_churn_1w']
IPD(test_data_X)
IPD(test_data_Y)
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = reg.model.predict_proba(test_data_X)[: ][: , 1]
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
result_oot = driver_data_test[['week', 'not_churn', 'not_churn_1w']]
result_oot['predict'] = np.where(Y_hat_test_proba > tr, 1, 0)
result_oot = result_oot.query('not_churn == 1').reset_index(drop = True)
del result_oot['not_churn']
result = pd.concat([result_dev, result_oot])
result
query = '''
WITH
t1 AS
(SELECT week, COUNT(*) AS not_churn
FROM result WHERE not_churn_1w = 1 GROUP BY week),
t2 AS
(SELECT week, COUNT(*) AS churn
FROM result WHERE not_churn_1w = 0 GROUP BY week),
t3 AS
(SELECT t1.week, t2.churn * 100.0 / (t1.not_churn + t2.churn) AS churn_actual
FROM t1 LEFT JOIN t2 ON t1.week = t2.week),

t4 AS
(SELECT week, COUNT(*) AS not_churn
FROM result WHERE predict = 1 GROUP BY week),
t5 AS
(SELECT week, COUNT(*) AS churn
FROM result WHERE predict = 0 GROUP BY week),
t6 AS
(SELECT t4.week, t5.churn * 100.0 / (t4.not_churn + t5.churn) AS churn_predict
FROM t4 LEFT JOIN t5 ON t4.week = t5.week)

SELECT t3.*, t6.churn_predict
FROM t3 LEFT JOIN t6 ON t3.week = t6.week
ORDER BY t3.week
'''
draw = con.execute(query).df()
draw.to_csv('logistic_1w.csv', index = False)
IPD(draw)
fig = make_subplots(specs = [[{'secondary_y' : True}]])
fig.add_trace(
    pgo.Scatter(x = draw['week'], y = draw['churn_actual'],
                name = 'Actual Churn Rate', marker = dict(color = '#1068B3'),
                mode = 'lines + markers'))
fig.add_trace(
    pgo.Scatter(x = draw['week'], y = draw['churn_predict'],
                name = 'Predicted Churn Rate', marker = dict(color = '#FEC318'),
                mode = 'lines + markers'))
fig.update_layout(
    title = 'Actual & Predicted Churn Rate - Logistic Regression',
    yaxis_title = 'Churn Rate (Percentage)',
    legend = dict(orientation = 'h'))
fig.show()
# RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
#                        criterion='gini', max_depth=None, max_features='auto',
#                        max_leaf_nodes=None, max_samples=None,
#                        min_impurity_decrease=0.0,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=100,
#                        n_jobs=None, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False) 
rf_model = RandomForestClassifier(n_estimators = 100)
rf_model.fit(train_data_X, train_data_Y)
feature_imp = pd.Series(rf_model.feature_importances_,
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
Y_hat_train_proba = rf_model.predict_proba(train_data_X)[: ][: , 1]
Y_hat_train_proba
pd.options.display.max_rows = 15
df_actual_predicted = pd.concat([train_data_Y, pd.DataFrame(Y_hat_train_proba)], axis = 1)
df_actual_predicted.columns = ['Y_train', 'Y_hat_train_proba']
tr = 0.6
df_actual_predicted['Y_hat_train'] = np.where(df_actual_predicted['Y_hat_train_proba'] > tr, 1, 0)
df_actual_predicted
df_actual_predicted['Y_hat_train'].value_counts()
print('Confusion Matrix: (Actual by Row, Predicted by Column)')
print(confusion_matrix(df_actual_predicted['Y_train'], df_actual_predicted['Y_hat_train']))
print(classification_report(df_actual_predicted['Y_train'], df_actual_predicted['Y_hat_train']))
# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted['Y_train'],
                                 df_actual_predicted['Y_hat_train_proba'])
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
plt.plot(df_actual_predicted['Cumulative Perc Population'],
         df_actual_predicted['Cumulative Perc Bad'])
plt.plot(df_actual_predicted['Cumulative Perc Population'],
         df_actual_predicted['Cumulative Perc Population'],
         linestyle = '--', color = 'k')
plt.plot(df_actual_predicted['Cumulative Perc Population'],
         (1 - df_actual_predicted['Y_hat_train']).cumsum() / sum(1 - df_actual_predicted['Y_hat_train']),
         linestyle = '--', color = 'k')
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
result_dev = driver_data_train[['week', 'not_churn', 'not_churn_1w']]
result_dev['predict'] = np.where(Y_hat_train_proba > tr, 1, 0)
result_dev = result_dev.query('not_churn == 1').reset_index(drop = True)
del result_dev['not_churn']
# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
Y_hat_test_proba = rf_model.predict_proba(test_data_X)[: ][: , 1]
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
result_oot = driver_data_test[['week', 'not_churn', 'not_churn_1w']]
result_oot['predict'] = np.where(Y_hat_test_proba > tr, 1, 0)
result_oot = result_oot.query('not_churn == 1').reset_index(drop = True)
del result_oot['not_churn']
result = pd.concat([result_dev, result_oot])
result
query = '''
WITH
t1 AS
(SELECT week, COUNT(*) AS not_churn
FROM result WHERE not_churn_1w = 1 GROUP BY week),
t2 AS
(SELECT week, COUNT(*) AS churn
FROM result WHERE not_churn_1w = 0 GROUP BY week),
t3 AS
(SELECT t1.week, t2.churn * 100.0 / (t1.not_churn + t2.churn) AS churn_actual
FROM t1 LEFT JOIN t2 ON t1.week = t2.week),

t4 AS
(SELECT week, COUNT(*) AS not_churn
FROM result WHERE predict = 1 GROUP BY week),
t5 AS
(SELECT week, COUNT(*) AS churn
FROM result WHERE predict = 0 GROUP BY week),
t6 AS
(SELECT t4.week, t5.churn * 100.0 / (t4.not_churn + t5.churn) AS churn_predict
FROM t4 LEFT JOIN t5 ON t4.week = t5.week)

SELECT t3.*, t6.churn_predict
FROM t3 LEFT JOIN t6 ON t3.week = t6.week
ORDER BY t3.week
'''
draw = con.execute(query).df()
draw.to_csv('rf_1w.csv', index = False)
fig = make_subplots(specs = [[{'secondary_y' : True}]])
fig.add_trace(
    pgo.Scatter(x = draw['week'], y = draw['churn_actual'],
                name = 'Actual Churn Rate', marker = dict(color = '#1068B3'),
                mode = 'lines + markers'))
fig.add_trace(
    pgo.Scatter(x = draw['week'], y = draw['churn_predict'],
                name = 'Predicted Churn Rate', marker = dict(color = '#FEC318'),
                mode = 'lines + markers'))
fig.update_layout(
    title = 'Actual & Predicted Churn Rate - Random Forest',
    yaxis_title = 'Churn Rate (Percentage)',
    legend = dict(orientation = 'h'))
fig.show()
