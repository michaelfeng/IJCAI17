from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from time import time
from sklearn.metrics import mean_squared_error
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor

def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    """
    vec = DictVectorizer()
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

### Preprocess user_pay data
t0 = time()
user_pay_names=['user_id', 'shop_id', 'time']
df = pd.read_csv('user_pay.txt', names=user_pay_names)
df['day'] = pd.Series(pd.to_datetime(df['time'])).dt.date
#df['time'] = pd.Series(pd.to_datetime(df['time']))
del df['time']
df = df.groupby(['shop_id','day'],as_index=False).count()
df['pay_count'] = df['user_id']
del df['user_id']
df['pay_count'] = pd.to_numeric(df['pay_count'], errors='coerce').fillna(0)

### Preprocess shop_info data
shop_info_names=['shop_id','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate_1_name','cate_2_name','cate_3_name']
shop_df = pd.read_csv('shop_info.txt', names=shop_info_names)

# Remove useless column
del shop_df['cate_3_name']

# Merge user_pay and shop_info data
merge_names=['city_name','location_id','per_pay','score','comment_cnt','shop_level','cate_1_name','cate_2_name']
df = df.merge(shop_df, left_on='shop_id', right_on='shop_id', how='left')
print 'Original data for merged(shop_info, user_pay pay count in shop per/day): '
print df

### Preprocess user_view data
user_view_names=['user_id', 'shop_id', 'view_time']
view_df = pd.read_csv('user_view.txt', names=user_view_names)  
view_df['day'] = pd.Series(pd.to_datetime(view_df['view_time'])).dt.date
del view_df['view_time']
view_df = view_df.groupby(['shop_id','day'],as_index=False).count()
view_df['view_count'] = view_df['user_id']
del view_df['user_id']

# Merge user_view with user_pay and shop_info data
df = df.merge(view_df, left_on=['shop_id','day'], right_on=['shop_id','day'], how='left')

### Clean data fill 0 for NaN value in column 
df = df.fillna(0) 

print 'Original data for merged(user_view, user_pay, view per/day on shop): '
print df

### One Hot encoding for column: cate_1_name, cate_2_name, city_name
# Vectorize the categorical columns: e & f
df = encode_onehot(df, cols=['city_name','cate_1_name','cate_2_name'])
#df['time'] = df['time'].apply(lambda x: x.toordinal())
df['day'] = df['day'].apply(lambda x: x.toordinal())

labels = df['pay_count']
del df['pay_count']

# print 'train features: ' + str(df.columns.values)
train_features,test_features,train_labels,test_labels=train_test_split(df,labels,test_size=0.2,random_state=0)

print 'train_features:'
print train_features.head()
print 'train_labels:'
print train_labels.head()

# Random Tree algorithm Classifier using grid search cv
#clf = RandomForestClassifier()
'''
scaler = StandardScaler()
scaler.fit(train_features)
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)
'''
# use a full grid over all parameters
'''
param_grid = {"max_depth": range(2,20,2),
              "min_samples_split": range(2, 100, 5),
              "bootstrap": [True, False],
              'max_features': ['auto', 'sqrt', 'log2'],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)

start = time()
grid_search.fit(train_features, train_labels)
print grid_search.grid_scores_
print grid_search.best_params_
print grid_search.best_score_
print "GridSearchCV took time:", round(time() - start, 3), "s"
'''

start = time()
clf =  GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
#clf =  svm.LinearSVC(random_state=0)
#clf = linear_model.LinearRegression()
#clf = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True, verbose=0, epsilon=0.1, random_state=0, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, average=False)
clf.fit(train_features, train_labels)
print "Train time: ", round(time() - start, 3), "s"
pred = clf.predict(test_features)
print "Predict time: ", round(time() - start, 3), "s"
#print 'Predict coefficience: ',clf.coef_
print 'Mean squared error: ',mean_squared_error(test_labels, pred)


### Predict data that 2016-11-1 to 2016-11-14
#date_list = ['2016-11-01','2016-11-02','2016-11-03','2016-11-04','2016-11-05','2016-11-06','2016-11-07','2016-11-08','2016-11-09','2016-11-10','2016-11-11','2016-11-12','2016-11-13','2016-11-14']
date_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
shopid_df = pd.DataFrame(shop_df['shop_id'])
date_df = pd.DataFrame(data=date_list, columns=['day'])
shopid_df['key'] = 0
date_df['key'] = 0
pred_df = pd.merge(shopid_df, date_df, on='key', how='left')
del pred_df['key']
pred_df = pred_df.merge(shop_df, left_on='shop_id', right_on='shop_id', how='left') 

# Add view_count column in order to keep same number features as train data
pred_df['view_count'] = np.random.choice(range(1, 100), pred_df.shape[0])

## TODO optimize generate average view_count for every shop_id
#pred_features_df = pred_df[['shop_id','day','view_count']]
#pred_features_ori_df = pred_df[['shop_id','day','view_count']]
pred_df = pred_df.fillna(0)
pred_features_df = encode_onehot(pred_df, cols=['city_name','cate_1_name','cate_2_name'])
pred_features_ori_df = pred_df[['shop_id','day']]

'''
scaler = StandardScaler()
scaler.fit(pred_features_df)
pred_features_df = scaler.transform(pred_features_df)
'''

start = time()
pred_labels = clf.predict(pred_features_df)
print 'Predict time: ', round(time() - start,3),'s'
print 'Predict result len: ', pred_labels.shape[0]
print 'Predict feature len: ', pred_features_df.shape[0]
pred_labels_df = pd.DataFrame(pred_labels)
print 'Predict label header: '
print pred_labels_df.head()
print type(pred_labels_df)

result = pd.concat([pred_labels_df, pred_features_ori_df], axis=1, join_axes=[pred_labels_df.index])

print 'Predict result header: '
print result.head()

'''
ori_pred_labels = scaler.inverse_transform(pred_labels)
ori_pred_labels_df = pd.DataFrame(ori_pred_labels)
print ori_pred_labels_df.head()
'''



'''
pred_labels_df = pd.DataFrame(pred_labels)
pred_labels_df['key'] = 0
pred_features_ori_df['key'] = 0


pred_labels_df['pay_count'] = pd.DataFrame(data=pred_labels_df,columns=['pay_count'])
'''
'''
result = pd.merge(pred_labels_df, pred_features_ori_df, on='key', how='left')
del result['key']
'''
# result.to_csv('result.txt', header=False, index=False)

# result.txt format: pay_count: shop_id, day
print 'Finish predict.'


