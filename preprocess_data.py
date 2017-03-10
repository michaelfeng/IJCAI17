from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import mean_squared_error
import datetime as dt
import matplotlib.pyplot as plt

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

# df contains 'user_id, shop_id, day'
del df['time']
df = df.groupby(['shop_id','day'],as_index=False).count()
df['pay_count'] = df['user_id']
del df['user_id']
df['pay_count'] = pd.to_numeric(df['pay_count'], errors='coerce').fillna(0)
# df contains 'pay_count, shop_id, day'

### Preprocess shop_info data
shop_info_names=['shop_id','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate_1_name','cate_2_name','cate_3_name']
shop_df = pd.read_csv('shop_info.txt', names=shop_info_names)

# shop_df contains 
"'shop_id','city_name','location_id','per_pay','score','comment_cnt','shop_level','cate_1_name','cate_2_name','cate_3_name'"
# Remove useless column
del shop_df['cate_3_name']

# Merge user_pay and shop_info data
merge_names=['city_name','location_id','per_pay','score','comment_cnt','shop_level','cate_1_name','cate_2_name']
df = df.merge(shop_df, left_on='shop_id', right_on='shop_id', how='left')
print 'Data after merged(shop_info, user_pay pay count in shop per/day): '
print df.head()

# df contains "'pay_count','shop_id','day', 'city_name','location_id','per_pay','score','comment_cnt','shop_level','cate_1_name','cate_2_name'"

### Preprocess user_view data
user_view_names=['user_id', 'shop_id', 'view_time']
view_df = pd.read_csv('user_view.txt', names=user_view_names)  
view_df['day'] = pd.Series(pd.to_datetime(view_df['view_time'])).dt.date
del view_df['view_time']
view_df = view_df.groupby(['shop_id','day'],as_index=False).count()
view_df['view_count'] = view_df['user_id']
del view_df['user_id']

view_df = view_df[view_df['view_count'] < 5000]

# Merge user_view with user_pay and shop_info data
df = df.merge(view_df, left_on=['shop_id','day'], right_on=['shop_id','day'], how='left')

### Clean data fill 0 for NaN value in column 
df = df.fillna(0) 

print 'Data after merged(user_view, user_pay, view per/day on shop): '
print df.head()

df['weekday'] = df['day'].apply(lambda x: dt.datetime.weekday(x))
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
df['day'] = df['day'].apply(lambda x: x.toordinal())

# Remove outliers
df = df[df['pay_count'] < 350000]

labels = df['pay_count']
del df['pay_count']

### One Hot encoding for column: cate_1_name, cate_2_name, city_name
# Vectorize the categorical columns: e & f
df = encode_onehot(df, cols=['city_name','cate_1_name','cate_2_name'])


# print 'train features: ' + str(df.columns.values)
train_features,test_features,train_labels,test_labels=train_test_split(df,labels,test_size=0.2,random_state=0)

print 'train_features header:'
print train_features.head()


'''
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn import cross_validation, metrics   #Additional scklearn functions

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
train_features_scale = scaler1.fit_transform(train_features)
train_labels_scale = scaler2.fit_transform(train_labels)

test_features_scale = scaler1.fit_transform(test_features)
test_labels_scale = scaler2.fit_transform(test_labels)


start = time()
param_test2b = {
     'gamma':[i/10.0 for i in range(0,5)]
    }
gsearch2b = GridSearchCV(estimator = xgb.XGBRegressor(learning_rate=0.1, n_estimators=1000, max_depth=9,
                                                       min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                       objective= 'count:poisson', nthread=20, scale_pos_weight=1,seed=27),
                          param_grid = param_test2b, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
gsearch2b.fit(train_features_scale, train_labels_scale)
clf = gsearch2b.best_estimator_
pred = clf.fit(train_features_scale, train_labels_scale)
print 'Output --->>>'
print gsearch2b.grid_scores_
print gsearch2b.best_params_
print gsearch2b.best_score_

print "Score : ", clf.score(test_features_scale, test_labels_scale)
print 'Mean squared error: ', mean_squared_error(test_labels, scaler2.inverse_transform(pred))
print "Train time: ", round(time() - start, 3), "s"
'''


start = time()
clf =  GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=9,min_samples_split=30, random_state=0, loss='ls')
clf.fit(train_features, train_labels)
print "Train time: ", round(time() - start, 3), "s"

start = time()
pred = clf.predict(test_features)
print "Predict time: ", round(time() - start, 3), "s"
print 'Mean squared error: ', mean_squared_error(test_labels, pred)

### Predict data that 2016-11-1 to 2016-11-14
date_list = ['2016-11-01','2016-11-02','2016-11-03','2016-11-04','2016-11-05','2016-11-06','2016-11-07','2016-11-08','2016-11-09','2016-11-10','2016-11-11','2016-11-12','2016-11-13','2016-11-14']

date_df = pd.DataFrame(data=date_list, columns=['day'])
date_df['day'] = pd.Series(pd.to_datetime(date_df['day'])).dt.date
print date_df.head()

shopid_df = pd.DataFrame(shop_df['shop_id'])
shopid_df['key'] = 0
date_df['key'] = 0
pred_df = pd.merge(shopid_df, date_df, on='key', how='left')
del pred_df['key']
# predict features contains 'shop_id, day, is_weekend, weekday, view_count, shop_info ...'
pred_df = pred_df.merge(shop_df, left_on='shop_id', right_on='shop_id', how='left') 

# Add view_count column in order to keep same number features as train data
pred_df['view_count'] = np.random.choice(range(1, 100), pred_df.shape[0])
pred_df['weekday'] = pred_df['day'].apply(lambda x: dt.datetime.weekday(x))
pred_df['is_weekend'] = pred_df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
pred_df['day'] = pred_df['day'].apply(lambda x: x.toordinal())

## TODO optimize generate average view_count for every shop_id
pred_df = pred_df.fillna(0)
pred_features_df = encode_onehot(pred_df, cols=['city_name','cate_1_name','cate_2_name'])
pred_features_ori_df = pred_df[['shop_id','day']]


print '###########'
print pred_features_ori_df.head()


start = time()
#pred_labels_df = pd.DataFrame(pred_df['shop_id'])
pred_labels = clf.predict(pred_features_df)
print 'Final Predict time: ', round(time() - start,3),'s'
#print 'Final Predict result len: ', pred_labels.shape[0]
print 'Final Predict feature len: ', pred_features_df.shape[0]

pred_labels_df = pd.DataFrame(pred_labels)
print 'Final Predict label header: '
print pred_labels_df.head()

print 'Final Predict features header: '
print pred_features_df.head()

pred_labels_df['pay_count'] = pd.DataFrame(data=pred_labels_df,columns=['pay_count'])

# pred_features_ori_df contains 'shop_id', 'day numbers'
date_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
date_df['day'] = pd.DataFrame(data=date_list, columns=['day'])
pred_features_ori_df = pd.merge(shopid_df, date_df, on='key', how='left')

print 'pred_features_ori_df header: '
print pred_features_ori_df.head()
print len(pred_features_ori_df)


print 'Begin concating data:'
result = pd.concat([pred_labels_df, pred_features_ori_df], axis=1, join_axes=[pred_labels_df.index])
del result['key']

result.to_csv('result.txt', header=False, index=False)
print result.head()

# result.txt format: pay_count: shop_id, day
print 'Finish predict.'


