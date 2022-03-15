import datetime
import numpy as np
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

train = pd.read_csv('./data/file1.csv')
test = pd.read_csv('./data/file2.csv')

# 1 特征工程
column_tra = ["carid", "tradeTime", "brand", "serial", "model", "mileage", "color", "cityId", "carCode", "transferCount", "seatings", "registerDate",
              "licenseDate", "country", "maketype", "modelyear", "displacement", "gearbox", "oiltype", "newprice", "anonymousFeature1", "anonymousFeature2",
              "anonymousFeature3", "anonymousFeature5", "anonymousFeature6", "anonymousFeature11", "anonymousFeature12", "anonymousFeature14", "price"]
column_te = ["carid", "tradeTime", "brand", "serial", "model", "mileage", "color", "cityId", "carCode", "transferCount", "seatings", "registerDate",
             "licenseDate", "country", "maketype", "modelyear", "displacement", "gearbox", "oiltype", "newprice", "anonymousFeature1", "anonymousFeature2",
             "anonymousFeature3", "anonymousFeature5", "anonymousFeature6", "anonymousFeature11", "anonymousFeature12", "anonymousFeature14"]
train = train[column_tra]
test = test[column_te]
## 1.1 缺失值处理
# 以下分类特征全部填充众数
train['carCode'] = train['carCode'].fillna(1)
train['modelyear'] = train['modelyear'].fillna(2017)
train['country'] = train['country'].fillna(779412)
train['maketype'] = train['maketype'].fillna(2)
train['gearbox'] = train['gearbox'].fillna(3)
train['anonymousFeature5'] = train['anonymousFeature5'].fillna(8)

test['carCode'] = test['carCode'].fillna(1)
test['modelyear'] = test['modelyear'].fillna(2017)
test['country'] = test['country'].fillna(779412)
test['maketype'] = test['maketype'].fillna(2)
test['gearbox'] = test['gearbox'].fillna(3)
test['anonymousFeature5'] = test['anonymousFeature5'].fillna(8)

train['anonymousFeature1'] = train['anonymousFeature1'].fillna(1)
train['anonymousFeature11'] = train['anonymousFeature11'].fillna('1+2')

test['anonymousFeature1'] = test['anonymousFeature1'].fillna(1)
test['anonymousFeature11'] = test['anonymousFeature11'].fillna('1+2')
## 1.2 提取时间特征
# # 时间处理(提取年月日)
train['tradeTime'] = pd.to_datetime(train['tradeTime'])
train['registerDate'] = pd.to_datetime(train['registerDate'])
train['licenseDate'] = pd.to_datetime(train['licenseDate'])
test['tradeTime'] = pd.to_datetime(test['tradeTime'])
test['registerDate'] = pd.to_datetime(test['registerDate'])
test['licenseDate'] = pd.to_datetime(test['licenseDate'])


train['tradeTime_year'] = train['tradeTime'].dt.year
train['tradeTime_month'] = train['tradeTime'].dt.month
train['tradeTime_day'] = train['tradeTime'].dt.day

train['registerDate_year'] = train['registerDate'].dt.year
train['registerDate_month'] = train['registerDate'].dt.month
train['registerDate_day'] = train['registerDate'].dt.day


test['tradeTime_year'] = test['tradeTime'].dt.year
test['tradeTime_month'] = test['tradeTime'].dt.month
test['tradeTime_day'] = test['tradeTime'].dt.day

test['registerDate_year'] = test['registerDate'].dt.year
test['registerDate_month'] = test['registerDate'].dt.month
test['registerDate_day'] = test['registerDate'].dt.day

train['licenseDate_year'] = train['licenseDate'].dt.year
train['licenseDate_month'] = train['licenseDate'].dt.month
train['licenseDate_day'] = train['licenseDate'].dt.day

test['licenseDate_year'] = test['licenseDate'].dt.year
test['licenseDate_month'] = test['licenseDate'].dt.month
test['licenseDate_day'] = test['licenseDate'].dt.day

del train['tradeTime']
del test['tradeTime']
del train['registerDate']
del test['registerDate']
del train['licenseDate']
del test['licenseDate']
## 1.3 匿名特征12的处理
series1 = train['anonymousFeature12'].str.split('*', expand=True)
train['length'] = series1[0]
train['width'] = series1[1]
train['high'] = series1[2]
series2 = test['anonymousFeature12'].str.split('*', expand=True)
test['length'] = series2[0]
test['width'] = series2[1]
test['high'] = series2[2]


train['length'] = train['length'].astype(float)
train['width'] = train['width'].astype(float)
train['high'] = train['high'].astype(float)

train['volume'] = train['length']*train['width']*train['high']

test['length'] = test['length'].astype(float)
test['width'] = test['width'].astype(float)
test['high'] = test['high'].astype(float)
test['volume'] = test['length']*test['width']*test['high']

del train['anonymousFeature12']
del test['anonymousFeature12']
## 1.4 填充测试集的缺失值
# tt = test.isnull().any()  # 用来判断某列是否有缺失值
# test中匿名特征12有一个缺失值
tt2 = test.isna().sum()
tt2[tt2 > 0]
test['length'] = test['length'].fillna(4630)
test['width'] = test['width'].fillna(1392)
test['high'] = test['high'].fillna(1470)
test['volume'] = test['volume'].fillna(4630*1392*1470)
## 1.5 匿名特征15的处理
dict = {'1': 1,
        '1+2': 2,
        '3+2': 3,
        '1+2,4+2': 4,
        '1,3+2': 5,
        '5': 6}

train['anonymousFeature11'] = train['anonymousFeature11'].map(dict)
test['anonymousFeature11'] = test['anonymousFeature11'].map(dict)
del train['anonymousFeature11']
del test['anonymousFeature11']

## 1.6 异常值的处理
train = train[train['price']<700]
train.shape

## 1.7 数据分布的转换
# 删除训练集中异常值
# df_train = train.drop(train[abs(train['newprice']-train['price']) > 100].index)
df_train = train

train_y = np.log1p(df_train['price'])
df_train['newprice'] = np.log1p(df_train['newprice'])
df_train['mileage'] = np.log1p(df_train['mileage'])
test['mileage'] = np.log1p(test['mileage'])
test['newprice'] = np.log1p(test['newprice'])
df_train.shape
## 1.8 特征交叉
## 数据分布转换
data = pd.concat([df_train[test.columns.tolist()], test],
                 ignore_index=True, sort=False)
#定义交叉特征统计
def cross_cat_num(df, num_col, cat_col):
    for f1 in tqdm(cat_col):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_col):
            feat = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max', '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
                '{}_{}_sum'.format(f1, f2): 'sum',
                '{}_{}_mad'.format(f1, f2): 'mad',
            })
            df = df.merge(feat, on=f1, how='left')
    return(df)

### 用数值特征对类别特征做统计刻画，随便挑了几个跟price相关性最高的匿名特征
cross_cat = ['width','length','anonymousFeature2','modelyear','maketype','registerDate_year','country','carCode','anonymousFeature5','gearbox',
            'cityId','anonymousFeature6','brand']

cross_num = ['newprice', 'mileage', 'displacement']
data2 = cross_cat_num(data, cross_num, cross_cat)  # 一阶交叉
data2.shape
del data2['licenseDate_year']
del data2['modelyear']
## 1.9 特征降维
# 步骤一：根据与price的相关性，筛选性相关性高的特征
# 筛选与price相关系数大于0.1的特征

tmp_train = data2[:-5000].copy()
tmp_train['price'] = train_y
col_list = tmp_train.columns.tolist()
mcorr = tmp_train[col_list].corr(method='spearman').abs()

tt = mcorr['price'] > 0.2
new_col = list(tt[tt == True].index)
new_col.remove('price')

train_df = data2[new_col][:-5000].copy()
train_df['price'] = list(train_y)
test_df = data2[new_col][-5000:].copy()
len(new_col)
# 步骤二：PCA特征降维（舍弃）
# from sklearn.decomposition import PCA
# pca = PCA(n_components=100)
# all_pca = pca.fit_transform(data2[new_col])
# all_pca_df = pd.DataFrame(all_pca)
# train_df = all_pca_df[:-5000].copy()
# train_df['price'] = list(train_y)
# test_df = all_pca_df[-5000:].copy()
# all_pca.shape
## 1.10 存储清洗后的数据
train_df.to_csv('./data/clear_train_{}.csv'.format(len(new_col)), index=0)
test_df.to_csv('./data/clear_test_{}.csv'.format(len(new_col)), index=0)

# 2 训练模型
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pandas as pd
import warnings
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import xgboost as xgb
from catboost import CatBoostRegressor
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
features = 49
train = pd.read_csv('./data/clear_train_{}.csv'.format(features))
test = pd.read_csv('./data/clear_test_{}.csv'.format(features))

train_y = train['price']
del train['price']
scaler = StandardScaler()
train_x = scaler.fit_transform(train)
test_x = scaler.fit_transform(test)
from sklearn import metrics

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mae',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
}

def MAPE_metric(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def MSE_metric(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)

def MAE_metric(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)

def Accuracy_metric(y_true, y_pred):
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n
    alpha = pd.DataFrame(abs(y_true - y_pred)/y_true)
    Accuracy = (alpha[alpha <= 0.05].count() /alpha.count())*0.8+0.2*(1-mape)
    return np.float(Accuracy)
import os
# 存储训练结果的字典
import pickle


def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if os.path.exists('./data/dict_acc.pkl_{}'.format(features)):
    # 读取字典
    dict_acc = load_dict("./data/dict_acc_{}".format(features))
    dict_mae = load_dict("./data/dict_mae_{}".format(features))
    dict_mape = load_dict("./data/dict_mape_{}".format(features))
else:
    dict_acc = {}
    dict_mape = {}
    dict_mae = {}
    preds_lgb = np.zeros(len(test_x))
    preds_xgb = np.zeros(len(test_x))
    preds_cat = np.zeros(len(test_x))
    folds = 5
    kfold = KFold(n_splits=folds, shuffle=True, random_state=5421)
    for fold, (trn_idx, val_idx) in enumerate(kfold.split(train_x, train_y)):
        import lightgbm as lgb
        print('-------fold {}-------'.format(fold))
        x_tra, y_trn, x_val, y_val = train_x[trn_idx], train_y.iloc[trn_idx], train_x[val_idx], train_y.iloc[val_idx]

        train_set = lgb.Dataset(x_tra, y_trn)
        val_set = lgb.Dataset(x_val, y_val)
        # lgb
        print('---正在训练lgb---')
        lgbmodel = lgb.train(params, train_set, num_boost_round=3000,
                             valid_sets=(train_set, val_set),
                             #   feval=Accuracy_metric,
                             early_stopping_rounds=500,
                             verbose_eval=False)
        val_pred_xgb = lgbmodel.predict(
            x_val, predict_disable_shape_check=True)
        preds_lgb += lgbmodel.predict(test_x,
                                      predict_disable_shape_check=True) / folds
        val_acc = Accuracy_metric(y_val, val_pred_xgb)
        val_mae = MAE_metric(y_val, val_pred_xgb)
        val_mape = MAPE_metric(y_val, val_pred_xgb)
        dict_acc['lgb_acc_{}'.format(fold+1)] = val_acc
        dict_mae['lgb_mae_{}'.format(fold+1)] = val_mae
        dict_mape['lgb_mape_{}'.format(fold+1)] = val_mape
        print('lgb val_acc {}'.format(val_acc))
        # xgb
        print('---正在训练XGBRegressor---')
        xgbmodel = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_jobs=-1,
            n_estimators=1000,
            max_depth=7,
            subsample=0.8,
            learning_rate=0.05,
            gamma=0,
            colsample_bytree=0.9,
            random_state=2021, max_features=None, alpha=0.3)
        xgbmodel2 = xgbmodel.fit(x_tra, y_trn, verbose=False)
        val_pred_xgb = xgbmodel2.predict(x_val)
        preds_xgb += xgbmodel2.predict(test_x,) / folds
        val_acc = Accuracy_metric(y_val, val_pred_xgb)
        val_mae = MAE_metric(y_val, val_pred_xgb)
        val_mape = MAPE_metric(y_val, val_pred_xgb)
        dict_acc['xgb_acc_{}'.format(fold+1)] = val_acc
        dict_mae['xgb_mae_{}'.format(fold+1)] = val_mae
        dict_mape['xgb_mape_{}'.format(fold+1)] = val_mape
        print('xgb val_acc {}'.format(val_acc))

        # cat
        print('---正在训练CatBoostRegressor---')
        catmodel = CatBoostRegressor(
            iterations=3000, learning_rate=0.03,
            depth=7,
            l2_leaf_reg=4,
            loss_function='MAE',
            eval_metric='MAE',
            random_seed=2021)
        catmodel2 = catmodel.fit(x_tra, y_trn, verbose=False)
        val_pred_cat = catmodel2.predict(x_val)
        preds_cat += catmodel2.predict(test_x,) / folds
        val_acc = Accuracy_metric(y_val, val_pred_cat)
        val_mae = MAE_metric(y_val, val_pred_cat)
        val_mape = MAPE_metric(y_val, val_pred_cat)
        dict_acc['cat_acc_{}'.format(fold+1)] = val_acc
        dict_mae['cat_mae_{}'.format(fold+1)] = val_mae
        dict_mape['cat_mape_{}'.format(fold+1)] = val_mape
        print('cat val_acc {}'.format(val_acc))

        # lr
        print('---正在训练LinearRegression---')
        lrmodel = LinearRegression().fit(x_tra, y_trn)
        val_pred_lr = lrmodel.predict(x_val)
        val_acc = Accuracy_metric(y_val, val_pred_lr)
        val_mae = MAE_metric(y_val, val_pred_lr)
        val_mape = MAPE_metric(y_val, val_pred_lr)
        dict_acc['lr_acc_{}'.format(fold+1)] = val_acc
        dict_mae['lr_mae_{}'.format(fold+1)] = val_mae
        dict_mape['lr_mape_{}'.format(fold+1)] = val_mape
        print('lr val_acc {}'.format(val_acc))

        # knn
        print('---正在训练KNeighborsRegressor---')
        knnmodel = KNeighborsRegressor(n_neighbors=8).fit(x_tra, y_trn)
        val_pred_knn = knnmodel.predict(x_val)
        val_acc = Accuracy_metric(y_val, val_pred_knn)
        val_mae = MAE_metric(y_val, val_pred_knn)
        val_mape = MAPE_metric(y_val, val_pred_knn)
        dict_acc['knn_acc_{}'.format(fold+1)] = val_acc
        dict_mae['knn_mae_{}'.format(fold+1)] = val_mae
        dict_mape['knn_mape_{}'.format(fold+1)] = val_mape
        print('knn val_acc {}'.format(val_acc))

        # rf
        print('---正在训练RandomForestRegressor---')
        rfmodel = RandomForestRegressor(n_estimators=200).fit(x_tra, y_trn)
        val_pred_rf = rfmodel.predict(x_val)
        val_acc = Accuracy_metric(y_val, val_pred_rf)
        val_mae = MAE_metric(y_val, val_pred_rf)
        val_mape = MAPE_metric(y_val, val_pred_rf)
        dict_acc['rf_acc_{}'.format(fold+1)] = val_acc
        dict_mae['rf_mae_{}'.format(fold+1)] = val_mae
        dict_mape['rf_mape_{}'.format(fold+1)] = val_mape
        print('rf val_acc {}'.format(val_acc))
        print('-'*20)

    # dict_acc['lr_5'] = 0.7032255321076033
    # 存储字典
    save_dict(dict_acc, "./data/dict_acc_{}".format(features))
    save_dict(dict_mae, "./data/dict_mae_{}".format(features))
    save_dict(dict_mape, "./data/dict_mape_{}".format(features))
## 2。2 模型对比的可视化
import os
# 存储训练结果的字典
import pickle


def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


features = 161
dict_acc = load_dict("./data/dict_acc_{}".format(features))
dict_mae = load_dict("./data/dict_mae_{}".format(features))
dict_mape = load_dict("./data/dict_mape_{}".format(features))
dict_mae['lr_mae_5'] = 0.11017896039957757
dict_acc['lr_acc_5'] = 0.6992424167110016
dict_mape['lr_mape_5'] = 5.245458311165871

val_acc_lr = []
val_acc_knn = []
val_acc_cat =[]
val_acc_lgb = []
val_acc_xgb = []

val_mae_lr = []
val_mae_knn = []
val_mae_cat = []
val_mae_lgb = []
val_mae_xgb = []

val_mape_lr = []
val_mape_knn = []
val_mape_cat = []
val_mape_lgb = []
val_mape_xgb = []
for i in range(5):
    val_acc_lr.append(dict_acc['lr_acc_{}'.format(i+1)])
    val_acc_knn.append(dict_acc['knn_acc_{}'.format(i+1)])
    val_acc_cat.append(dict_acc['cat_acc_{}'.format(i+1)])
    val_acc_lgb.append(dict_acc['lgb_acc_{}'.format(i+1)])
    val_acc_xgb.append(dict_acc['xgb_acc_{}'.format(i+1)])

    val_mae_lr.append(dict_mae['lr_mae_{}'.format(i+1)])
    val_mae_knn.append(dict_mae['knn_mae_{}'.format(i+1)])
    val_mae_cat.append(dict_mae['cat_mae_{}'.format(i+1)])
    val_mae_lgb.append(dict_mae['lgb_mae_{}'.format(i+1)])
    val_mae_xgb.append(dict_mae['xgb_mae{}'.format(i+1)])

    val_mape_lr.append(dict_mape['lr_mape_{}'.format(i+1)])
    val_mape_knn.append(dict_mape['knn_mape_{}'.format(i+1)])
    val_mape_cat.append(dict_mape['cat_mape_{}'.format(i+1)])
    val_mape_lgb.append(dict_mape['lgb_mape_{}'.format(i+1)])
    val_mape_xgb.append(dict_mape['xgb_mape_{}'.format(i+1)])
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(20, 5))
plt.subplot(131)

x = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']
plt.plot(x, val_acc_lr)
plt.plot(x, val_acc_knn)
plt.plot(x, val_acc_cat)
plt.plot(x, val_acc_lgb)
plt.plot(x, val_acc_xgb, "#FF0000")
plt.xlabel("Fold", fontsize=15)
plt.ylabel("Acc", fontsize=15)
plt.legend(['LR', 'KNN', 'Cat', 'LGB', 'XGB'], loc="upper right")
plt.title("准确率", fontsize=20)


plt.subplot(132)
plt.plot(x, val_mae_lr)
plt.plot(x, val_mae_knn)
plt.plot(x, val_mae_cat)
plt.plot(x, val_mae_lgb)
plt.plot(x, val_mae_xgb, "#FF0000")
plt.xlabel("Fold", fontsize=15)
plt.ylabel("Mae", fontsize=15)
plt.legend(['LR', 'KNN', 'Cat', 'LGB', 'XGB'], loc="upper right")
plt.title("平均绝对误差", fontsize=20)


plt.subplot(133)
plt.plot(x, val_mape_lr)
plt.plot(x, val_mape_knn)
plt.plot(x, val_mape_cat)
plt.plot(x, val_mape_lgb)
plt.plot(x, val_mape_xgb, "#FF0000")
plt.xlabel("Fold", fontsize=15)
plt.ylabel("Mape", fontsize=15)
plt.legend(['LR', 'KNN', 'Cat', 'LGB', 'XGB'], loc="upper right")
plt.title("平均绝对百分比误差", fontsize=20)
# plt.savefig('img/Q1-img/多模型对比.png', dpi=300)
# 3模型融合
submit_df = pd.DataFrame(columns=['price'])
submit_df['price'] = (preds_xgb+preds_lgb+preds_cat)/3
submit_df = submit_df.price.apply(np.expm1)  # np.log1p与np.expm1互为逆运算
submit_df.to_csv('./data/估价模型结果.csv', header=0)
