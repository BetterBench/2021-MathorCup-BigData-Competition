import scipy.stats as st
import pandas as pd 
import seaborn as sns
from pylab import mpl 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
plt.rcParams['font.sans-serif'] = ['STSong']
mpl.rcParams['font.sans-serif'] = ['STSong'] # 指定默认字体 
mpl.rcParams['axes.unicode_minus'] = False
df1 = pd.read_csv('./data/file1.csv')
df4 = pd.read_csv('./data/file4.csv')
df_11 = df1[['carid','price']]
df4 = df_11.merge(df4,on='carid')

# 将最后一次调整价格的时间，作为pushDate，最新的价格作为pushPrice
def clear_time(txt_json):
    if txt_json == '{}':
        return np.nan
    else:
        update_time = txt_json[2:-2].split(',')[-1].split(':')[0].strip()[2:-2]
        update_price = txt_json[2:-
                                2].split(',')[-1].split(':')[1].strip()[2:-2]
        barging_times = len(txt_json[2:-2].split(','))
        new_str = update_time+','+update_price+','+str(barging_times)
        return new_str


df4['updatePriceTimeJson'] = df4['updatePriceTimeJson'].progress_apply(
    clear_time)
sep = df4['updatePriceTimeJson'].astype(str).str.split(',', expand=True)
df4['update_time'] = sep[0].astype(str)
df4['update_price'] = sep[1].astype(float)
# 提取降价次数特征
df4['barging_times'] = sep[2].astype(float)
# 提取降价幅度特征
df4['barging_price'] = np.nan
for i in range(df4.shape[0]):
    update_time = df4['update_time'][i]
    update_price = df4['update_price'][i]
    if pd.isnull(update_price) == False:
        df4['pushDate'][i] = update_time
        df4['pushPrice'][i] = update_price
        df4['barging_price'][i] = df4['price'][i]-update_price
    else:
        df4['barging_price'][i] = 0
# df4 = tmpdf4.copy()
# 1 数据预处理
df_trans = df4[df4.withdrawDate.notna()]
df_trans.shape
## 1。1 计算交易周期
df_trans['pushDate'] = pd.to_datetime(df_trans['pushDate'])
df_trans['withdrawDate'] = pd.to_datetime(df_trans['withdrawDate'])
trans_circle = pd.DataFrame(df_trans['withdrawDate'] - df_trans['pushDate'])
df_trans['transcycle'] = trans_circle[0]

# 转为整型
sep = df_trans['transcycle'].astype(str).str.split(' ', expand=True)
df_trans['transcycle'] = sep[0].astype(int)
df4['transcycle'] = sep[0].astype(int)
# 取file4中与file1中相同cardid的数据
trans_circle_info = df1[df1.carid.isin(df4.carid.tolist())]
# trans_circle_info.info()
df5 = df4[['carid','transcycle','barging_times','barging_price']]
NEW_trans_circle = trans_circle_info.merge(df5,on='carid')
NEW_trans_circle
## 1.2 交易周期的分桶
# 分成三类，没卖出去，为一类
NEW_trans_circle.loc[NEW_trans_circle[NEW_trans_circle['transcycle']<= 7].index, 'trans_category'] = 1
NEW_trans_circle.loc[NEW_trans_circle[NEW_trans_circle['transcycle']> 7].index, 'trans_category'] = 2

# 没卖出去的，交易周期编码为0
NEW_trans_circle['trans_category'] = NEW_trans_circle['trans_category'].fillna(0)
NEW_trans_circle['carCode'] = NEW_trans_circle['carCode'].fillna(1)
NEW_trans_circle['maketype'] = NEW_trans_circle['maketype'].fillna(2)
plt.figure(figsize=(14, 5))
plt.subplot(122)
plt.title('正态分布拟合-已处理', fontsize=20)
sns.distplot(np.log1p(NEW_trans_circle['mileage']), kde=False, fit=st.norm)
plt.xlabel('里程', fontsize=20)
plt.subplot(121)
plt.title('正态分布拟合-未处理', fontsize=20)
sns.distplot(NEW_trans_circle['mileage'], kde=False, fit=st.norm)
plt.xlabel('里程', fontsize=20)
plt.savefig('img/Q2-img/1 里程正态分布拟合.png',dpi=300)
# plt.savefig('img/test2.png')
## 1.3 里程分桶
# y_p = NEW_trans_circle[NEW_trans_circle['mileage'] <= 200]
y_p = NEW_trans_circle
## 3) 查看预测值的具体频数
plt.hist(y_p['mileage'], orientation='vertical',
         histtype='bar')
plt.savefig('img/Q2-img/2 里程分桶.png',dpi=300)
plt.show()
NEW_trans_circle.loc[NEW_trans_circle[NEW_trans_circle['mileage']<= 10].index, 'mileage_category'] = 1
NEW_trans_circle.loc[NEW_trans_circle[(NEW_trans_circle['mileage']> 10)&(NEW_trans_circle['mileage']<= 20)].index, 'mileage_category'] = 2
NEW_trans_circle.loc[NEW_trans_circle[(NEW_trans_circle['mileage']> 20)&(NEW_trans_circle['mileage']<= 30)].index, 'mileage_category'] = 3
NEW_trans_circle.loc[NEW_trans_circle[(NEW_trans_circle['mileage']> 30)].index, 'mileage_category'] = 4
## 1.4 排量的分桶
#微型轿车（排量为1L以下）、普通级轿车（排量为1.0~1.6L)、中级轿车（排量为1.6~2.5L）、中高级轿车（排量为2.5~4.0L）、高级轿车（排量为4L以上）
NEW_trans_circle.loc[NEW_trans_circle[NEW_trans_circle['displacement']<= 1].index, 'displacement_category'] = 1
NEW_trans_circle.loc[NEW_trans_circle[(NEW_trans_circle['displacement']> 1)&(NEW_trans_circle['displacement']<= 1.6)].index, 'displacement_category'] = 2
NEW_trans_circle.loc[NEW_trans_circle[(NEW_trans_circle['displacement']> 1.6)&(NEW_trans_circle['displacement']<=2.5)].index, 'displacement_category'] = 3
NEW_trans_circle.loc[NEW_trans_circle[(NEW_trans_circle['displacement']> 2.5)&(NEW_trans_circle['displacement']<=4.0)].index, 'displacement_category'] = 4
NEW_trans_circle.loc[NEW_trans_circle[(NEW_trans_circle['displacement']> 4.0)].index, 'displacement_category'] = 5
## 1.5 价格分桶
# 15w下，15-35W 35-75W 75-250W 250w上，对应低端，中端，中高端，高端，豪车
NEW_trans_circle.loc[NEW_trans_circle[NEW_trans_circle['price']<= 15].index, 'price_category'] = 1 # 低端价
NEW_trans_circle.loc[NEW_trans_circle[(NEW_trans_circle['price']> 15)&(NEW_trans_circle['price']<= 35)].index, 'price_category'] = 2 # 中端价
NEW_trans_circle.loc[NEW_trans_circle[(NEW_trans_circle['price']> 35)&(NEW_trans_circle['price']<= 75)].index, 'price_category'] = 3 # 中高端价
NEW_trans_circle.loc[NEW_trans_circle[(NEW_trans_circle['price']> 75)&(NEW_trans_circle['price']<= 250)].index, 'price_category'] = 4 # 高端价
NEW_trans_circle.loc[NEW_trans_circle[(NEW_trans_circle['price']> 250)].index, 'price_category'] = 5 # 豪车价
cols = ["carid", "tradeTime", "brand", "serial", "model", "mileage", "color", "cityId", "carCode", "transferCount", "seatings", "registerDate",
        "licenseDate", "country", "maketype", "modelyear", "displacement", "gearbox", "oiltype", "newprice", "price",'barging_times',
        'barging_price','price_category','displacement_category','mileage_category','trans_category','transcycle']
df = NEW_trans_circle[cols]
# 以下分类特征全部填充众数
## 1.6缺失值处理
df['carCode'] = df['carCode'].fillna(1)
df['modelyear'] = df['modelyear'].fillna(2017)
df['country'] = df['country'].fillna(779412)
df['maketype'] = df['maketype'].fillna(2)
df['gearbox'] = df['gearbox'].fillna(3)
df['barging_times'] = df['barging_times'].fillna(0)
df['barging_price'] = df['barging_price'].fillna(0)
df['transcycle'] = df['transcycle'].fillna(0)

## 1.7 提取时间特征
# # 时间处理(提取年月日)
df['tradeTime'] = pd.to_datetime(df['tradeTime'])
df['registerDate'] = pd.to_datetime(df['registerDate'])
df['licenseDate'] = pd.to_datetime(df['licenseDate'])


df['tradeTime_year'] = df['tradeTime'].dt.year
df['tradeTime_month'] = df['tradeTime'].dt.month
df['tradeTime_day'] = df['tradeTime'].dt.day

df['registerDate_year'] = df['registerDate'].dt.year
df['registerDate_month'] = df['registerDate'].dt.month
df['registerDate_day'] = df['registerDate'].dt.day

df['licenseDate_year'] = df['licenseDate'].dt.year
df['licenseDate_month'] = df['licenseDate'].dt.month
df['licenseDate_day'] = df['licenseDate'].dt.day

del df['tradeTime']
del df['registerDate']
del df['licenseDate']
## 1.8 数据分布的转换
# 删除训练集中异常值
# df_train = train.drop(train[abs(train['newprice']-train['price']) > 100].index)
df_a = df.copy()
df_a['price'] = np.log1p(df_a['price'])
df_a['newprice'] = np.log1p(df_a['newprice'])
df_a['mileage'] = np.log1p(df_a['mileage'])
df_a
## 1.9 筛选价格回归的关键因素
# 筛选与price相关系数大于0.1的特征
mcorr = df_a.corr(method='spearman').abs()

tt = mcorr['price'] > 0.05
new_col = list(tt[tt == True].index)
new_col.remove('price')
# 删除共线性的三个特征，registerDate_year、modelyear
new_col.remove('licenseDate_year')
new_col.remove('modelyear')

df_D = df_a[new_col]
df_D['price'] = df_a['price']
df_D['trans_category'] = df_a['trans_category']
mcorr = df_a.corr(method='spearman').abs()
trans_df = pd.DataFrame(mcorr['transcycle']).sort_values(by='transcycle',ascending=False)
del trans_df['transcycle']
dict = {
 'model':'车型ID',
 'brand':'品牌ID',
 'mileage':'里程',
 'serial':'车系ID',
 'color':'颜色',
 'cityId':'城市ID',
 'oiltype':'燃油类型',
 'carCode':'国标码',
 'seatings':'载客人数',
 'country':'国别',
 'maketype':'厂商类型',
 'displacement':'排量',
 'displacement_category':'排量类别',
 'gearbox':'变速箱',
 'newprice':'新车价',
 'price':'价格',
 'transferCount':'过户次数',
 'modelyear':'年款',
 'price':'价格',
 'barging_times':'降价次数',
 'price_category':'价格级别',
 'mileage_category':'里程类别',
 'tradeTime_year':'展销年份',
 'registerDate_year':"注册年份",
 'licenseDate_month':'上牌月份',
 'licenseDate_year':"上牌时间"}
mcorr_df = pd.DataFrame(mcorr[mcorr['price']> 0.05]['price'])
# mcorr_df['price'].shape
mcorr_df2 = mcorr_df.rename(dict)
# mcorr_df2.T.to_csv('./data/Q2/第二问与价格的相关性.csv',index=False)
mcorr_df = pd.DataFrame(mcorr[mcorr['transcycle']> 0.02]['transcycle'])
mcorr_df2 = mcorr_df.rename(dict)
# mcorr_df2.T.to_csv('./data/Q2/第二问与交易周期的相关性.csv',index=False)
## 1.10 筛选与交易周期的相关性
mcorr_df2 =mcorr_df2.sort_values(by='transcycle',ascending=False)
dict = {
 'model':'车型ID',
 'transcycle':'交易周期',
 'trans_category':'交易周期类别',
 'brand':'品牌ID',
 'mileage':'里程',
 'serial':'车系ID',
 'color':'颜色',
 'cityId':'城市ID',
 'oiltype':'燃油类型',
 'carCode':'国标码',
 'seatings':'载客人数',
 'country':'国别',
 'maketype':'厂商类型',
 'displacement':'排量',
 'displacement_category':'排量类别',
 'gearbox':'变速箱',
 'newprice':'新车价',
 'price':'价格',
 'transferCount':'过户次数',
 'modelyear':'年款',
 'price':'价格',
 'barging_times':'降价次数',
 'price_category':'价格级别',
 'mileage_category':'里程类别',
 'tradeTime_year':'展销年份',
 'registerDate_year':"注册年份",
 'registerDate_month':"注册月份",
 'licenseDate_month':'上牌月份',
 'licenseDate_year':"上牌时间"}

mcorr = df_a.corr(method='spearman').abs()
correlation = mcorr[mcorr['transcycle']>0.02]['transcycle']
del correlation['transcycle']
del correlation['trans_category']
col = list(correlation)
idx = list(correlation.index)
map_df = pd.DataFrame(idx,columns=['f'])
map_df = map_df['f'].map(dict)
row = list(map_df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pylab import mpl 
mpl.rcParams['font.sans-serif'] = ['STSong'] # 指定默认字体 
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] =['STSong']
import warnings
warnings.filterwarnings('ignore')
dataset = pd.DataFrame(data=col,index=row,columns=['影响交易周期的关键因素'])
radar_labels=dataset.index
nAttr=dataset.shape[0]
data=dataset.values #数据值
data_labels=dataset.columns
# 设置角度
angles=np.linspace(0,2*np.pi,nAttr,
                   endpoint= False)
data=np.concatenate((data, [data[0]])) 
angles=np.concatenate((angles, [angles[0]]))
# 设置画布
fig=plt.figure(facecolor="white",figsize=(10,6))
plt.subplot(111, polar=True)
# 绘图
plt.plot(angles,data,'o-',
         linewidth=1.5, alpha= 0.2)
# 填充颜色
plt.fill(angles,data, alpha=0.25) 
plt.thetagrids(angles[:-1]*180/np.pi, 
               radar_labels,1.2,fontsize=15) 
plt.grid(True)
plt.savefig('img/Q2-img/3 影响交易周期的关键因素.png',dpi=300)
plt.show()

## 1.11 存储要回归方程的数据
# df_D[df_D['trans_category']==0].to_csv('./data/Q2/Q2_没卖出20.csv', index=0)
# df_D[df_D['trans_category'] == 1].to_csv('./data/Q2/Q2_一周内卖出20.csv', index=0)
# df_D[df_D['trans_category'] == 2].to_csv('./data/Q2/Q2_一周之后卖出20.csv', index=0)
# df_D.to_csv('./data/Q2/Q2_所有数据.csv', index=0)
## 1。2
## 分析所在城市、展销年份、降价次数和价格 与交易周期的关系
# df_a[['cityId','tradeTime_year','barging_times','trans_category']].to_csv('./data/Q2/交易周期的对应分析.csv',index=False)
# tmpdf = df_a[['cityId','tradeTime_year','barging_times','trans_category']]
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['STsong']

labels = ['没卖出', '第一周内卖出', '第二周后卖出']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, year2020, width, label='2020展销')
rects2 = ax.bar(x + width/2, year2021, width, label='2021展销')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('样本数',fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=15)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('./img/Q2-img/6 展销因素与交易周期关系的柱状图.png',dpi=300)
plt.show()


# 2 聚类
## 2.1 热销车的聚类，K值的确定
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['STSong']
df = pd.read_csv('./data/Q2/Q2_一周内卖出20.csv')
data = np.array(df)

Scores = []  # 存放轮廓系数
SSE = []  # 存放每次结果的误差平方和
for k in range(2, 9):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(data)
    Scores.append(silhouette_score(
        np.array(df), estimator.labels_, metric='euclidean'))
    SSE.append(estimator.inertia_) # estimator.inertia_获取聚类准则的总和
X = range(2,9)
X = range(2, 9)
plt.figure(figsize=(15,5))
plt.subplot(121)
plt.xlabel('k值',fontsize=20)
plt.ylabel('误差平方和',fontsize=20)
plt.plot(X, SSE, 'o-')
plt.subplot(122)
plt.xlabel('k值',fontsize=20)
plt.ylabel('轮廓系数',fontsize=20)
plt.plot(X, Scores, 'o-')

plt.savefig('./img/Q2-img/7 手肘法.png',dpi=300)
plt.show()

## 2.2 随机种子的确定
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] =  ['STSong']
df = pd.read_csv('./data/Q2/Q2_一周内卖出20.csv')
data = np.array(df)

Scores = []  # 存放轮廓系数
for i in range(2000,2025):
    estimator = KMeans(n_clusters=3, random_state=i)  # 构造聚类器
    estimator.fit(data)
    Scores.append(silhouette_score(np.array(df), estimator.labels_, metric='euclidean'))
X = range(2000, 2025)
plt.figure(figsize=(7,5))
plt.xlabel('k值',fontsize=20)
plt.ylabel('轮廓系数',fontsize=20)
plt.plot(X, Scores, 'o-')
plt.xlim(2000, 2025)

plt.savefig('./img/Q2-img/8 随机种子的确定.png',dpi=300)
plt.show()

## 2。3 滞销车的样本分类
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
df1 = pd.read_csv('./data/Q2/Q2_一周内卖出20.csv')
data1 = np.array(df1)

df0 = pd.read_csv('./data/Q2/Q2_没卖出20.csv')
data0 = np.array(df0)

df2 = pd.read_csv('./data/Q2/Q2_一周之后卖出20.csv')
data2 = np.array(df2)

clf0 = KMeans(n_clusters=3, init='k-means++',random_state=2016)
clf1 = KMeans(n_clusters=3, init='k-means++',random_state=2016)
clf2 = KMeans(n_clusters=3, init='k-means++',random_state=2016)
# s = clf.fit(data1)
pred0 = clf0.fit_predict(data0)
pred1 = clf1.fit_predict(data1)
pred2 = clf2.fit_predict(data2)
score0 = silhouette_score(data0, pred0)
score1 = silhouette_score(data1, pred1)
score2 = silhouette_score(data2, pred2)
print(score1,score0,score2)
pca = PCA(n_components=3)  # 输出两维
newData0 = pca.fit_transform(data0)  # 载入N维
newData1 = pca.fit_transform(data1)  # 载入N维
newData2 = pca.fit_transform(data2)  # 载入N维

x1, y1, z1 = [], [], []
x2, y2, z2 = [], [], []
x3, y3, z3 = [], [], []
for index, value in enumerate(pred1):
    if value == 0:
        x1.append(newData1[index][0])
        y1.append(newData1[index][1])
        z1.append(newData1[index][2])
    elif value == 1:
        x2.append(newData1[index][0])
        y2.append(newData1[index][1])
        z2.append(newData1[index][2])
    elif value == 2:
        x3.append(newData1[index][0])
        y3.append(newData1[index][1])
        z3.append(newData1[index][2])
# plt.subplot(132)
plt.figure(figsize=(10, 10))

# #定义坐标轴
ax1 = plt.axes(projection='3d')
ax1.scatter3D(x1, y1, z1,marker='^')
ax1.scatter3D(x2, y2, z2, marker='o',c='r')
ax1.scatter3D(x3, y3, z3, marker='*')
plt.savefig('./img/Q2-img/9 第一周卖出的数据分布三维.png',dpi=300)
plt.show()


x1, y1, z1 = [], [], []
x2, y2, z2 = [], [], []
x3, y3, z3 = [], [], []
for index, value in enumerate(pred2):
    if value == 0:
        x1.append(newData2[index][0])
        y1.append(newData2[index][1])
        z1.append(newData2[index][2])
    elif value == 1:
        x2.append(newData2[index][0])
        y2.append(newData2[index][1])
        z2.append(newData2[index][2])
    elif value == 2:
        x3.append(newData2[index][0])
        y3.append(newData2[index][1])
        z3.append(newData2[index][2])
# #定义坐标轴
plt.figure(figsize=(10, 10))
# plt.subplot(132)
ax1 = plt.axes(projection='3d')
ax1.scatter3D(x1, y1, z1,marker='^')
ax1.scatter3D(x2, y2, z2, marker='o',c='r')
ax1.scatter3D(x3, y3, z3, marker='*')
plt.savefig('./img/Q2-img/10 第二周之后卖出的数据分布三维.png',dpi=300)
plt.show()

x1, y1, z1 = [], [], []
x2, y2, z2 = [], [], []
x3, y3, z3 = [], [], []
for index, value in enumerate(pred0):
    if value == 0:
        x1.append(newData0[index][0])
        y1.append(newData0[index][1])
        z1.append(newData0[index][2])
    elif value == 1:
        x2.append(newData0[index][0])
        y2.append(newData0[index][1])
        z2.append(newData0[index][2])
    elif value == 2:
        x3.append(newData0[index][0])
        y3.append(newData0[index][1])
        z3.append(newData0[index][2])
# #定义坐标轴
# plt.subplot(133)
plt.figure(figsize=(10, 10))
ax1 = plt.axes(projection='3d')
ax1.scatter3D(x1, y1, z1,marker='^')
ax1.scatter3D(x2, y2, z2, marker='o',c='r')
ax1.scatter3D(x3, y3, z3, marker='*')
plt.savefig('./img/Q2-img/11 没卖出的数据分布三维.png',dpi=300)

plt.show()
# 输出三个类别的聚类中心
center_df = pd.DataFrame(clf1.cluster_centers_,columns=df0.columns)
center_df.to_csv('./data/Q2/三个类别的聚类中心.csv',index=0)
df1[pred1 == 0].to_csv('./data/Q2/第一周20/Q2_cluster_week1_1.csv', index=False)
df1[pred1 == 1].to_csv('./data/Q2/第一周20/Q2_cluster_week1_2.csv', index=False)
df1[pred1 == 2].to_csv('./data/Q2/第一周20/Q2_cluster_week1_3.csv', index=False)

df2[pred2 == 0].to_csv('./data/Q2/第二周之后20/Q2_cluster_week2_1.csv', index=False)
df2[pred2 == 1].to_csv('./data/Q2/第二周之后20/Q2_cluster_week2_2.csv', index=False)
df2[pred2 == 2].to_csv('./data/Q2/第二周之后20/Q2_cluster_week2_3.csv', index=False)

df0[pred0 == 0].to_csv('./data/Q2/没卖出20/Q2_cluster_week0_1.csv', index=False)
df0[pred0 == 1].to_csv('./data/Q2/没卖出20/Q2_cluster_week0_2.csv', index=False)
df0[pred0 == 2].to_csv('./data/Q2/没卖出20/Q2_cluster_week0_3.csv', index=False)

# 3 回归分析
## 3。1 多元回归方程
import pandas as pd
import numpy as np
week1_data_1 = pd.read_csv('./data/Q2/第一周20/Q2_cluster_week1_1.csv')
week1_data_2 = pd.read_csv('./data/Q2/第一周20/Q2_cluster_week1_2.csv')
week1_data_3 = pd.read_csv('./data/Q2/第一周20/Q2_cluster_week1_3.csv')

week2_data_1 = pd.read_csv('./data/Q2/第二周之后20/Q2_cluster_week2_1.csv')
week2_data_2 = pd.read_csv('./data/Q2/第二周之后20/Q2_cluster_week2_2.csv')
week2_data_3 = pd.read_csv('./data/Q2/第二周之后20/Q2_cluster_week2_3.csv')


week0_data_1 = pd.read_csv('./data/Q2/没卖出20/Q2_cluster_week0_1.csv')
week0_data_2 = pd.read_csv('./data/Q2/没卖出20/Q2_cluster_week0_2.csv')
week0_data_3 = pd.read_csv('./data/Q2/没卖出20/Q2_cluster_week0_3.csv')
# 由于modelyear、registerDate_year和licenseDate_year共线，只保留，rigerterDate_year
cols = ['brand', 'model', 'mileage', 'color', 'cityId', 'carCode', 'seatings',
       'country', 'maketype', 'displacement', 'gearbox',
       'newprice', 'barging_times', 'registerDate_year',
       'licenseDate_month']
cols2 = ['brand', 'model', 'mileage', 'color', 'cityId', 'carCode', 'seatings',
       'country', 'maketype', 'displacement', 'gearbox',
       'newprice', 'barging_times', 'registerDate_year',
       'licenseDate_month','price']
week1_X1 = week1_data_1[cols]
week1_y1  = week1_data_1['price']
week1_X2 = week1_data_2[cols]
week1_y2  = week1_data_2['price']

week1_X3 = week1_data_3[cols]
week1_y3  = week1_data_3['price']

week2_X1 = week2_data_1[cols]
week2_y1  = week2_data_1['price']
week2_X3 = week2_data_3[cols]
week2_y3  = week2_data_3['price']


week0_X1 = week0_data_1[cols]
week0_y1  = week0_data_1['price']
week0_X3 = week0_data_3[cols]
week0_y3  = week0_data_3['price']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
X_train,X_test,y_train,y_test=train_test_split(week1_X1,week1_y1,test_size=.9,random_state=0)
linreg=LinearRegression()
model=linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)

plt.figure()
plt.plot(range(len(y_pred[100:180])),y_pred[100:180],'b',label="回归价格")
plt.plot(range(len(y_pred[100:180])),y_test[100:180],'r',label="原始定价")
plt.legend(loc="upper right",prop = {'size':15})
# plt.savefig('./img/Q2-img/12 第一周第一个方程.png',dpi=300)
plt.show()

from sklearn import metrics

def MAPE(y_true,y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))
y_true = np.array(y_test)
y_pre = np.array(y_pred)

print('MAE:',metrics.mean_absolute_error(y_true,y_pre))
print('MAPE:',MAPE(y_true,y_pre))
print('RMSE:',np.sqrt(metrics.mean_absolute_error(y_true,y_pre)))
# 第三个方程
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
X_train,X_test,y_train,y_test=train_test_split(week1_X3,week1_y3,test_size=.9,random_state=0)
linreg=LinearRegression()
model=linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)

plt.figure()
plt.plot(range(len(y_pred[100:180])),y_pred[100:180],'b',label="回归价格")
plt.plot(range(len(y_pred[100:180])),y_test[100:180],'r',label="原始定价")
plt.legend(loc="upper right",prop = {'size':15})
# plt.savefig('./img/Q2-img/13 第一周第三个方程.png',dpi=300)
plt.show()
y_true = np.array(y_test)
y_pre = np.array(y_pred)

print('MAE:',metrics.mean_absolute_error(y_true,y_pre))
print('MAPE:',MAPE(y_true,y_pre))
print('RMSE:',np.sqrt(metrics.mean_absolute_error(y_true,y_pre)))
# from statsmodels.formula.api import ols
import statsmodels.api as sm
X= sm.add_constant(week1_X1)
result = sm.OLS(week1_y1,X).fit()
result.summary()

import statsmodels.api as sm
X = sm.add_constant(week1_X3)
result = sm.OLS(week1_y3,X).fit()
result.summary()

import statsmodels.api as sm
X = sm.add_constant(week1_X2)
result = sm.OLS(week1_y2,X).fit()
result.summary()

## 3.2 回归滞销车的方程
model1 = linreg.fit(week1_X1,week1_y1)
model3 = linreg.fit(week1_X3,week1_y3)
# 第一个类别的回归价格
cluter1_price = model1.predict(week2_X1)
x = range(80)
plt.plot(x,np.expm1(cluter1_price[30:110]),color='b',label='回归价格')
plt.plot(x,np.expm1(week2_y1[30:110]),color='r',label='原始定价')
plt.legend(loc="upper right",prop = {'size':15})
plt.rcParams['font.sans-serif'] = ['STSong']
plt.savefig('img/Q2-img/14 第二周第一个方程.png',dpi=300)
# 第三个类别的回归价格
cluter3_price = model3.predict(week2_X3)
x = range(80)
plt.plot(x,np.expm1(cluter3_price[100:180]),color='b',label='回归价格')
plt.plot(x,np.expm1(week2_y3[100:180]),color='r',label='原始价格')
plt.legend(loc="upper right",prop = {'size':15})
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.savefig('img/Q2-img/15 第二周第三个方程.png',dpi=300)

## 3.3 回归没卖出的车辆的价格
model1 = linreg.fit(week1_X1,week1_y1)
model3 = linreg.fit(week1_X3,week1_y3)
# 第一个类别的回归价格
cluter1_price = model1.predict(week0_X1)
x = range(80)
plt.plot(x,np.expm1(cluter1_price[100:180]),color='b',label='回归价格')
plt.plot(x,np.expm1(week0_y1[100:180]),color='r',label='原始价格')
plt.rcParams['font.sans-serif'] = ['STSong']
plt.legend(loc="upper right",prop = {'size':15})
plt.savefig('img/Q2-img/16 没卖出第一个方程.png',dpi=300)
# 第三个类别的回归价格
cluter1_price = model1.predict(week0_X3)
x = range(80)
plt.plot(x,np.expm1(cluter1_price[100:180]),color='b',label='回归价格')
plt.plot(x,np.expm1(week0_y3[100:180]),color='r',label='原始价格')
plt.legend(loc="upper right",prop = {'size':15})
plt.rcParams['font.sans-serif'] = ['STSong']
plt.savefig('img/Q2-img/17 没卖出第三个方程.png',dpi=300)