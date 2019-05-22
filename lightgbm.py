#!/usr/bin/env python
# coding: utf-8


# 重新编写lgb模型
import pandas as pd
test_sampledf_origin= pd.read_csv('E:\\work\\tencent\\algo.qq.com_641013010_testa\\testA\\sample\\sample_total_no0_typeStatisticAll_1hot.csv'
                           ,sep='\t',low_memory=False)

test_sampledf_origin=pd.DataFrame(test_sampledf_origin)
test_sampledf_origin=test_sampledf_origin.reset_index(drop=True)
origin_features=['广告id','创建时间','素材尺寸','广告行业id','商品类型','商品id','广告账号id','投放时间','人群定向',
                 '出价','星期','曝光日期','曝光次数']
test_sampledf=test_sampledf_origin[origin_features]
test_sampledf['创建时间']=test_sampledf_origin['创建时间']//1000000  # -20190000
test_sampledf['曝光日期']=test_sampledf_origin['曝光日期'] -20190000
test_sampledf['istest']=False
del test_sampledf_origin


# In[5]:


import time
Btest_sample_newdf= pd.read_csv('E:\\work\\tencent\\algo.qq.com_641013010_testa\\testA\\Btest_sample_new_post.dat ',sep='\t')

Btest_sample_newdf.drop(['样本id'],axis=1,inplace=True)
Btest_sample_newdf['曝光日期']=Btest_sample_newdf['创建时间'].apply(lambda x: 321 if (x<20190321) else (x+1)-20190000 )
#Btest_sample_newdf['创建时间']=Btest_sample_newdf['创建时间']-20190000
Btest_sample_newdf['创建时间']=Btest_sample_newdf['创建时间'].apply(lambda x: 20190217 if x<20190217 else x)
Btest_sample_newdf['曝光次数']=-999
Btest_sample_newdf=Btest_sample_newdf[origin_features]
Btest_sample_newdf['istest']=True


all_test_sampledf=test_sampledf.append(Btest_sample_newdf)
#all_test_sampledf=test_sampledf.copy()


# In[10]:


from sklearn.preprocessing import LabelEncoder
all_test_sampledf['人群定向']=pd.DataFrame(LabelEncoder().fit(all_test_sampledf['人群定向']).
                                   transform(all_test_sampledf['人群定向']).reshape(-1,1))

all_test_sampledf['投放时间']=pd.DataFrame(LabelEncoder().fit(all_test_sampledf['投放时间']).
                                   transform(all_test_sampledf['投放时间']).reshape(-1,1))
all_test_sampledf['商品id']=all_test_sampledf['商品id'].apply(lambda x: 999999 if x==-1 else x )
all_test_sampledf['人群定向']=all_test_sampledf['人群定向'].astype('int32')
all_test_sampledf['已投放天数']=20190000+all_test_sampledf['曝光日期']-all_test_sampledf['创建时间']


# In[14]:


Btest_sample_newdf.shape


# In[12]:


test_sampledf=all_test_sampledf[all_test_sampledf['istest']==False]
test_sampledf.drop(['istest'],axis=1,inplace=True)
Btest_sample_newdf=all_test_sampledf[all_test_sampledf['istest']==True]
Btest_sample_newdf.drop(['istest'],axis=1,inplace=True)

x_train=pd.DataFrame()
x_train=test_sampledf[(test_sampledf['曝光日期'] !=319)] # &(test_sampledf['曝光日期'] !=0)  &(data_tr['曝光次数'] !=0)
y_train=x_train.pop('曝光次数')
#x_train.pop('出价')
#x_train.pop('曝光日期')

x_test=test_sampledf[(test_sampledf['曝光日期'] ==319) ] #&(test_sampledf['曝光日期'] !=0) & (data_tr['曝光次数'] !=0)
#x_test.pop('出价')
#x_test.pop('曝光日期')
y_test=x_test.pop('曝光次数')


# In[31]:


x_test.shape


# In[32]:


import lightgbm as lgb
# 将参数写成字典下形式
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression', # 目标函数
    'metric': {'l2', 'mse'},  # 评估函数
    #'eval_metric':'mape',
    'max_depth':-1,#树的最大深度
    'num_leaves': 1000,   # 叶子节点数
    'learning_rate': 0.04,  # 学习速率
    'feature_fraction': 0.9, # 建树的特征选择比例
    'bagging_fraction': 0.8, # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1, # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    'free_raw_data': False,
    #'min_child_weight':50, 
   # 'random_state':2018, 
    'n_jobs':-1
}
# 创建成lgb特征的数据集格式
lgb_train = lgb.Dataset(x_train, y_train, feature_name=['广告id','创建时间','素材尺寸','广告行业id','商品类型','商品id','广告账号id','投放时间','人群定向',
                 '出价','星期','曝光日期','已投放天数'], categorical_feature=['广告id','创建时间','素材尺寸','广告行业id','商品类型','商品id','广告账号id','投放时间','人群定向',
                 '星期','曝光日期','已投放天数']) # 将数据保存到LightGBM二进制文件将使加载更快
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train, feature_name=['广告id','创建时间','素材尺寸','广告行业id','商品类型','商品id','广告账号id','投放时间','人群定向',
                 '出价','星期','曝光日期','已投放天数'], categorical_feature=['广告id','创建时间','素材尺寸','广告行业id','商品类型','商品id','广告账号id','投放时间','人群定向',
                 '星期','曝光日期','已投放天数'])  # 创建验证数据
 
num_round = 10
#lgb.cv(params, train_data, num_round, nfold=5)

print('Start training...')
# 训练 cv and train
gbm = lgb.train(params,lgb_train,num_boost_round=100,valid_sets=lgb_eval,early_stopping_rounds=90) # 训练数据需要参数列表和数据集
#gbm=lgb.cv(params, lgb_train, num_round, nfold=5)

print('Save model...') 
 
gbm.save_model('model.txt')   # 训练后保存模型到文件
 
print('Start predicting...')
# 预测数据集
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration) #如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
# 评估模型
# 测试分数 (0-2之间,越小越好)

import numpy as np
def getSMAPEScore (y_true, y_pred):    
    # 预测值小于0如何处理 y_pred=np.abs(y_pred)
    #y_pred=(np.abs(y_pred) + y_pred) / 2
    y_pred=np.abs(y_pred)
    SMAPE=2.0 * np.mean(np.abs(y_pred - y_true) / (y_pred+ y_true)) 
    return  SMAPE,40*(1-SMAPE/2)
print('The smape of prediction is:', getSMAPEScore(y_test, y_pred)) # 计算真实值和预测值之间的均方根误差


# In[25]:


def getSMAPEScore (y_true, y_pred):    
    # 预测值小于0如何处理 y_pred=np.abs(y_pred)
    # y_pred=(np.abs(y_pred) + y_pred) / 2
    y_pred=np.abs(y_pred)
    SMAPE=2.0 * np.mean(np.abs(y_pred - y_true) / (y_pred+ y_true)) 
    return  SMAPE,40*(1-SMAPE/2)
print('The smape of prediction is:', getSMAPEScore(y_test, y_pred)) # 计算真实值和预测值之间的均方根误差


# In[33]:


# 将线上测试集合并进来一起训练,预测结果
# 预测数据集
x_online_test=Btest_sample_newdf.drop(['曝光次数'],axis=1)

y_pred = gbm.predict(x_online_test, num_iteration=gbm.best_iteration) #如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测


# In[42]:


B_sample.shape


# In[53]:


B_sample= pd.read_csv('E:\\work\\tencent\\algo.qq.com_641013010_testa\\testA\\Btest_sample_new.dat ',sep='\t',
                             header=None,names=['样本id','广告id','创建时间','素材尺寸','广告行业id'
                                                           ,'商品类型','商品id','广告账号id','投放时间','人群定向'
                                                            ,'出价'])
B_sample = pd.DataFrame(B_sample)
B_sample['曝光次数']=pd.DataFrame(y_pred).reset_index(drop=True)[0].round(decimals=4)

B_sample['曝光次数']=B_sample['曝光次数'].apply(lambda x: abs(x))


# In[54]:


B_sample[B_sample['曝光次数']<0].sort_values(by=['广告id','出价'])


# In[55]:


# 不符合单调性,所以进行变动
B_sample['出价排序']=B_sample['出价'].groupby(B_sample['广告id']).rank(ascending=1,method='dense')
B_sample.reset_index()


B_sample_base=B_sample.sort_values(by=['广告id','曝光次数'],ascending=False)
#[B_sample['出价排序']==5]
B_sample_base=B_sample_base[['广告id','曝光次数'] ]
B_sample_base=B_sample_base.groupby(by=['广告id'])['曝光次数'].min().reset_index()

B_sample=pd.merge(B_sample,B_sample_base,how='left',left_on='广告id',right_on='广告id')
B_sample['曝光次数']=B_sample.apply(lambda x: (x['曝光次数_y']+x['出价排序']/10000),axis=1)
B_sample
# Btest_sample_1=B_sample.sort_values(by=['广告id','出价'])


# In[56]:


#检查单调性
number=0;
for name,group in B_sample.groupby(by=['广告id']):
    group=group.sort_values(by=['出价']).reset_index(drop=True)
    list=group['曝光次数'].values    
    for i in range(len(list)-1):
        if list[i+1]<list[i]:
            number=number+1
            print(group.head()[['广告id','出价','曝光次数']])
            break
            #print("异常数据:广告id为"+str(name)+",其中第"+str(i)+"个数据的曝光小于第"+str(i+1)+"个数据")      
print(str(number))


# In[57]:


#后处理

B_sample['出价排序']=B_sample['出价'].groupby(B_sample['广告id']).rank(ascending=1,method='dense')
B_sample['new曝光']=B_sample.apply(lambda x: (round(x['曝光次数'])+x['出价排序']/10000)  ,axis=1).round(decimals=4)

#不在投放时间的广告样本id 
list=[ 118,  2149,  2301,  3672,  4091,  4332,  4567,  4710,  5302,        6427,  6448,  6588,  6688,  6967,  7264,  7521,  8723,  8832,        8901,  9285,  9403,  9907, 10084, 10268, 10650, 11167, 12083,       13285, 13500, 13539, 13797, 13951, 14089, 15127, 15382, 15624,       15839, 16316, 16633, 16688, 17658, 18009, 18075, 18177, 18835,       19491, 20040, 20617, 20642, 21473, 22139, 22229, 23203, 24952,       25197, 26594, 27125, 27680, 27784, 29265, 30155, 30382, 33647,       33758, 34032, 34465, 34721, 35070, 35520, 36413, 36822, 36906,       37025, 37630, 37736]
B_sample['new曝光']=B_sample.apply(lambda x: x['出价排序']/10000 if x['样本id'] in list else x['new曝光'],axis=1)
B_sample['new曝光']=B_sample['new曝光']
B_sample.reset_index().round(decimals=4)
B_sample.to_csv("E:\\work\\tencent\\algo.qq.com_641013010_testa\\testA\\submission\\0520\\submission.csv",sep=",",
                    index=False,encoding="utf-8",header=False,columns=["样本id","new曝光"])


# In[64]:


B_sample['new曝光1'].describe()
