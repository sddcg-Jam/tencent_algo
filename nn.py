#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 对训练集进行离散特征统计,除了计算sum,还需要计算 mean media 等用于训练
import pandas as pd
import numpy as np


# In[2]:


# 取出训练集
#广告id	素材尺寸	广告行业id	商品类型	商品id	广告账号id	出价_x	曝光日期	曝光次数  投放时间	人群定向
# names=['广告id','素材尺寸','广告行业id','商品类型','商品id','广告账号id','出价_x','曝光日期','曝光次数']
test_samplefile = pd.read_csv('E:\\work\\tencent\\algo.qq.com_641013010_testa\\testA\\sample\\sample_total_no0.csv',sep='\t')
test_sampledf = pd.DataFrame(test_samplefile)
test_sampledf.astype('str')

test_sampledf['出价']=test_sampledf['出价'].astype('float')

test_sampledf['曝光次数']=test_sampledf['曝光次数'].astype('int64')
test_sampledf=test_sampledf[test_sampledf['曝光次数']!=0]

test_sampledf['曝光日期']=test_sampledf['曝光日期'].astype('int64')

#转换成星期几 
test_sampledf['日期']=test_sampledf['曝光日期'].apply(lambda x: (x-20190218+1) if (x<20190301) else (x-20190301+11+1))
test_sampledf['星期']=test_sampledf['日期'].mod(7)


# In[3]:


ad_id_stat_s=test_sampledf.groupby(by=['广告id'])['曝光次数'].sum()  #广告id 为index
ad_size_stat_s=test_sampledf.groupby(by=['素材尺寸'])['曝光次数'].sum()  
ad_industry_s=test_sampledf.groupby(by=['广告行业id'])['曝光次数'].sum()  
good_type_s=test_sampledf.groupby(by=['商品类型'])['曝光次数'].sum()  
goods_id_s=test_sampledf.groupby(by=['商品id'])['曝光次数'].sum()  
ad_account_id_s=test_sampledf.groupby(by=['广告账号id'])['曝光次数'].sum()  
day_week_s=test_sampledf.groupby(by=['星期'])['曝光次数'].sum()  
exp_time_s=test_sampledf.groupby(by=['投放时间'])['曝光次数'].sum()  
exp_people_s=test_sampledf.groupby(by=['人群定向'])['曝光次数'].sum()  


# In[4]:


ad_id_stat_m=test_sampledf.groupby(by=['广告id'])['曝光次数'].mean()  #广告id 为index
ad_size_stat_m=test_sampledf.groupby(by=['素材尺寸'])['曝光次数'].mean()  
ad_industry_m=test_sampledf.groupby(by=['广告行业id'])['曝光次数'].mean()  
good_type_m=test_sampledf.groupby(by=['商品类型'])['曝光次数'].mean()  
goods_id_m=test_sampledf.groupby(by=['商品id'])['曝光次数'].mean()  
ad_account_id_m=test_sampledf.groupby(by=['广告账号id'])['曝光次数'].mean()  
day_week_m=test_sampledf.groupby(by=['星期'])['曝光次数'].mean()
exp_time_m=test_sampledf.groupby(by=['投放时间'])['曝光次数'].mean()  
exp_people_m=test_sampledf.groupby(by=['人群定向'])['曝光次数'].mean()


# In[5]:


ad_id_stat_media=test_sampledf.groupby(by=['广告id'])['曝光次数'].median()  #广告id 为index
ad_size_stat_media=test_sampledf.groupby(by=['素材尺寸'])['曝光次数'].median()  
ad_industry_media=test_sampledf.groupby(by=['广告行业id'])['曝光次数'].median()  
good_type_media=test_sampledf.groupby(by=['商品类型'])['曝光次数'].median()  
goods_id_media=test_sampledf.groupby(by=['商品id'])['曝光次数'].median()  
ad_account_id_media=test_sampledf.groupby(by=['广告账号id'])['曝光次数'].median()  
day_week_media=test_sampledf.groupby(by=['星期'])['曝光次数'].median()
exp_time_media=test_sampledf.groupby(by=['投放时间'])['曝光次数'].median()  
exp_people_media=test_sampledf.groupby(by=['人群定向'])['曝光次数'].median()  


# In[6]:


test_sampledf_origin= pd.read_csv('E:\\work\\tencent\\algo.qq.com_641013010_testa\\testA\\sample\\sample_total_no0_typeStatisticAll_1hot.csv'
                           ,sep='\t',low_memory=False)
test_sampledf_origin=pd.DataFrame(test_sampledf_origin)
test_sampledf_origin=test_sampledf_origin.reset_index(drop=True)


# In[7]:


# 测试分数 (0-2之间,越小越好)  19.418521650188318
def getSMAPEScore (y_true, y_pred):    
    #print(y_true)
    # 预测值小于0如何处理 y_pred=np.abs(y_pred)
    y_pred=(np.abs(y_pred) + y_pred) / 2
    SMAPE=2.0 * np.mean(np.abs(y_pred - y_true+0.0001) / (y_pred+ y_true+0.0001)) 
    return   SMAPE,40*(1-SMAPE/2)  #SMAPE #
# 损失函数
def mean_squared_error3(y_true, y_pred):
    return 2*K.mean(K.abs(y_pred-y_true)/(y_pred+y_true),axis=-1)


# In[8]:


#'商品id_m',
attrs=['广告id_m','素材尺寸_m','广告行业id_m','商品类型_m','广告账号id_m','星期_m','投放时间_m','人群定向_m',
#       '广告id_median','素材尺寸_median','广告行业id_median','商品类型_median','商品id_median','广告账号id_median',
#       '星期_median','投放时间_median','人群定向_median',
      '出价','曝光次数','曝光日期']
# 
test_sampledf=test_sampledf_origin[attrs]

# 划分训练集和验证集
x_train=pd.DataFrame()
x_train=test_sampledf[(test_sampledf['曝光日期'] !=20190319) ] # &(test_sampledf['曝光日期'] !=0)


x_train.drop(['曝光日期'],axis=1,inplace=True)
#x_train.drop(['出价'],axis=1,inplace=True)
x_train=np.log1p(x_train)
y_train=x_train['曝光次数']
x_train.drop(['曝光次数'],axis=1,inplace=True)
x_train=x_train.reset_index(drop=True)
#categorical_feature=['广告id','素材尺寸','广告行业id','商品类型','商品id','广告账号id','星期','曝光日期']
#x_train[categorical_feature]=test_sampledf_origin[test_sampledf_origin['曝光日期'] !=20190319][categorical_feature].reset_index(drop=True)

x_test=test_sampledf[(test_sampledf['曝光日期'] ==20190319)] #&(test_sampledf['曝光日期'] !=0)
x_test.drop(['曝光日期'],axis=1,inplace=True)
#x_test.drop(['出价'],axis=1,inplace=True)
x_test=np.log1p(x_test)
y_test=x_test['曝光次数']
x_test.drop(['曝光次数'],axis=1,inplace=True)
x_test=x_test.reset_index(drop=True)
#x_test[categorical_feature]=test_sampledf_origin[test_sampledf_origin['曝光日期']==20190319][categorical_feature].reset_index(drop=True)


# In[9]:


x_train.shape


# In[379]:


import tensorflow as tf
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

#19.543254691924503

seed=7
np.random.seed(seed)
# 建立模型
optimizer='adam'
init='normal'
model=Sequential()
model.add(Dense(units=22,activation='relu',input_dim=x_train.shape[1],kernel_initializer=init))
#构建更多的隐藏层  
model.add(Dense(units=12,activation='sigmoid',kernel_initializer=init))
# model.add(Dense(units=8,activation='relu',kernel_initializer=init))
# model.add(Dense(units=4,activation='relu',kernel_initializer=init))
model.add(Dense(units=1,kernel_initializer=init)) #输出层不需要进行激活函数,预测回归的话unit=1
# 编译模型
model.compile(optimizer=optimizer,loss=mean_squared_error3,metrics=['acc'])#,optimizer=tf.train.GradientDescentOptimizer(0.03)
model.fit(x_train.values,y_train.values,epochs=100,batch_size=1600,verbose=1, validation_split=0.3, shuffle=True)

#model.evaluate(x_test.get_values(), y_test.get_values())
temp=model.predict(x_test.values, batch_size=10)
y_pred=np.expm1(temp)
#y_pred=temp
y_true=np.expm1(y_test)
#y_true=y_test
smape,smapescore=getSMAPEScore(y_true.values,y_pred)
print("本次训练结果的smape部分总分为"+str(smapescore))


# In[10]:


import tensorflow as tf
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

#19.57695999836607 用中位值
# 19.581931417387246 用中位值和平均值
#用平均值 19.577111159881202

# 19.45284459873507,  8-11-2-1,12次迭代,1600batch
#19.5251400337353(1.023746061164272) ,6-11-2-1,12次迭代,1600batch,  val_loss: 0.4500
# 拿掉广告id,19.5332899053006	(1.0233355047349701) ,6-11-2-1,12次迭代,1600batch,  val_loss: 0.4710
# 拿掉'商品id_m', 19.53698244270679	(1.0231508778646605)  val_loss: 0.4757
# 平均值特征拿掉'广告id_m','商品id_m', 带着出价 19.617445423851578	(1.0191277288074212)  6-11-2-1,12次迭代,1600batch, val_loss: 0.4825
# 去掉走后一层的relu 其余同上  19.61867090066771	(1.0190664549666146) val_loss: 0.4776 
# 加回广告id,8-11-2-1层数,19.623944250028458	(1.018802787498577) val_loss: 0.4634   B榜更新后第一次提交 82.7053
seed=7
np.random.seed(seed)
# 建立模型
optimizer='adam'
init='normal'
model=Sequential()
model.add(Dense(units=8,activation='relu',input_dim=x_train.shape[1],kernel_initializer=init))
#构建更多的隐藏层  19.453021334225507线上降分,不理想. 18.913335197714552对应线上最高分83.04 loss0.3765
model.add(Dense(units=11,activation='relu',kernel_initializer=init))
model.add(Dense(units=2,kernel_initializer=init))
model.add(Dense(units=1,kernel_initializer=init)) #输出层不需要进行激活函数,预测回归的话unit=1
# 编译模型
model.compile(optimizer=optimizer,loss=mean_squared_error3,metrics=['acc'])#,optimizer=tf.train.GradientDescentOptimizer(0.03)
model.fit(x_train.values,y_train.values,epochs=12,batch_size=1600,verbose=1, validation_split=0.3, shuffle=True)

#model.evaluate(x_test.get_values(), y_test.get_values())
temp=model.predict(x_test.values, batch_size=1600)
y_pred=np.expm1(temp)
#y_pred=temp
y_true=np.expm1(y_test)
#y_true=y_test
smape,smapescore=getSMAPEScore(y_true.values,y_pred)
print("本次训练结果的smape部分总分为"+str(smapescore)+"\t("+str(smape)+")")


# In[17]:


tx_btest_sampledf=pd.read_csv('E:\\work\\tencent\\algo.qq.com_641013010_testa\\testA\\update_Btest_sample_post.dat',header=0,sep='\t')
attrs=['广告id_m','素材尺寸_m','广告行业id_m','商品类型_m','商品id_m','广告账号id_m','星期_m','投放时间_m','人群定向_m',
      '广告id_median','素材尺寸_median','广告行业id_median','商品类型_median','商品id_median','广告账号id_median',
      '星期_median','投放时间_median','人群定向_median']
#     , '出价']
attrs=['广告id_m','素材尺寸_m','广告行业id_m','商品类型_m','广告账号id_m','星期_m','投放时间_m','人群定向_m', '出价']
#     , '出价']
tx_btest_sampledf_origin=tx_btest_sampledf[attrs]
tx_btest_sampledf=np.log1p(tx_btest_sampledf_origin)
# 预测结果
temp=model.predict(tx_btest_sampledf.values, batch_size=10)
temp=np.expm1(temp)


# In[18]:


# 纯模型线上 82.9933分
Btest_sample_newdf= pd.read_csv('E:\\work\\tencent\\algo.qq.com_641013010_testa\\testA\\Btest_sample_new.dat ',sep='\t',
                             header=None,names=['样本id','广告id','创建时间','素材尺寸','广告行业id'
                                                           ,'商品类型','商品id','广告账号id','投放时间','人群定向'
                                                            ,'出价'])
Btest_sample_newdf = pd.DataFrame(Btest_sample_newdf)
Btest_sample_newdf['曝光次数']=pd.DataFrame(temp).reset_index(drop=True)[0].round(decimals=4)
Btest_sample_newdf.reset_index()

Btest_sample_newdf_1=Btest_sample_newdf.sort_values(by=['广告id','出价'])


# In[19]:


#检查单调性
number=0;
for name,group in Btest_sample_newdf_1.groupby(by=['广告id']):
    group=group.sort_values(by=['出价']).reset_index(drop=True)
    list=group['曝光次数'].values    
    for i in range(len(list)-1):
        if list[i+1]<list[i]:
            number=number+1
            print(group.head()[['广告id','出价','曝光次数']])
            break
            #print("异常数据:广告id为"+str(name)+",其中第"+str(i)+"个数据的曝光小于第"+str(i+1)+"个数据")      
print(str(number))


# In[458]:


# 去掉出价出价特征,需要手动维护单调性


# In[20]:


#后处理
#82.9933到83.0338  四舍五入 后处理再做单调性 rank/10000,  round()函数
# 尝试*2后四舍五入,发现线上分数降低了将近2分

Btest_sample_newdf['出价排序']=Btest_sample_newdf['出价'].groupby(Btest_sample_newdf['广告id']).rank(ascending=1,method='dense')
Btest_sample_newdf['new曝光']=Btest_sample_newdf.apply(lambda x: (round(x['曝光次数'])+x['出价排序']/10000)  ,axis=1).round(decimals=4)

#不在投放时间的广告样本id , 83.02提分 83.028
list=[ 118,  2149,  2301,  3672,  4091,  4332,  4567,  4710,  5302,        6427,  6448,  6588,  6688,  6967,  7264,  7521,  8723,  8832,        8901,  9285,  9403,  9907, 10084, 10268, 10650, 11167, 12083,       13285, 13500, 13539, 13797, 13951, 14089, 15127, 15382, 15624,       15839, 16316, 16633, 16688, 17658, 18009, 18075, 18177, 18835,       19491, 20040, 20617, 20642, 21473, 22139, 22229, 23203, 24952,       25197, 26594, 27125, 27680, 27784, 29265, 30155, 30382, 33647,       33758, 34032, 34465, 34721, 35070, 35520, 36413, 36822, 36906,       37025, 37630, 37736]
Btest_sample_newdf['new曝光']=Btest_sample_newdf.apply(lambda x: x['出价排序']/10000 if x['样本id'] in list else x['new曝光'],axis=1)
Btest_sample_newdf.reset_index()
Btest_sample_newdf.to_csv("E:\\work\\tencent\\algo.qq.com_641013010_testa\\testA\\submission\\0520\\submission.csv",sep=",",
                    index=False,encoding="utf-8",header=False,columns=["样本id","new曝光"])


# In[21]:


Btest_sample_newdf['new曝光'].mean()


# In[22]:


# # 平均值在8-15之间
# Btest_sample_newdf['new曝光1']=Btest_sample_newdf.apply(lambda x: x['new曝光']+5 
#                                                       if ((x['new曝光']<15) & (x['new曝光']> 1)) else x['new曝光'],axis=1)

# Btest_sample_newdf.to_csv("E:\\work\\tencent\\algo.qq.com_641013010_testa\\testA\\submission\\0516\\submission.csv",sep=",",
#                     index=False,encoding="utf-8",header=False,columns=["样本id","new曝光1"])


# In[ ]:





# In[ ]:





# In[ ]:




