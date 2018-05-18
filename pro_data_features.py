# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 13:33:34 2018

@author: dashen
"""

import pandas as pd
import pdb
import random
import time
import numpy as np
import gc

def classfily_data(data):
    #global is_0,is_not_0
    data_1 = list(set(data[data['Y'] !=0]['TERMINALNO']))
    #is_not_0 = len(data_1)
    data_0 = list(set(data[data['Y'] ==0]['TERMINALNO']))
    #is_0  = len(data_0)
    #data_0 = random.sample(data_0,5*len(data_1)) #这里本来使用  len(data_1) 的倍数来进行随机抽样 现在直接变为取整数
    data_0 = data_0+data_1
    # random.shuffle(data_0)
    return data_0,data_1

def feature_creat(data):
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np
    poly = PolynomialFeatures(degree=3)
    creat_data = poly.fit_transform(data)
    return creat_data



def pro_data(filedir,Y=True):
    #data_train = pd.DataFrame()
    data = pd.read_csv(filedir)
    if Y:
        id_data,data_1= classfily_data(data)
        data = data.loc[data['TERMINALNO'].isin(id_data)]
    else:
        id_data = list(set(data['TERMINALNO']))
    #pdb.set_trace()
    
    
    #data['TIME'] = list(map(lambda t: time.strftime("%H:%M:%S",time.localtime(int(t))),data['TIME']))
   # data['MIN'] = list(map(lambda t: int(time.strftime("%M",time.localtime(int(t)))),data['TIME']))
    data['TIME'] = list(map(lambda t: int(time.strftime("%H",time.localtime(int(t)))),data['TIME']))
    

    user_id = id_data
    new_columns = []
    for i in range(1,50):
        new_columns.append('id_min_'+str(i))
        new_columns.append('id_hour_'+str(i))
        #new_columns.append('id_longitude_'+str(i))
        #new_columns.append('id_latitude_'+str(i))
        new_columns.append('id_high_'+str(i))
        new_columns.append('id_speed_mean_'+str(i))
        new_columns.append('id_speed_std_'+str(i))
        new_columns.append('id_speed_max_'+str(i))
        new_columns.append('id_speed_min_'+str(i))
        new_columns.append('id_total_way_'+str(i))
    new_columns.append('call_0')
    new_columns.append('call_1')
    new_columns.append('call_2')
    new_columns.append('call_3')
    new_columns.append('call_4')
    new_columns.append('Y')
    new_columns.append('Y_0_1')
    new_data = pd.DataFrame(index = user_id,columns=list(new_columns))
    #new_data.to_csv('model/new_train.csv')
    
    
    df_speed = data.groupby(['TERMINALNO','TRIP_ID'])['SPEED']
    df_time  = data.groupby(['TERMINALNO','TRIP_ID'])['TIME']
    df_local_1 = data.groupby(['TERMINALNO','TRIP_ID'])['LONGITUDE']
    #df_local_2 = data.groupby(['TERMINALNO','TRIP_ID'])['LATITUDE']
    df_heigh = data.groupby(['TERMINALNO','TRIP_ID'])['HEIGHT']
    
    
    
    df_time_min  = df_time.size().unstack()
    df_time_hour = df_time.median().unstack()
    #df_time_min=df_time_min.dropna(axis=1, how='any', thresh=None, subset=None, inplace=False)
    
    df_speed_mean = df_speed.mean().unstack()
    df_speed_max  = df_speed.max().unstack()
    df_speed_min  = df_speed.mad().unstack()
    df_speed_std  = df_speed.std().unstack()
    
    
    df_local_1_median = df_local_1.median().unstack()
    #df_local_2_median = df_local_2.median().unstack()
    df_heigh_median   = df_heigh.median().unstack()
    #data=data.dropna(axis=1, how='all', thresh=None, subset=None, inplace=False)
    
    
    
    
    

    
    for i in user_id:
        lenth = int(len(data[data['TERMINALNO'] ==i]))
        for j in range(1,50):
            
            new_data.loc[i,:]['id_min_'+str(j)] = df_time_min.loc[i,j]
            new_data.loc[i,:]['id_hour_'+str(j)] = df_time_hour.loc[i,j]
            #if len(data[data['TERMINALNO'] ==i][data['TRIP_ID'] == j]):
             #   new_data.loc[i,:]['id_hour_'+str(j)] = data[data['TERMINALNO'] ==i][data['TRIP_ID'] == j]['TIME'].iloc[0]
           # else:
              #  new_data.loc[i,:]['id_hour_'+str(j)] =0
                
            #new_data.loc[i,:]['id_longitude_'+str(j)] = df_local_1_median.loc[i,j]
            #new_data.loc[i,:]['id_latitude_'+str(j)] = df_local_2_median.loc[i,j]
            new_data.loc[i,:]['id_high_'+str(j)] = df_heigh_median.loc[i,j]
            new_data.loc[i,:]['id_speed_mean_'+str(j)] = df_speed_mean.loc[i,j]
            new_data.loc[i,:]['id_speed_std_'+str(j)] = df_speed_std.loc[i,j]
            new_data.loc[i,:]['id_speed_max_'+str(j)] = df_speed_max.loc[i,j]
            new_data.loc[i,:]['id_speed_min_'+str(j)] = df_speed_min.loc[i,j]
            new_data.loc[i,:]['id_total_way_'+str(j)] = (df_time_min.loc[i,j])*(df_speed_mean.loc[i,j])
        new_data.loc[i,:]['call_0']       = np.sum(data[data['TERMINALNO'] ==i]['CALLSTATE']==0)/lenth
        new_data.loc[i,:]['call_1']       = np.sum(data[data['TERMINALNO'] ==i]['CALLSTATE']==1)/lenth
        new_data.loc[i,:]['call_2']       = np.sum(data[data['TERMINALNO'] ==i]['CALLSTATE']==2)/lenth
        new_data.loc[i,:]['call_3']       = np.sum(data[data['TERMINALNO'] ==i]['CALLSTATE']==3)/lenth
        new_data.loc[i,:]['call_4']       = np.sum(data[data['TERMINALNO'] ==i]['CALLSTATE']==4)/lenth
        if Y == True:
            new_data.loc[i,:]['Y'] = data[data['TERMINALNO'] ==i]['Y'].iloc[0]
            if new_data.loc[i,:]['Y'] == 0:
                new_data.loc[i,:]['Y_0_1'] = 0
            else:
                new_data.loc[i,:]['Y_0_1'] = 1
            
            
            
    new_data = new_data.fillna(-1)
    #new_data = feature_creat(new_data)
    #new_data.to_csv('model/new_train.csv')
    del data
    gc.collect()
    if Y == True:
        he = abs(new_data.corr()['Y'])> 0.001
        global keys 
        keys = []
        j=0
        for num in new_data.columns:
            if he[num]:
                keys.append(num)
                j +=1
        print(keys)

    new_data = new_data[keys]
    
    if Y == True:
        data_1_train = pd.DataFrame(new_data.loc[data_1,:])
        for p in range(0):  #这是4
            new_data = pd.concat([new_data,data_1_train],axis = 0)
        
        return new_data,data_1_train
    else:
        return new_data
        
    #new_data.to_csv('model/new_train.csv')
    #pdb.set_trace()
    #print("successed to product new_train.csv")
def train_classify(train_data,test_data):
    train_data = train_data.sample(frac=1) 
   
    
    test_data.iloc[:,:-2] =(test_data.iloc[:,:-2]-train_data.iloc[:,:-2].mean())/(train_data.iloc[:,:-2].std())
    train_data.iloc[:,:-2]=(train_data.iloc[:,:-2]-train_data.iloc[:,:-2].mean())/(train_data.iloc[:,:-2].std())
    
    test_file = pd.DataFrame(index = None,columns=list(["Id", "Pred"]))
    test_file["Id"] = test_data.index
    
    test_data = test_data.as_matrix()
    train_data= train_data.as_matrix()
    #random.shuffle(train_data)
     
    x_train = train_data[:,:-2]
    y_train = train_data[:,-2]
    x_test  = test_data[:,:-2]
    
    import lightgbm as lgb
    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.01, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
# 训练、预测
    model_lgb.fit(x_train, y_train)
    y_pred_xgb = model_lgb.predict(x_test)
    #。。。。。。。。。。
    
   
    test_file["Pred"] = y_pred_xgb
    
    print(test_file.iloc[:5,:])
    test_file.to_csv('model/test.csv',index = None)
    
    #data_1 = test_file["Id"][test_file["Pred"] ==1]
    #data_0 = test_file["Id"][test_file["Pred"] ==0]
    #return data_1,data_0    



def train_lr(train_data,test_data,train_regress):
    y_mean = train_regress['Y'].mean()
    y_std = train_regress['Y'].std()
    
    #test_file = pd.DataFrame(index = None,columns=list(["Id", "Pred"]))
    #test_file["Id"] = test_data.index
    
    data_1,data_0 = train_classify(train_data,test_data)
    
    test_data_1 = test_data.loc[data_1,:]
    test_data_0 = test_data.loc[data_0,:]
    
    test_file_1 = pd.DataFrame(index = None,columns=list(["Id", "Pred"]))
    test_file_1["Id"] = test_data_1.index
    test_file_0 = pd.DataFrame(index = None,columns=list(["Id", "Pred"]))
    test_file_0["Id"] = test_data_0.index
    
    
    test_data_1.iloc[:,:-2] =(test_data_1.iloc[:,:-2]-train_regress.iloc[:,:-2].mean())/(train_regress.iloc[:,:-2].std())
    test_data_0.iloc[:,:-2] =(test_data_0.iloc[:,:-2]-train_regress.iloc[:,:-2].mean())/(train_regress.iloc[:,:-2].std())
    train_regress.iloc[:,:-1]=(train_regress.iloc[:,:-1]-train_regress.iloc[:,:-1].mean())/(train_regress.iloc[:,:-1].std())
    
    test_data_1 = test_data_1.as_matrix()
    test_data_0 = test_data_0.as_matrix()
    train_regress = train_regress.as_matrix()
    
    x_train = train_regress[:,:-2]
    #x_train = feature_creat(x_train)
    
    y_train = train_regress[:,-2]
    
   
    
    from sklearn import linear_model
    reg = linear_model.Lasso(alpha = 0.1)
    reg.fit(x_train,y_train)
    y_pre_xgb_1 = pd.Series(reg.predict(x_test_1)*y_std+y_mean)
    y_pre_xgb_0 = pd.Series(reg.predict(x_test_0)*y_std+y_mean)
    
    
   

   
   
    
    test_file_1["Pred"] = y_pre_xgb_1
    pre_0_max = y_pre_xgb_0.max()
    if pre_0_max >0:
        test_file_0["Pred"] = y_pre_xgb_0-y_pre_xgb_0.max()
    else:
        test_file_0["Pred"] = y_pre_xgb_0+y_pre_xgb_0.max()
        
    test_file_0 = pd.concat([test_file_0,test_file_1],axis=0)
    
    
    print(test_file_0.iloc[:5,:])
    test_file_0.to_csv('model/test.csv',index = None)


def start_enter(train_file,test_file):
    train_data_classify,train_data_regress = pro_data(train_file)
    test_data  = pro_data(test_file,False)
    train_classify(train_data_classify,test_data)
    
    #train_lr(train_data_classify,test_data,train_data_regress)
    
