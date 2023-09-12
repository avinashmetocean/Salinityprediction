#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:44:51 2023

@author: avi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





def sigma_S(df, z):
    
    e = []
    for i in range(len(z)):
        d3 = df.copy()
        d3 = d3[d3.DEPTH.isin([z[i]])]
        savg = d3.SALT.mean()
        ssd = d3.SALT.std()
        
        ### 2 Std Deviation
        s1 = savg-2*ssd
        s2 = savg+2*ssd        
        d3.loc[d3['SALT'] >= s2,'SALT'] = np.nan
        d3.loc[d3['SALT'] <= s1,'SALT'] = np.nan        
        d3.dropna(subset = ["SALT"], inplace=True) 
        e.append(d3)
        
    e = pd.concat(e)
    return e
        
        



def sigma_T(df, z):
    
    f = []
    for i in range(len(z)):
        d3 = df.copy()
        d3 = d3[d3.DEPTH.isin([z[i]])]
        savg = d3.TEMP.mean()
        ssd = d3.TEMP.std()
        
        ### 2 Std Deviation
        s1 = savg-2*ssd
        s2 = savg+2*ssd        
        d3.loc[d3['TEMP'] >= s2,'TEMP'] = np.nan
        d3.loc[d3['TEMP'] <= s1,'TEMP'] = np.nan        
        d3.dropna(subset = ["TEMP"], inplace=True) 
        f.append(d3)
    f = pd.concat(f)
    return f


### Arabian Sea

def as_tqc(df):
    ### Arabian Sea
    t_zlev = [0, 100, 300, 1000, 1500, 2500, 4000]

    t_mins = [10, 5, 3, 0, 0, 0]
    t_maxs = [32, 28, 25, 18, 13, 7]
    
    as_df = df.copy()
    as_df.loc[as_df['LON'] >= 78,'LON'] = np.nan
    as_df = as_df.dropna()
    
    e = []
    for i in range(len(t_zlev)-1):
        as_dff = as_df.copy()
        as_dff.loc[as_dff['DEPTH'] >= t_zlev[i+1],'DEPTH'] = np.nan
        as_dff.loc[as_dff['DEPTH'] <= t_zlev[i],'DEPTH'] = np.nan
        as_dff.dropna(subset = ["DEPTH"], inplace=True)
        
    
        t1 = t_mins[i]
        t2 = t_maxs[i]
        
        as_dff.loc[as_dff['TEMP'] >= t2,'TEMP'] = np.nan
        as_dff.loc[as_dff['TEMP'] <= t1,'TEMP'] = np.nan        
        as_dff.dropna(subset = ["TEMP"], inplace=True)   
        e.append(as_dff)
        del as_dff
    e = pd.concat(e)
    e = e.dropna()
    return e


## Bay of Bengal

def bb_tqc(df):
    
    ### Bay of Bengal
    ### Need to check
    t_zlev = [0, 100, 300, 1000, 1500, 2500, 4000]
    t_mins = [16, 10, 3, 2, 2, 2]
    t_maxs = [32, 28, 14, 8, 6, 4]
    
    bb_df = df.copy()
    bb_df.loc[bb_df['LON'] <= 78,'LON'] = np.nan
    bb_df = bb_df.dropna()
    
    f = []
    for i in range(len(t_zlev)-1):
        bb_dff = bb_df.copy()
        bb_dff.loc[bb_dff['DEPTH'] >= t_zlev[i+1],'DEPTH'] = np.nan
        bb_dff.loc[bb_dff['DEPTH'] <= t_zlev[i],'DEPTH'] = np.nan
        bb_dff.dropna(subset = ["DEPTH"], inplace=True)
        
    
        t1 = t_mins[i]
        t2 = t_maxs[i]
        
        bb_dff.loc[bb_dff['TEMP'] >= t2,'TEMP'] = np.nan
        bb_dff.loc[bb_dff['TEMP'] <= t1,'TEMP'] = np.nan        
        bb_dff.dropna(subset = ["TEMP"], inplace=True)   
        f.append(bb_dff)
        del bb_dff
    f = pd.concat(f)
    f = f.dropna()
    
    return f


#%%

#### Arabian Sea datagen

as_p1 = '/media/avi/One Touch/3 files/MODELS_TEST/AS/AS_TRAIN.csv'
as_d1 = pd.read_csv(as_p1).iloc[:, 1:]
# d1['TEMP2'] = d1.TEMP*d1.SSS*-d1.LAT

z = as_d1.DEPTH.unique()
z.sort()
# z = z[:-19]
as_d2 = as_d1.copy()
as_d2 = sigma_S(as_d2, z)
as_d2 = sigma_T(as_d2, z)

as_d3 = as_d2.copy()
as_d3 = as_tqc(as_d3)


### Test

as_c1 = pd.read_csv('/media/avi/One Touch/3 files/MODELS_TEST/AS/AS_TEST.csv')

as_c2 = as_c1.copy().iloc[:, 1:]
as_c2 = sigma_S(as_c2, z)
as_c2 = sigma_T(as_c2, z)
as_c3 = as_tqc(as_c2)

#%%

#### Bay of Bengal datagen

bb_p1 = '/media/avi/One Touch/3 files/MODELS_TEST/BB/BB_TRAIN.csv'
bb_d1 = pd.read_csv(bb_p1).iloc[:, 1:]
# d1['TEMP2'] = d1.TEMP*d1.SSS*-d1.LAT

z = bb_d1.DEPTH.unique()
z.sort()
# z = z[:-19]
bb_d2 = bb_d1.copy()
bb_d2 = sigma_S(bb_d2, z)
bb_d2 = sigma_T(bb_d2, z)

bb_d3 = bb_d2.copy()
bb_d3 = bb_tqc(bb_d3)


### Test

bb_c1 = pd.read_csv('/media/avi/One Touch/3 files/MODELS_TEST/BB/BB_TEST.csv')

bb_c2 = bb_c1.copy().iloc[:, 1:]
bb_c2 = sigma_S(bb_c2, z)
bb_c2 = sigma_T(bb_c2, z)
bb_c3 = bb_tqc(bb_c2)



#%%

as_train = as_d3.copy().sample(100000)
as_test = as_tqc(as_c3).sample(10000)
as_train_X, as_train_Y = as_train.drop(columns='SALT'), as_train['SALT']
as_test_X, as_test_Y = as_test.drop(columns='SALT'), as_test['SALT']



# bb_train = bb_d3.copy().sample(100000)
# bb_test = bb_tqc(bb_c3).sample(10000)
# bb_train_X, bb_train_Y = bb_train.drop(columns='SALT'), bb_train['SALT']
# bb_test_X, bb_test_Y = bb_test.drop(columns='SALT'), bb_test['SALT']



#%%
### XGBOOST


import xgboost as xgb
# from sklearn.cross_validation import train_test_split


# Hyperparameters tuning

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

import gc

# Some constants

SEED = 46
VALID_SIZE = 0.2
TARGET = 'outcome'


# Extract the train and valid (used for validation) dataframes from the train_df

# train, valid = train_test_split(train_df, test_size=VALID_SIZE,
#                                 random_state=SEED)

train_features = as_train_X
valid_features = as_test_X
y_train = as_train_Y
y_valid = as_test_Y

print('The training set is of length: ', len(as_train_X))
print('The validation set is of length: ', len(as_test_X))


from sklearn.metrics import mean_squared_error

def objective(params):
    

    dtrain = xgb.DMatrix(train_features, label=y_train)
    dvalid = xgb.DMatrix(valid_features, label=y_valid)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgb.train(params, dtrain,  early_stopping_rounds=10, 
                          evals=watchlist, num_boost_round=100,
                          verbose_eval=True)
    
    pred = gbm_model.predict(dvalid)
    

    rmse = mean_squared_error(y_valid, pred, squared=False)
    print ("SCORE:", rmse)
    
    del dtrain, dvalid
    _ = gc.collect()
    
    #Specify what the loss is for each model.
    return {'loss':rmse, 'status': STATUS_OK, 'model': gbm_model}

space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'learning_rate': hp.quniform('learning_rate', 0.001, 0.9, 0.01),
    # A problem with max_depth casted to float instead of int with
    # the hp.quniform method.
    'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 0.99, 0.05),
    #'eval_metric': 'auc',
   # 'objective': 'binary:logistic',
   'objective': 'reg:squarederror',
    # Increase this number if you have more cores. Otherwise, remove it and it will default 
    # to the maxium number. 
    'nthread': -1,
    'booster': 'gbtree',
    'tree_method': 'gpu_hist',
    'predictor':'gpu_predictor',
    #'silent': 1,
    'seed': 46,
    'gpu_id': 0,
}

#Run 20 trials.
trials = Trials(100)

_ = gc.collect()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1000,
            trials=trials)
_ = gc.collect()

print(best)
print("Hyperopt estimated optimum {}".format(best))

#%%

as_train = as_d3.copy()
as_test = as_tqc(as_c3)
as_train_X, as_train_Y = as_train.drop(columns='SALT'), as_train['SALT']
as_test_X, as_test_Y = as_test.drop(columns='SALT'), as_test['SALT']





# bb_train = bb_d3.copy()
# bb_test = bb_tqc(bb_c3)
# bb_train_X, bb_train_Y = bb_train.drop(columns='SALT'), bb_train['SALT']
# bb_test_X, bb_test_Y = bb_test.drop(columns='SALT'), bb_test['SALT']

# Bay of Bengal Hyperparameters
# colsample_bytree 0.95	
# gamma 0.5	
# learning_rate 0.08	
# max_depth 12	
# min_child_weight 5
# n_estimators 659	
# subsample 1
	


# Arabian Sea Hyperparameters
# colsample_bytree 1	
# gamma 0.5	
# learning_rate 0.1	
# max_depth 11	
# min_child_weight 4
# n_estimators 769	
# subsample 1
								



from scipy.stats import pearsonr
import xgboost as xgb
import joblib

dtrain = xgb.DMatrix(as_train_X, label=as_train_Y)
dvalid = xgb.DMatrix(as_test_X, label=as_test_Y)
watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
gbm_model = xgb.train(best, dtrain,  #early_stopping_rounds=200, 
                      evals=watchlist, num_boost_round=1000,
                      verbose_eval=True)

pred = gbm_model.predict(dvalid)

mse = mean_squared_error(as_test_Y, pred)
print(mse)
corr, _ = pearsonr(as_test_Y, pred)
print('Pearsons correlation: %.3f' % corr)  

c5 = as_test.copy()
c5['XGB'] = pred

#%%

c5 = c5[c5.LON<100]
Z = c5.DEPTH.unique()
Z.sort()
# Z = Z[:-18]
dat = np.empty([len(Z), 7])
dat.fill(np.nan)

for i in range(len(Z)):
    
    a = c5[c5.DEPTH.isin([Z[i]])]
    a1 = np.around(a.corr().SALT.values, 3)
    # a = a.sample(10000)
    print(Z[i])
    print(f'Length of dataset is ', len(a))
    print('Corr of dataset is ', a1)
    print('\n\n')
    
    dat[i, 0] = Z[i]
    dat[i, 1] = a1[0]
    dat[i, 2] = a1[2]
    dat[i, 3] = a1[3]
    dat[i, 4] = a1[4]
    dat[i, 5] = a1[5]
    dat[i, 6] = a1[7]

#%%


import xarray as xr

p3 =  ["/media/avi/BDAA-3039/EN4/EN.4.2.1.f.analysis.g10.2015.nc",
"/media/avi/BDAA-3039/EN4/EN.4.2.1.f.analysis.g10.2016.nc",
"/media/avi/BDAA-3039/EN4/EN.4.2.1.f.analysis.g10.2017.nc",
"/media/avi/BDAA-3039/EN4/EN.4.2.1.f.analysis.g10.2018.nc"]


d4 = []
for i in p3:
    
    d3 = xr.open_dataset(i)
    d3 = d3.sel(lat=slice(0, 40), lon=slice(40,110))
    d4.append(d3)

d4 = xr.concat(d4, dim='time')


la = 8
lo = 70
point1 = d4.sel(lat=la, lon=lo)
t = point1.temperature.values
s = point1.salinity.values
sss = point1.salinity.values[:, 0]


time = np.array(pd.to_datetime(point1.time.values).strftime('%j'), dtype='int')

z = np.around(point1.depth.values, 1)
z = np.array(z, dtype='int')

la = np.repeat(la, len(z))
lo = np.repeat(lo, len(z))

dat2 = []
for i in range(t.shape[0]):
    
    t1 = t[i, :]-273.15
    s1 = s[i, :]
    la1, lo1 = la, lo
    ss1 = sss[i]
    ss1 = np.repeat(ss1, len(z))
    z1 = z
    time1 = time[i]
    time1 = np.repeat(time1, len(z))
    
    dat = np.vstack([t1, z1, la1, lo1, time1, ss1, s1]).T
    
    dat2.append(dat)
    
dat2 = np.array(dat2)
dat2 = np.vstack(dat2)

dat2 = pd.DataFrame(dat2, columns=['TEMP', 'DEPTH', 'LAT', 'LON', 'JDAY', 'SSS', 'SALT'])
dat2 = dat2.dropna()

#%% 
### TESTING EN4

dvalid = xgb.DMatrix(dat2.drop(columns='SALT'), label=dat2.SALT)

dat2['PRED'] = gbm_model.predict(dvalid)


# bb_dvalid = xgb.DMatrix(dat2.drop(columns='SALT'), label=dat2.SALT)

# dat2['PRED'] = xgb_bb.predict(bb_dvalid)


#%%
import matplotlib.pyplot as plt

Z = dat2.DEPTH.unique()
T = pd.to_datetime(point1.time.values).strftime('%Y-%m')

fig = plt.figure()
for i in range(len(Z)):
    
    
    dummy = dat2[dat2.DEPTH==Z[i]]
    plt.plot(T, dummy.SALT, c='k')
    plt.plot(T, dummy.PRED, c='r')
    plt.ylim([38, 34])
    plt.show()