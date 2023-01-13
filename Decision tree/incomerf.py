#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv('trainrf2.csv')
df.head
df.info()


# In[2]:


from collections import Counter
print('income ', Counter(df['y']))


# In[ ]:





# In[3]:


from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.model_selection import train_test_split
for line in df:
    line = line.strip("")
    line = line.strip("'")
    line = line.split(",")
    #line = [float(x) for x in line]
y = df['y'] 
X = df.iloc[:, 1:-1]
rf = RandomForestRegressor()
scores = defaultdict(list)
names = list(df.columns.values)
for i in range(1,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=12345)
    r = rf.fit(X_train, y_train)
    acc = r2_score(y_test, rf.predict(X_test))    
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        X_t=np.array(X_t)
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(y_test, rf.predict(X_t))
        scores[names[i]].append((acc-shuff_acc)/acc)
print ("Features sorted by their score:")
print( sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True))


# In[6]:


#决策树
import sklearn.tree as tree
# 直接使用交叉网格搜索来优化决策树模型，边训练边优化
from sklearn.model_selection import GridSearchCV
# 网格搜索的参数：正常决策树建模中的参数 - 评估指标，树的深度，最小拆分的叶子样本数与树的深度
param_grid = {'criterion': ['entropy', 'gini'],
             'max_depth': [2, 3, 4, 5, 6, 7, 8],
             'min_samples_split': [4, 8, 12, 16, 20, 24, 28]}
clf = tree.DecisionTreeClassifier()  # 定义一棵树
clfcv = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc',
                    cv=4) 

clfcv.fit(X=X_train, y=y_train)


# In[7]:


test_est = clfcv.predict(X_test)

# 模型评估
import sklearn.metrics as metrics

print("决策树准确度:")
print(metrics.classification_report(y_test,test_est)) # 该矩阵表格其实作用不大
print("决策树 AUCclfcv.best_params_:")
fpr_test, tpr_test, th_test = metrics.roc_curve(y_test, test_est)
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))


# In[8]:


clfcv.best_params_


# In[9]:


# 将最优参数代入到模型中，重新训练、预测

clf2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=4)

clf2.fit(X_train, y_train)

test_res2 = clf2.predict(X_test)


# In[10]:


#随机森林建模
param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[5, 6, 7, 8],    # 深度：这里是森林中每棵决策树的深度
    'n_estimators':[11,13,15],  # 决策树个数-随机森林特有参数
    'max_features':[0.3,0.4,0.5], # 每棵决策树使用的变量占比-随机森林特有参数（结合原理）
    'min_samples_split':[4,8,12,16]  # 叶子的最小拆分样本量
}
import sklearn.ensemble as ensemble
rfc = ensemble.RandomForestClassifier()
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      scoring='roc_auc', cv=4)
rfc_cv.fit(X_train, y_train)


# In[11]:


# 使用随机森林对测试集进行预测
test_est = rfc_cv.predict(X_test)
print('随机森林精确度...')
print(metrics.classification_report(test_est, y_test))
print('随机森林 AUC...')
fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test) # 构造 roc 曲线
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))
# AUC ，即预测类模型的精度大大提升


# In[12]:


rfc_cv.best_params_


# In[89]:


param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[7, 8, 10, 12], # 前面的 5，6 也可以适当的去掉，反正已经没有用了
    'n_estimators':[11, 13, 15, 17, 19],  #决策树个数-随机森林特有参数
    'max_features':[0.4, 0.5, 0.6, 0.7], #每棵决策树使用的变量占比-随机森林特有参数
    'min_samples_split':[2, 3, 4, 8, 12, 16]  # 叶子的最小拆分样本量
}
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      scoring='roc_auc', cv=4)
rfc_cv.fit(X_train, y_train)
test_est = rfc_cv.predict(X_test)
print('随机森林精确度...')
print(metrics.classification_report(test_est, y_test))
print('随机森林 AUC...')
fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test) # 构造 roc 曲线
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))


# In[90]:


rfc_cv.best_params_


# In[91]:


param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[12, 15, 17, 19], 
    'n_estimators':[17, 19, 22, 25, 27],  #决策树个数-随机森林特有参数
    'max_features':[0.1, 0.2, 0.3, 0.4], #每棵决策树使用的变量占比-随机森林特有参数
    'min_samples_split':[2, 3, 4, 8, 12, 16]  # 叶子的最小拆分样本量
}
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      scoring='roc_auc', cv=4)
rfc_cv.fit(X_train, y_train)
test_est = rfc_cv.predict(X_test)
print('随机森林精确度...')
print(metrics.classification_report(test_est, y_test))
print('随机森林 AUC...')
fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test) # 构造 roc 曲线
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))


# In[2]:


rfc_cv.best_params_


# In[95]:


param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[15, 17, 19, 21], 
    'n_estimators':[25, 27, 29,32,36],  #决策树个数-随机森林特有参数
    'max_features':[0.02,0.05,0.1, 0.2],#每棵决策树使用的变量占比-随机森林特有参数
    'min_samples_split':[2, 3, 4, 8, 12, 16]  # 叶子的最小拆分样本量
}
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      scoring='roc_auc', cv=4)
rfc_cv.fit(X_train, y_train)
test_est = rfc_cv.predict(X_test)
print('随机森林精确度...')
print(metrics.classification_report(test_est, y_test))
print('随机森林 AUC...')
fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test) # 构造 roc 曲线
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))


# In[13]:


param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[15, 17, 19, 21], 
    'n_estimators':[25, 27, 29,32,36],  #决策树个数-随机森林特有参数
    'max_features':[0.02,0.05,0.1, 0.2],#每棵决策树使用的变量占比-随机森林特有参数
    'min_samples_split':[2, 3, 4, 8, 12, 16]  # 叶子的最小拆分样本量
}
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      scoring='roc_auc', cv=4)
rfc_cv.fit(X_train, y_train)
test_est = rfc_cv.predict(X_test)
print('随机森林精确度...')
print(metrics.classification_report(test_est, y_test))
print('随机森林 AUC...')
fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test) # 构造 roc 曲线
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))


# In[14]:


rfc_cv.best_params_


# In[15]:


param_grid = {
    'criterion':['entropy','gini'],
    'max_depth':[9, 12, 15, 17], 
    'n_estimators':[32,36,40,45],  #决策树个数-随机森林特有参数
    'max_features':[0.1, 0.2,0.4,0.7],#每棵决策树使用的变量占比-随机森林特有参数
    'min_samples_split':[12, 16,20,26,31]  # 叶子的最小拆分样本量
}
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid,
                      scoring='roc_auc', cv=4)
rfc_cv.fit(X_train, y_train)
test_est = rfc_cv.predict(X_test)
print('随机森林精确度...')
print(metrics.classification_report(test_est, y_test))
print('随机森林 AUC...')
fpr_test, tpr_test, th_test = metrics.roc_curve(test_est, y_test) # 构造 roc 曲线
print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))


# In[16]:


rfc_cv.best_params_


# In[ ]:




