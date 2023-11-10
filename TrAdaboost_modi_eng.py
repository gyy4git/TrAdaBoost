# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
import numpy as np
import pandas as pd
import copy

import warnings
warnings.filterwarnings('ignore')
import os
import gc
os.chdir('D:\Work\python_working\transferlearning')
gc.collect()


# In[modify tradaboost]  
class TrAdaboost:
    def __init__(self, base_classifier=DecisionTreeClassifier(), N=10, w_source=0.5, predict_p=0.5, score=roc_auc_score):
        self.base_classifier = base_classifier
        self.N = N
        self.w_source = w_source    #Proportion of initial source domain weights
        self.predict_p = predict_p   #Proportion of weak classifiers used for predicting the final result
        self.score = score
        self.beta_T = np.zeros(self.N)
        self.threshold = None
        self.classifiers = []
        self.best_round = None    #Optimal classifier no

    # calculate_weight
    def _calculate_weight(self, weights):
        sum_weight = np.sum(weights)
        shape = weights.shape[0]
        new_weights = np.asarray(weights / sum_weight * shape, order='C')
        return new_weights

    # calculate_error_rate
    def _calculate_error_rate(self, y_target, y_predict, weight_target):
        sum_weight = np.sum(weight_target)
        return np.sum(weight_target * np.abs(y_target - y_predict) / sum_weight)

    def fit(self, x_source, x_target, y_source, y_target, early_stopping_rounds=None, weights_all=False):
        x_train = np.concatenate((x_source, x_target), axis=0)
        y_train = np.concatenate((y_source, y_target), axis=0)
        x_train = np.asarray(x_train, order='C')
        y_train = np.asarray(y_train, order='C')
        y_source = np.asarray(y_source, order='C')
        y_target = np.asarray(y_target, order='C')
        
        row_source = x_source.shape[0]
        row_target = x_target.shape[0]
        self.threshold = np.sum(y_target) / row_target
        
        # Initialize weights
        # (adjust the proportion of source and target domain weights)
        weight_source = np.ones(row_source) / row_source * self.w_source
        weight_target = np.ones(row_target) / row_target * (1 - self.w_source)
        weights = np.concatenate((weight_source, weight_target), axis=0)

        beta = 1 / (1 + np.sqrt(2 * np.log(row_source / self.N)))

        result = np.ones([row_source + row_target, self.N])
        y_pred = np.zeros([row_target, self.N])
        score = np.zeros(self.N)
        count = 0
        
        for i in range(self.N):
            weights = self._calculate_weight(weights)
            self.base_classifier.fit(x_train, y_train, sample_weight=weights)
            self.classifiers.append(copy.deepcopy(self.base_classifier))
            result[:, i] = self.base_classifier.predict_proba(x_train)[:,1]
            
            # Note that only the error rate on the target domain is calculated
            # and the decision threshold is adjusted based on weights.
            y_pred[:,i] = (result[row_source:, i] > self.threshold).astype(int)
            error_rate = self._calculate_error_rate(y_target, y_pred[:, i], weights[row_source:])
            print("Error Rate in target data:", round(error_rate,3), 'round:', i)
            
            # If the early stopping parameter (stopping_rounds) is not None,
            # early stopping is triggered when the score decreases continuously
            # for the specified number of rounds.
            score[i] = self.score(y_target, result[row_source:, i])
            print("Score is:", round(score[i],3))
            if isinstance(early_stopping_rounds, int) and i>0:  # Count is not updated in the first round
                if round(score[i],3) < round(score[i-1],3):
                    count += 1 
                else:
                    count = 0
                    self.best_round = i+1
                if count >= early_stopping_rounds:
                    self.N = i+1
                    print("Score not improve, early stop at %d rounds" %i)
                    break
                
            # when error_rate<=0.5 can make sure beta_T<=1;
            # The smaller the error_rate is,
            # the greater the reciprocal of beta_T is,
            # and the greater the change in the weight of the misclassified samples is.
            if error_rate > 0.5:
                self.beta_T[i] = 1
            elif error_rate == 0:
                self.N = i+1
                print("Error Rate is zero, early stop at %d rounds" %i)
                break
            else:
                self.beta_T[i] = error_rate / (1 - error_rate)
                
            # Adjust the weight of the target sample, and the weight of the wrong sample becomes larger
            # (the weight of the correct sample remains unchanged)
            # (the default is not adjusted)
            if weights_all:
                for t in range(row_target):
                    weights[row_source + t] = weights[row_source + t] * np.power(self.beta_T[i], -np.abs(result[row_source + t, i] - y_target[t]))
            # Adjust the source sample weight,
            # and the weight of the wrong sample becomes smaller
            # (the weight of the correct sample remains unchanged)
            for s in range(row_source):
                weights[s] = weights[s] * np.power(beta, np.abs(result[s, i] - y_source[s]))

    def predict(self, x_test):
        result = np.ones([x_test.shape[0], self.N])

        for i,classifier in enumerate(self.classifiers):
            y_prob = classifier.predict_proba(x_test)[:,1]
            y_pred = (y_prob > self.threshold).astype(int)
            result[:, i] = y_pred

        start = int(np.floor((1 - self.predict_p) * self.N))
        predict = np.sum(result[:,start:],axis=1) / (self.N - start)
        predict = np.array([round(x,0) for x in predict])
                
        return predict
    
    # def predict(self, x_test):
    #     result = np.ones([x_test.shape[0], self.N])
    #     predict = np.zeros(x_test.shape[0])

    #     for i,classifier in enumerate(self.classifiers):
    #         y_prob = classifier.predict_proba(x_test)[:,1]
    #         y_pred = (y_prob > self.threshold).astype(int)
    #         result[:, i] = y_pred

    #     start = int(np.floor((1 - self.predict_p) * self.N))
    #     for i in range(x_test.shape[0]):
    #         left = np.sum(result[i, start:] * np.log(1 / self.beta_T[start:self.N]))
    #         right = 0.5 * np.sum(np.log(1 / self.beta_T[start:self.N]))
    #         if left >= right:
    #             predict[i] = 1
    #         else:
    #             predict[i] = 0
                
    #     return predict

    def predict_proba(self, x_test):
        result = np.ones([x_test.shape[0], self.N])
        predict = np.zeros([x_test.shape[0], 2])

        for i,classifier in enumerate(self.classifiers):
            y_prob = classifier.predict_proba(x_test)[:,0]
            result[:, i] = y_prob
            
        start = int(np.floor((1 - self.predict_p) * self.N))
        predict[:, 0] = np.sum(result[:,start:],axis=1) / (self.N - start)
        predict[:, 1] = 1 - predict[:, 0]
            
        return predict


if __name__ == '__main__':

    # read csv
    train_A = pd.read_csv('A_train.csv')
    train_B = pd.read_csv('B_train.csv')
    # test = pd.read_csv('B_test.csv')
    # sample = pd.read_csv('submit_sample.csv')

    flag_A = train_A['flag']
    train_A.drop('no', axis=1, inplace=True)
    train_A.drop('flag', axis=1, inplace=True)
    flag_B = train_B['flag']
    train_B.drop('no', axis=1, inplace=True)
    train_B.drop('flag', axis=1, inplace=True)

    # Feature processing: Filtering features with na>1%.
    info = train_B.describe()
    useful_col = []
    for col in info.columns:
        if info.loc['count',col] > train_B.shape[0]*0.01:
            useful_col.append(col)
    train_B = train_B[useful_col]
    train_A = train_A[useful_col]

    # In[parameter settings]
    # scale_pos_weight = (flag_A==0).sum()/flag_A.sum()
    # weights = np.ones(train_A_valid.shape[0]) * 5
    # weights = weights * np.array(flag_A_valid) + 1
    # scale_pos_weight = 6

    params = {'booster':'gbtree',
              'eta':0.1,
              'max_depth':3,
              'max_delta_step':0,
              'subsample':1,
              'colsample_bytree':1,
              # 'scale_pos_weight':scale_pos_weight,
              'objective':'binary:logistic',
              'lambda':3,
              'alpha':8,
              'n_estimators':100,
              'seed':100,
              'n_jobs':-1}

    # In[baseline model]
    # train_A_valid,train_A_test,flag_A_valid,flag_A_test = train_test_split(train_A,flag_A,test_size=0.3)
    # #dtrain_B = xgb.DMatrix(data=train_B, label=flag_B)

    # clf = xgb.XGBClassifier(**params)
    # # clf.fit(train_A_valid, flag_A_valid, sample_weight=weights)
    # clf.fit(train_A_valid, flag_A_valid)

    # pred_A_valid = clf.predict_proba(train_A_valid)[:,1]
    # pred_A_test = clf.predict_proba(train_A_test)[:,1]
    # pred_B = clf.predict_proba(train_B)[:,1]
    # print(f" train AUC = {roc_auc_score(flag_A_valid,pred_A_valid)}")
    # print(f" test AUC = {roc_auc_score(flag_A_test,pred_A_test)}")
    # print(f" valid AUC = {roc_auc_score(flag_B,pred_B)}")

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.figure(figsize=(16,9))
    # sns.kdeplot(pred_A_valid, shade=True, label='Train_A_valid')
    # sns.kdeplot(pred_A_test, shade=True, label='Train_A_test')
    # # sns.kdeplot(pred_B, shade=True, label='Train_B')
    # plt.legend(loc='upper left')
    # plt.title('scale平衡')
    # plt.savefig("scale平衡.png")
    # plt.close()

    train_B_valid,train_B_test,flag_B_valid,flag_B_test = train_test_split(train_B,flag_B,test_size=0.5)

    train = pd.concat([train_A, train_B_valid])
    flag = pd.concat([flag_A, flag_B_valid])
    clf = xgb.XGBClassifier(**params)
    clf.fit(train, flag)

    pred_A = clf.predict_proba(train_A)[:,1]
    pred_B_valid = clf.predict_proba(train_B_valid)[:,1]
    pred_B_test = clf.predict_proba(train_B_test)[:,1]
    print(f" train AUC = {roc_auc_score(flag_A,pred_A)}")
    print(f" valid AUC = {roc_auc_score(flag_B_valid,pred_B_valid)}")
    print(f" test AUC = {roc_auc_score(flag_B_test,pred_B_test)}")

    # In[function]
    clf2 = TrAdaboost(base_classifier=xgb.XGBClassifier(**params), N=50)
    clf2.fit(train_A, train_B_valid, flag_A, flag_B_valid, early_stopping_rounds=1)
    pred_A2 = clf2.predict_proba(train_A)[:,1]
    pred_B_valid2 = clf2.predict_proba(train_B_valid)[:,1]
    pred_B_test2 = clf2.predict_proba(train_B_test)[:,1]
    print(f" train AUC = {roc_auc_score(flag_A,pred_A2)}")
    print(f" valid AUC = {roc_auc_score(flag_B_valid,pred_B_valid2)}")
    print(f" test AUC = {roc_auc_score(flag_B_test,pred_B_test2)}")

    # Performance of each base learner
    for i,estimator in enumerate(clf2.classifiers):
        print('The '+str(i+1)+' estimator:')
        pred_A = estimator.predict_proba(train_A)[:,1]
        pred_B_valid = estimator.predict_proba(train_B_valid)[:,1]
        pred_B_test = estimator.predict_proba(train_B_test)[:,1]
        print(f" train AUC = {roc_auc_score(flag_A,pred_A)}")
        print(f" valid AUC = {roc_auc_score(flag_B_valid,pred_B_valid)}")
        print(f" test AUC = {roc_auc_score(flag_B_test,pred_B_test)}")



