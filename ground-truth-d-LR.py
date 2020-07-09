#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.ma as ma
import pymc3 as pm
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, CategoricalNB
from sklearn.metrics import log_loss

NumEMSteps=15


# In[2]:


# run_id=0
import argparse

parser = argparse.ArgumentParser(description='Compare models with different ground truths.')
parser.add_argument("id", help="run ID",type=int)
args = parser.parse_args()
run_id=args.id


# In[3]:


# os.environ["THEANO_FLAGS"]="base_compiledir=/home/MINERVA/amehr/.theano1/d-%d/"%(run_id)
# print(os.environ["THEANO_FLAGS"])
import theano
import theano.tensor as tt
print(theano.config.base_compiledir)


# In[4]:


def get_3rands_with_zero_at(i):
    ret=np.random.rand(3)
    ret[i]=0
    return ret/np.sum(ret)

haveSibling_gt=np.random.rand(1)[0]#0.5

Anxiety_gt=np.random.rand(1)[0]#0.32

threeWs=3*np.random.uniform(-1,1,3)
H_g_SA=np.zeros((2,2))
H_g_SA[0,0]=threeWs[0]
H_g_SA[0,1]=threeWs[0]+threeWs[1]
H_g_SA[1,0]=threeWs[0]+threeWs[2]
H_g_SA[1,1]=threeWs[0]+threeWs[1]+threeWs[2]
H_g_SA=1/(1+np.exp(-H_g_SA))

happyAlone_given_S_A=tt.as_tensor_variable(H_g_SA) #[[0.89,0.67],[0.47,0.18]]

ObsHaveSibling_given_HaveSiblingT_gt=get_3rands_with_zero_at(0)
ObsHaveSibling_given_HaveSiblingF_gt=get_3rands_with_zero_at(1)

ObsAnxiety_given_AnxietyT_gt=get_3rands_with_zero_at(0)
ObsAnxiety_given_AnxietyF_gt=get_3rands_with_zero_at(1)

print({
    'haveSibling_gt':haveSibling_gt,
    'Anxiety_gt':Anxiety_gt,
    'happyAlone_given_S_A':happyAlone_given_S_A,
    'ObsHaveSibling_given_HaveSiblingT_gt':ObsHaveSibling_given_HaveSiblingT_gt,
    'ObsHaveSibling_given_HaveSiblingF_gt':ObsHaveSibling_given_HaveSiblingF_gt,
    'ObsAnxiety_given_AnxietyT_gt':ObsAnxiety_given_AnxietyT_gt,
    'ObsAnxiety_given_AnxietyF_gt':ObsAnxiety_given_AnxietyF_gt,
})


basic_model = pm.Model()

with basic_model:
    haveSibling=pm.Bernoulli('haveSibling',haveSibling_gt)
    
    Anxiety=pm.Bernoulli('Anxiety',Anxiety_gt)
    
    happyAlone_p=happyAlone_given_S_A[haveSibling,Anxiety]
    happyAlone=pm.Bernoulli('happyAlone',happyAlone_p)
    
    ObsHaveSibling_p=pm.math.switch(haveSibling, ObsHaveSibling_given_HaveSiblingT_gt, ObsHaveSibling_given_HaveSiblingF_gt)
    ObsHaveSibling=pm.Categorical('ObsHaveSibling',ObsHaveSibling_p)
    
    ObsAnxiety_p=pm.math.switch(Anxiety, ObsAnxiety_given_AnxietyT_gt, ObsAnxiety_given_AnxietyF_gt)
    ObsAnxiety=pm.Categorical('ObsAnxiety',ObsAnxiety_p)
#     trace = pm.sample(10000, tune=1000, cores=5)
    trace=pm.sample_prior_predictive((10,2500))


# In[5]:


# pm.summary(trace)


# In[6]:


# pm.traceplot(trace)


# In[7]:


new_trace=[{} for _ in range(trace['happyAlone'].size)]
for k,v in trace.items():
    print(k)
    v_flat = np.ndarray.flatten(v)
    print(v_flat.sum(),v_flat.shape)
    for i in range(len(v_flat)):
        new_trace[i][k]=v_flat[i]
        
print(len(new_trace))
# print(new_trace[:10])


# In[8]:


N=int(len(new_trace) *0.8)
N_test=len(new_trace)-N
print(N,N_test)


X=np.zeros((N,4),dtype=np.int)
y=np.zeros((N,),dtype=np.int)
X_3lvls=np.zeros((X.shape[0],int(X.shape[1]/2)),dtype=np.int)
X_all_observed=np.zeros((X.shape[0],int(X.shape[1]/2)),dtype=np.int)

X_test=np.zeros((N_test,4),dtype=np.int)
y_test=np.zeros((N_test,),dtype=np.int)
X_test_3lvls=np.zeros((X_test.shape[0],int(X_test.shape[1]/2)),dtype=np.int)
X_test_all_observed=np.zeros((X_test.shape[0],int(X_test.shape[1]/2)),dtype=np.int)

for j,s in enumerate(new_trace):
#     print(s)
    i=j
    
    if j<N:
        temp_X=X
        temp_X_3lvls=X_3lvls
        temp_y=y 
        temp_X_all_observed=X_all_observed
    else:
        temp_X=X_test
        temp_X_3lvls=X_test_3lvls
        temp_y=y_test
        temp_X_all_observed=X_test_all_observed
        i=j-N
        
    temp_X_all_observed[i,0]=s['haveSibling']
    temp_X_3lvls[i,0]=s['ObsHaveSibling']
    if s['ObsHaveSibling']!=2:
        temp_X[i,[0,1]]=(s['ObsHaveSibling'],1-s['ObsHaveSibling'])
    else:
        temp_X[i,[0,1]]=(0,0)

    temp_X_all_observed[i,1]=s['Anxiety']
    temp_X_3lvls[i,1]=s['ObsAnxiety']
    if s['ObsAnxiety']!=2:
        temp_X[i,[2,3]]=(s['ObsAnxiety'],1-s['ObsAnxiety'])
        
    else:
        temp_X[i,[2,3]]=(0,0)
        
#     print(s['haveSibling'], temp_X[i,:],temp_X_3lvls[i,:])

    temp_y[i]=s['happyAlone']
    


# In[ ]:





# In[9]:


model_comparison={}
def calc_RMSE_error(preds,labels):
    return np.sqrt(np.mean((labels-preds)**2))

def calc_logloss(preds,labels):
    logloss = 0 

    for i in range(0,len(preds)):
        if labels[i] == 1:
            logloss += -np.log(preds[i])
        else:
            logloss += -np.log(1-preds[i])

    logloss = logloss/len(labels)
    return logloss

def calc_logloss_sklearn(preds,labels):
    l=np.reshape(labels,(-1,1))
    l=np.concatenate((1-l,l),axis=1)
    return log_loss(l,preds)

def cal_all_errors(preds,labels):
    d={}
    d['RMSE']=calc_RMSE_error(preds[:,1],labels)
    d['logloss']=calc_logloss(preds[:,1],labels)
    d['logloss_sklearn']=calc_logloss_sklearn(preds,labels)
    return d


# In[10]:


# LR+/- model
clf = LogisticRegression(random_state=0,max_iter=1000,tol=1e-5,penalty='none').fit(X, y)
print('intercept:',clf.intercept_)
print('weights:', clf.coef_)
text_x=np.zeros((1,4),dtype=np.int)
ls=[]
ss=0
for i in range(3):
    for j in range(3):
        
        i_txt=str(i)
        j_txt=str(j)
        
        if i==2:
            i_txt='UNK'
            text_x[0,[0,1]]=(0,0)
        else:
            text_x[0,[0,1]]=(i,1-i)
        
        if j==2:
            j_txt='UNK'
            text_x[0,[2,3]]=(0,0)
        else:
            text_x[0,[2,3]]=(j,1-j)
            
        print('i=%s\t,j=%s\t'%(i_txt,j_txt),text_x,'P(H=T|S=i,A=j)=%.3f'%(clf.predict_proba(text_x)[0,1]))
#         ls.append(clf.predict_proba(text_x)[0,1])
#         ss+=np.sum(np.all(X[:,[0,1,2,3]]==text_x[0,:],axis=1))/N*clf.predict_proba(text_x)[0,1]
#         print(np.sum(np.all(X[:,[0,1,2,3]]==text_x[0,:],axis=1)))
# print(ss)
preds=clf.predict_proba(X_test)
model_comparison['lR+/-']=cal_all_errors(preds,y_test)
model_comparison


# $P(H=T|S=i,A=j)=sigmoid(w0+w_1^+S_T+w_1^-S_F+w_2^+A_T+w_2^-A_T)$

# Naive bayes: P(H=T|S=UNK, A=UNK)= P(H=T)P(S=UNK|H=T)P(A=UNK|H=T)

# In[11]:


# # My own implementation of naive bayes for P(H=1|S=UNK, A=UNK)
# e1=np.sum(y)/N
# X_HT=X[y==1,:]
# e2=np.sum(np.all(X_HT[:,[0,1]]==np.array([0,0]),axis=1))/X_HT.shape[0]
# e3=np.sum(np.all(X_HT[:,[2,3]]==np.array([0,0]),axis=1))/X_HT.shape[0]
# p_HT_givenSUNKandAUNK=e1*e2*e3

# e1=1-e1
# X_HF=X[y==0,:]
# e2=np.sum(np.all(X_HF[:,[0,1]]==np.array([0,0]),axis=1))/X_HF.shape[0]
# e3=np.sum(np.all(X_HF[:,[2,3]]==np.array([0,0]),axis=1))/X_HF.shape[0]

# print(p_HT_givenSUNKandAUNK/(p_HT_givenSUNKandAUNK+e1*e2*e3))

# # # Compare to LR+/-:
# # sigmoid= lambda x: 1/(1+np.exp(-x))
# # sigmoid_inverse= lambda x: np.log(1/(1/x-1))
# # sigmoid(1.73118894)


# In[12]:


# clf = MultinomialNB()
# clf.fit(X, y)
# print(clf.intercept_,clf.coef_)
# print('class_count_',clf.class_count_,'class_log_prior_',np.exp(clf.class_log_prior_),'classes_',clf.classes_,
#       'feature_count_',clf.feature_count_,'feature_log_prob_',np.exp(clf.feature_log_prob_))

# text_x=np.zeros((1,4),dtype=np.int)
# ls=[]
# ss=0
# for i in range(3):
#     for j in range(3):
        
#         i_txt=str(i)
#         j_txt=str(j)
        
#         if i==2:
#             i_txt='UNK'
#             text_x[0,[0,1]]=(0,0)
#         else:
#             text_x[0,[0,1]]=(i,1-i)
        
#         if j==2:
#             j_txt='UNK'
#             text_x[0,[2,3]]=(0,0)
#         else:
#             text_x[0,[2,3]]=(j,1-j)
            
#         print('i=%s\t,j=%s\t'%(i_txt,j_txt),text_x,'P(H=T|S=i,A=j)=%.3f'%(clf.predict_proba(text_x)[0,1]))

# preds=clf.predict_proba(X_test)
# model_comparison['MultinomialNB']=cal_all_errors(preds,y_test)
# model_comparison


# In[13]:


# clf = BernoulliNB()
# clf.fit(X, y)

# text_x=np.zeros((1,4),dtype=np.int)
# ls=[]
# ss=0
# for i in range(3):
#     for j in range(3):
        
#         i_txt=str(i)
#         j_txt=str(j)
        
#         if i==2:
#             i_txt='UNK'
#             text_x[0,[0,1]]=(0,0)
#         else:
#             text_x[0,[0,1]]=(i,1-i)
        
#         if j==2:
#             j_txt='UNK'
#             text_x[0,[2,3]]=(0,0)
#         else:
#             text_x[0,[2,3]]=(j,1-j)
            
#         print('i=%s\t,j=%s\t'%(i_txt,j_txt),text_x,'P(H=T|S=i,A=j)=%.3f'%(clf.predict_proba(text_x)[0,1]))
        
# preds=clf.predict_proba(X_test)
# model_comparison['BernoulliNB']=cal_all_errors(preds,y_test)
# model_comparison


# In[14]:


clf = CategoricalNB()
clf.fit(X, y)
print('category_count_',clf.category_count_,'class_count_',clf.class_count_,
      'class_log_prior_',np.exp(clf.class_log_prior_),'classes_',clf.classes_,
     'feature_log_prob_',clf.feature_log_prob_)

text_x=np.zeros((1,4),dtype=np.int)
ls=[]
ss=0
for i in range(3):
    for j in range(3):
        
        i_txt=str(i)
        j_txt=str(j)
        
        if i==2:
            i_txt='UNK'
            text_x[0,[0,1]]=(0,0)
        else:
            text_x[0,[0,1]]=(i,1-i)
        
        if j==2:
            j_txt='UNK'
            text_x[0,[2,3]]=(0,0)
        else:
            text_x[0,[2,3]]=(j,1-j)
        if text_x[0,1]==1:
            continue
        print('i=%s\t,j=%s\t'%(i_txt,j_txt),text_x,'P(H=T|S=i,A=j)=%.3f'%(clf.predict_proba(text_x)[0,1]))
        
preds=clf.predict_proba(X_test)
model_comparison['CategoricalNB']=cal_all_errors(preds,y_test)
model_comparison


# In[15]:


clf = CategoricalNB(alpha=0.0)
clf.fit(X_3lvls, y)
print('category_count_',clf.category_count_,'class_count_',clf.class_count_,
      'class_log_prior_',np.exp(clf.class_log_prior_),'classes_',clf.classes_,
     'feature_log_prob_',np.exp(clf.feature_log_prob_))

text_x=np.zeros((1,2),dtype=np.int)
ls=[]
ss=0
for i in range(3):
    for j in range(3):
        
        i_txt=str(i)
        j_txt=str(j)
        
        if i==2:
            i_txt='UNK'
            text_x[0,0]=2
        else:
            text_x[0,0]=i
        
        if j==2:
            j_txt='UNK'
            text_x[0,1]=2
        else:
            text_x[0,1]=j

        print('i=%s\t,j=%s\t'%(i_txt,j_txt),text_x,'P(H=T|S=i,A=j)=%.3f'%(clf.predict_proba(text_x)[0,1]))
preds=clf.predict_proba(X_test_3lvls)
print(X_test_3lvls.shape,preds.shape,y_test.shape)
model_comparison['CategoricalNB_3lvls']=cal_all_errors(preds,y_test)
model_comparison


# In[16]:


def ali_classicalNB_predict(clf,X_test_3lvls,y_test):
    class_prior_=np.exp(clf.class_log_prior_)
    feature_prob_=np.exp(clf.feature_log_prob_) # feature, class, category
    preds=np.zeros((y_test.shape[0],2))
    for i in range(len(y_test)):
        for c in range(2): # c is class
            p=class_prior_[c]
            for f in range(2): # f is feature
                if X_test_3lvls[i,f]!=2:
                    p=p*feature_prob_[f,c,X_test_3lvls[i,f]]
            preds[i,c]=p
    preds=preds/np.sum(preds,axis=1).reshape((-1,1)).repeat(2,axis=1)
    return preds
clf = CategoricalNB(alpha=0.0)
clf.fit(X_3lvls, y)
preds=ali_classicalNB_predict(clf,X_test_3lvls,y_test)
model_comparison['classicalNB']=cal_all_errors(preds,y_test)
model_comparison


# In[17]:


print(model_comparison['lR+/-']['logloss'],model_comparison['CategoricalNB_3lvls']['logloss'])


# In[18]:


# # a) Naive Bayes using pymc3 for sanity check.
# y_Nby3=np.reshape(y,(-1,1))
# y_Nby3=np.repeat(y_Nby3,3,axis=1)
# # temp_y=[1,1,1,1,1,1,0,0,0,0,1]
# # y_Nby3=np.repeat(np.reshape(np.array(temp_y),(-1,1)),3,axis=1)

# basic_model = pm.Model()

# with basic_model:
#     p_happyAlone=pm.Beta('p_happyAlone', 1,1)
#     happyAlone=pm.Bernoulli('happyAlone',p_happyAlone, observed=y)
    
#     p_haveSibling_given_happyAloneT=pm.Dirichlet('p_haveSibling_given_happyAloneT', a=np.ones(3)).reshape((1,-1)).repeat(y_Nby3.shape[0],axis=0)
#     p_haveSibling_given_happyAloneF=pm.Dirichlet('p_haveSibling_given_happyAloneF', a=np.ones(3)).reshape((1,-1)).repeat(y_Nby3.shape[0],axis=0)
    
#     haveSibling_probs = (y_Nby3==1)*p_haveSibling_given_happyAloneT + (y_Nby3==0)*p_haveSibling_given_happyAloneF


#     haveSibling=pm.Categorical('haveSibling', 
#         p=haveSibling_probs
#         ,observed=X_3lvls[:,0]
#     )
    
    
#     p_Anxiety_given_happyAloneT=pm.Dirichlet('p_Anxiety_given_happyAloneT', a=np.ones(3)).reshape((1,-1)).repeat(y_Nby3.shape[0],axis=0)
#     p_Anxiety_given_happyAloneF=pm.Dirichlet('p_Anxiety_given_happyAloneF', a=np.ones(3)).reshape((1,-1)).repeat(y_Nby3.shape[0],axis=0)
    
#     Anxiety_probs = (y_Nby3==1)*p_Anxiety_given_happyAloneT + (y_Nby3==0)*p_Anxiety_given_happyAloneF
#     Anxiety=pm.Categorical('Anxiety',
#         p=Anxiety_probs
#         ,observed=X_3lvls[:,1]
#         )
#     mean_q = pm.find_MAP(maxeval=50000,return_raw=True)
    
# #     prior_trace = pm.sample(2000, tune=1000, cores=10)
#                         # step=pm.Metropolis()
# # pm.traceplot(prior_trace)    
# # pm.summary(prior_trace)
# mean_q


# In[ ]:





# # 1. Train model C on data:

# In[ ]:





# In[ ]:





# In[19]:


# c) using pymc3 


basic_model = pm.Model()

SA_predicts=np.copy(X_3lvls)
SA_predicts=(SA_predicts!=2)*SA_predicts + (SA_predicts==2)*np.random.randint(0,2,size=SA_predicts.shape)


with basic_model:
    p_happyAlone=pm.Uniform('p_happyAlone')
    happyAlone=pm.Bernoulli('happyAlone',p_happyAlone, observed=y)

    p_haveSiblingT_given_happyAloneT=pm.Uniform('p_haveSiblingT_given_happyAloneT')
    p_haveSiblingT_given_happyAloneF=pm.Uniform('p_haveSiblingT_given_happyAloneF')
    
    
#     haveSibling_probs = (y==1)*p_haveSiblingT_given_happyAloneT + (y==0)*p_haveSiblingT_given_happyAloneF
    haveSibling_probs = pm.math.switch(happyAlone,p_haveSiblingT_given_happyAloneT,p_haveSiblingT_given_happyAloneF)
    haveSibling=pm.Bernoulli('haveSibling', haveSibling_probs,shape=(N,),testval=SA_predicts[:,0]).reshape((-1,1))
    
    p_ObsHaveSibling_given_haveSiblingT=pm.Dirichlet('p_ObsHaveSibling_given_haveSiblingT',a=np.ones(3)).reshape((1,3)).repeat(N,axis=0)
    p_ObsHaveSibling_given_haveSiblingF=pm.Dirichlet('p_ObsHaveSibling_given_haveSiblingF',a=np.ones(3)).reshape((1,3)).repeat(N,axis=0)
    
    ObsHaveSibling_probs= pm.math.switch(haveSibling,p_ObsHaveSibling_given_haveSiblingT,p_ObsHaveSibling_given_haveSiblingF)
    ObsHaveSibling=pm.Categorical('ObsHaveSibling',ObsHaveSibling_probs,observed=X_3lvls[:,0])
    
    p_AnxietyT_given_happyAloneT=pm.Uniform('p_AnxietyT_given_happyAloneT')
    p_AnxietyT_given_happyAloneF=pm.Uniform('p_AnxietyT_given_happyAloneF')
    
#     Anxiety_probs = (y==1)*p_AnxietyT_given_happyAloneT + (y==0)*p_AnxietyT_given_happyAloneF
    Anxiety_probs= pm.math.switch(happyAlone,p_AnxietyT_given_happyAloneT,p_AnxietyT_given_happyAloneF)
    Anxiety=pm.Bernoulli('Anxiety',p=Anxiety_probs,shape=(N,),testval=SA_predicts[:,1]).reshape((-1,1))

    
    p_ObsAnxiety_given_AnxietyT=pm.Dirichlet('p_ObsAnxiety_given_AnxietyT',a=np.ones(3)).reshape((1,-1)).repeat(N,axis=0)
    p_ObsAnxiety_given_AnxietyF=pm.Dirichlet('p_ObsAnxiety_given_AnxietyF',a=np.ones(3)).reshape((1,-1)).repeat(N,axis=0)
    
    ObsAnxiety_probs= pm.math.switch(Anxiety,p_ObsAnxiety_given_AnxietyT,p_ObsAnxiety_given_AnxietyF)
    ObsAnxiety=pm.Categorical('ObsAnxiety',ObsAnxiety_probs,observed=X_3lvls[:,1])

test_point={
    'haveSibling':SA_predicts[:,0],
    'Anxiety': SA_predicts[:,1]
}
with basic_model:
    mean_q = pm.find_MAP(start=test_point,
                         maxeval=50000,return_raw=True)

mean_q


# In[20]:


old_neglogprob=mean_q[1]['fun']
for EM_step in range(NumEMSteps):
    shared_p_happyAlone=(mean_q[0]['p_happyAlone'])
    shared_p_haveSiblingT_given_happyAloneT=(mean_q[0]['p_haveSiblingT_given_happyAloneT'])
    shared_p_haveSiblingT_given_happyAloneF=(mean_q[0]['p_haveSiblingT_given_happyAloneF'])
    shared_p_ObsHaveSibling_given_haveSiblingT=(mean_q[0]['p_ObsHaveSibling_given_haveSiblingT'])
    shared_p_ObsHaveSibling_given_haveSiblingF=(mean_q[0]['p_ObsHaveSibling_given_haveSiblingF'])
    shared_p_AnxietyT_given_happyAloneT=(mean_q[0]['p_AnxietyT_given_happyAloneT'])
    shared_p_AnxietyT_given_happyAloneF=(mean_q[0]['p_AnxietyT_given_happyAloneF'])
    shared_p_ObsAnxiety_given_AnxietyT=(mean_q[0]['p_ObsAnxiety_given_AnxietyT'])
    shared_p_ObsAnxiety_given_AnxietyF=(mean_q[0]['p_ObsAnxiety_given_AnxietyF'])

    SA_probs_given_UNK=np.zeros((2,2,2))
    for H in range(2):
        shared_happyAlone=H
        
        basic_model2=pm.Model()
        with basic_model2:
            happyAlone=pm.Bernoulli('happyAlone',shared_p_happyAlone, observed=shared_happyAlone)    

            haveSibling_probs = pm.math.switch(happyAlone,shared_p_haveSiblingT_given_happyAloneT,shared_p_haveSiblingT_given_happyAloneF)
            haveSibling=pm.Bernoulli('haveSibling', haveSibling_probs)#.reshape((-1,1))


            ObsHaveSibling_probs= pm.math.switch(haveSibling,shared_p_ObsHaveSibling_given_haveSiblingT,shared_p_ObsHaveSibling_given_haveSiblingF)
            ObsHaveSibling=pm.Categorical('ObsHaveSibling',ObsHaveSibling_probs,observed=2)

            Anxiety_probs= pm.math.switch(happyAlone,shared_p_AnxietyT_given_happyAloneT,shared_p_AnxietyT_given_happyAloneF)
            Anxiety=pm.Bernoulli('Anxiety',p=Anxiety_probs)#.reshape((-1,1))


            ObsAnxiety_probs= pm.math.switch(Anxiety,shared_p_ObsAnxiety_given_AnxietyT,shared_p_ObsAnxiety_given_AnxietyF)
            ObsAnxiety=pm.Categorical('ObsAnxiety',ObsAnxiety_probs,observed=2)
        for A in range(2):
            for S in range(2):
                SA_probs_given_UNK[H,S,A]=basic_model2.logp({'haveSibling':S,'Anxiety':A})

    SA_probs_given_UNK=np.exp(SA_probs_given_UNK)
    for H in range(2):
        Aprobs=SA_probs_given_UNK.sum(axis=1)[H,:]
        Sprobs=SA_probs_given_UNK.sum(axis=2)[H,:]
        Aprobs=Aprobs/Aprobs.sum()
        Sprobs=Sprobs/Sprobs.sum()
        print(H,Aprobs/Aprobs.sum(),Sprobs/Sprobs.sum())


        unobsS=(y==H)*(X_3lvls[:,0]==2)
        unobsA=(y==H)*(X_3lvls[:,1]==2)
        SA_predicts[unobsS,0]=np.random.choice([0,1],p=Sprobs/Sprobs.sum(),size=unobsS.sum())
        SA_predicts[unobsA,1]=np.random.choice([0,1],p=Aprobs/Aprobs.sum(),size=unobsA.sum())

    test_point={
        'haveSibling':SA_predicts[:,0],
        'Anxiety': SA_predicts[:,1]
    }  

    with basic_model:
        mean_q = pm.find_MAP(start=test_point,
                             maxeval=50000,return_raw=True)

#     print(mean_q)
    new_neglogprob=mean_q[1]['fun']
    if abs(new_neglogprob-old_neglogprob)<40:
        break
    old_neglogprob=new_neglogprob


# 

# In[ ]:





# ## 1.1 Test model c

# In[21]:


# c) using pymc3 

shared_p_happyAlone=(mean_q[0]['p_happyAlone'])
shared_p_haveSiblingT_given_happyAloneT=(mean_q[0]['p_haveSiblingT_given_happyAloneT'])
shared_p_haveSiblingT_given_happyAloneF=(mean_q[0]['p_haveSiblingT_given_happyAloneF'])
shared_p_ObsHaveSibling_given_haveSiblingT=(mean_q[0]['p_ObsHaveSibling_given_haveSiblingT'])
shared_p_ObsHaveSibling_given_haveSiblingF=(mean_q[0]['p_ObsHaveSibling_given_haveSiblingF'])
shared_p_AnxietyT_given_happyAloneT=(mean_q[0]['p_AnxietyT_given_happyAloneT'])
shared_p_AnxietyT_given_happyAloneF=(mean_q[0]['p_AnxietyT_given_happyAloneF'])
shared_p_ObsAnxiety_given_AnxietyT=(mean_q[0]['p_ObsAnxiety_given_AnxietyT'])
shared_p_ObsAnxiety_given_AnxietyF=(mean_q[0]['p_ObsAnxiety_given_AnxietyF'])

shared_ObsAnxiety=theano.shared(0)
shared_ObsHaveSibling=theano.shared(0)

test_model=pm.Model()
with test_model:
    happyAlone=pm.Bernoulli('happyAlone',shared_p_happyAlone)    

    haveSibling_probs = pm.math.switch(happyAlone,shared_p_haveSiblingT_given_happyAloneT,shared_p_haveSiblingT_given_happyAloneF)
    haveSibling=pm.Bernoulli('haveSibling', haveSibling_probs)#.reshape((-1,1))


    ObsHaveSibling_probs= pm.math.switch(haveSibling,shared_p_ObsHaveSibling_given_haveSiblingT,shared_p_ObsHaveSibling_given_haveSiblingF)
    ObsHaveSibling=pm.Categorical('ObsHaveSibling',ObsHaveSibling_probs,observed=shared_ObsHaveSibling)

    Anxiety_probs= pm.math.switch(happyAlone,shared_p_AnxietyT_given_happyAloneT,shared_p_AnxietyT_given_happyAloneF)
    Anxiety=pm.Bernoulli('Anxiety',p=Anxiety_probs)#.reshape((-1,1))


    ObsAnxiety_probs= pm.math.switch(Anxiety,shared_p_ObsAnxiety_given_AnxietyT,shared_p_ObsAnxiety_given_AnxietyF)
    ObsAnxiety=pm.Categorical('ObsAnxiety',ObsAnxiety_probs,observed=shared_ObsAnxiety)

my_X_test_3lvls=np.zeros((9,2),dtype=np.int)
preds=np.zeros(my_X_test_3lvls.shape)

i=0
for S in range(3):
    for A in range(3):
        my_X_test_3lvls[i,0]=S
        my_X_test_3lvls[i,1]=A
        i+=1
for i in range(my_X_test_3lvls.shape[0]):
    nom=0
    denom=0
    print('------------',my_X_test_3lvls[i,:])
    for H in range(2):
        for S in range(2):
            for A in range(2):
                t_sample_input={}
                t_sample_input['happyAlone']=H
                t_sample_input['haveSibling']=S
                t_sample_input['Anxiety']=A
                
                
                shared_ObsHaveSibling.set_value(my_X_test_3lvls[i,0])
                shared_ObsAnxiety.set_value(my_X_test_3lvls[i,1])
                
#                 print(H,S,A)
                prob=np.exp(test_model.logp(t_sample_input))
                if H==1:
                    nom+=prob
                    
                denom+=prob
    
    print(nom/denom)

    preds[i,1]=(nom/denom)  
    preds[i,0]=1-(nom/denom) 

logloss=0
for i in range(my_X_test_3lvls.shape[0]):
    for my_y in range(2):
        similar_samples_in_test= (y_test==my_y) * np.all(X_test_3lvls==my_X_test_3lvls[i,:],axis=1)
        num_sim_samples_in_test=np.sum(similar_samples_in_test)

        if my_y == 1:
            logloss += -np.log(preds[i,1])*num_sim_samples_in_test
        else:
            logloss += -np.log(1-preds[i,1])*num_sim_samples_in_test
        
print('logloss of trained (c) on the data:',logloss/len(y_test))
model_comparison['trained (c)']={'logloss':logloss/len(y_test)}
    


# # 2. Train model (d) on data

# In[22]:


# c) using pymc3 

basic_model = pm.Model()


SA_predicts=np.copy(X_3lvls)
SA_predicts=(SA_predicts!=2)*SA_predicts + (SA_predicts==2)*np.random.randint(0,2,size=SA_predicts.shape)

with basic_model:
    p_haveSibling=pm.Uniform('p_haveSibling')
    haveSibling=pm.Bernoulli('haveSibling',p_haveSibling,shape=(N,))
    
    p_Anxiety = pm.Uniform('p_Anxiety')
    Anxiety=pm.Bernoulli('Anxiety',p_Anxiety,shape=(N,))
    
    p_happyAlone_given_S_A=pm.Uniform('p_happyAlone_given_S_A',shape=(2,2))
    happyAlone=pm.Bernoulli('happyAlone',p_happyAlone_given_S_A[haveSibling,Anxiety], observed=y)
    
    p_ObsHaveSibling_given_haveSiblingT=pm.Dirichlet('p_ObsHaveSibling_given_haveSiblingT',a=np.ones(3)).reshape((1,3)).repeat(N,axis=0)
    p_ObsHaveSibling_given_haveSiblingF=pm.Dirichlet('p_ObsHaveSibling_given_haveSiblingF',a=np.ones(3)).reshape((1,3)).repeat(N,axis=0)
    
    ObsHaveSibling_probs= pm.math.switch(haveSibling.reshape((-1,1)),p_ObsHaveSibling_given_haveSiblingT,p_ObsHaveSibling_given_haveSiblingF)
    ObsHaveSibling=pm.Categorical('ObsHaveSibling',ObsHaveSibling_probs,observed=X_3lvls[:,0])
    
    p_ObsAnxiety_given_AnxietyT=pm.Dirichlet('p_ObsAnxiety_given_AnxietyT',a=np.ones(3)).reshape((1,-1)).repeat(N,axis=0)
    p_ObsAnxiety_given_AnxietyF=pm.Dirichlet('p_ObsAnxiety_given_AnxietyF',a=np.ones(3)).reshape((1,-1)).repeat(N,axis=0)
    
    ObsAnxiety_probs= pm.math.switch(Anxiety.reshape((-1,1)),p_ObsAnxiety_given_AnxietyT,p_ObsAnxiety_given_AnxietyF)
    ObsAnxiety=pm.Categorical('ObsAnxiety',ObsAnxiety_probs,observed=X_3lvls[:,1])
    
    

test_point={
    'haveSibling':SA_predicts[:,0],
    'Anxiety': SA_predicts[:,1]
}
with basic_model:
    mean_q = pm.find_MAP(start=test_point,
                         maxeval=50000,return_raw=True)

mean_q


# In[23]:


old_neglogprob=mean_q[1]['fun']
for EM_step in range(NumEMSteps):
    shared_p_haveSibling=mean_q[0]['p_haveSibling']
    shared_p_Anxiety=mean_q[0]['p_Anxiety'] 
    shared_p_happyAlone_given_S_A=tt.as_tensor_variable(mean_q[0]['p_happyAlone_given_S_A'])
    shared_p_ObsHaveSibling_given_haveSiblingT=mean_q[0]['p_ObsHaveSibling_given_haveSiblingT']
    shared_p_ObsHaveSibling_given_haveSiblingF=mean_q[0]['p_ObsHaveSibling_given_haveSiblingF']
    shared_p_ObsAnxiety_given_AnxietyT=mean_q[0]['p_ObsAnxiety_given_AnxietyT']  
    shared_p_ObsAnxiety_given_AnxietyF=mean_q[0]['p_ObsAnxiety_given_AnxietyF']

    SA_probs_given_UNK=np.zeros((2,2,2))
    for H in range(2):
        shared_happyAlone=H
        
        basic_model2=pm.Model()
        with basic_model2:
            haveSibling=pm.Bernoulli('haveSibling',shared_p_haveSibling)

            Anxiety=pm.Bernoulli('Anxiety',shared_p_Anxiety)

            happyAlone=pm.Bernoulli('happyAlone',shared_p_happyAlone_given_S_A[haveSibling,Anxiety], observed=shared_happyAlone)

            ObsHaveSibling_probs= pm.math.switch(haveSibling,shared_p_ObsHaveSibling_given_haveSiblingT,shared_p_ObsHaveSibling_given_haveSiblingF)
            ObsHaveSibling=pm.Categorical('ObsHaveSibling',ObsHaveSibling_probs,observed=2)


            ObsAnxiety_probs= pm.math.switch(Anxiety,shared_p_ObsAnxiety_given_AnxietyT,shared_p_ObsAnxiety_given_AnxietyF)
            ObsAnxiety=pm.Categorical('ObsAnxiety',ObsAnxiety_probs,observed=2)
            
            
        for A in range(2):
            for S in range(2):
                SA_probs_given_UNK[H,S,A]=basic_model2.logp({'haveSibling':S,'Anxiety':A})

    SA_probs_given_UNK=np.exp(SA_probs_given_UNK)
    for H in range(2):
        Aprobs=SA_probs_given_UNK.sum(axis=1)[H,:]
        Sprobs=SA_probs_given_UNK.sum(axis=2)[H,:]
        Aprobs=Aprobs/Aprobs.sum()
        Sprobs=Sprobs/Sprobs.sum()
        print(H,Aprobs/Aprobs.sum(),Sprobs/Sprobs.sum())


        unobsS=(y==H)*(X_3lvls[:,0]==2)
        unobsA=(y==H)*(X_3lvls[:,1]==2)
        SA_predicts[unobsS,0]=np.random.choice([0,1],p=Sprobs/Sprobs.sum(),size=unobsS.sum())
        SA_predicts[unobsA,1]=np.random.choice([0,1],p=Aprobs/Aprobs.sum(),size=unobsA.sum())

    test_point={
        'haveSibling':SA_predicts[:,0],
        'Anxiety': SA_predicts[:,1]
    }  

    with basic_model:
        mean_q = pm.find_MAP(start=test_point,
                             maxeval=50000,return_raw=True)

    new_neglogprob=mean_q[1]['fun']
    if abs(new_neglogprob-old_neglogprob)<40:
        break
    old_neglogprob=new_neglogprob
#     print(mean_q)


# ## 2.1 Test model (d) on test data:

# In[24]:


# d) using pymc3 

shared_p_haveSibling=mean_q[0]['p_haveSibling']
shared_p_Anxiety=mean_q[0]['p_Anxiety'] 
shared_p_happyAlone_given_S_A=tt.as_tensor_variable(mean_q[0]['p_happyAlone_given_S_A'])
shared_p_ObsHaveSibling_given_haveSiblingT=mean_q[0]['p_ObsHaveSibling_given_haveSiblingT']
shared_p_ObsHaveSibling_given_haveSiblingF=mean_q[0]['p_ObsHaveSibling_given_haveSiblingF']
shared_p_ObsAnxiety_given_AnxietyT=mean_q[0]['p_ObsAnxiety_given_AnxietyT']  
shared_p_ObsAnxiety_given_AnxietyF=mean_q[0]['p_ObsAnxiety_given_AnxietyF']

shared_ObsAnxiety=theano.shared(0)
shared_ObsHaveSibling=theano.shared(0)

test_model=pm.Model()
with test_model:
    haveSibling=pm.Bernoulli('haveSibling',shared_p_haveSibling)

    Anxiety=pm.Bernoulli('Anxiety',shared_p_Anxiety)

    happyAlone=pm.Bernoulli('happyAlone',shared_p_happyAlone_given_S_A[haveSibling,Anxiety])

    ObsHaveSibling_probs= pm.math.switch(haveSibling,shared_p_ObsHaveSibling_given_haveSiblingT,shared_p_ObsHaveSibling_given_haveSiblingF)
    ObsHaveSibling=pm.Categorical('ObsHaveSibling',ObsHaveSibling_probs,observed=shared_ObsHaveSibling)


    ObsAnxiety_probs= pm.math.switch(Anxiety,shared_p_ObsAnxiety_given_AnxietyT,shared_p_ObsAnxiety_given_AnxietyF)
    ObsAnxiety=pm.Categorical('ObsAnxiety',ObsAnxiety_probs,observed=shared_ObsAnxiety)
    

my_X_test_3lvls=np.zeros((9,2),dtype=np.int)
preds=np.zeros(my_X_test_3lvls.shape)

i=0
for S in range(3):
    for A in range(3):
        my_X_test_3lvls[i,0]=S
        my_X_test_3lvls[i,1]=A
        i+=1
for i in range(my_X_test_3lvls.shape[0]):
    nom=0
    denom=0
    print('------------',my_X_test_3lvls[i,:])
    for H in range(2):
        for S in range(2):
            for A in range(2):
                t_sample_input={}
                t_sample_input['happyAlone']=H
                t_sample_input['haveSibling']=S
                t_sample_input['Anxiety']=A
                
                
                shared_ObsHaveSibling.set_value(my_X_test_3lvls[i,0])
                shared_ObsAnxiety.set_value(my_X_test_3lvls[i,1])
                
#                 print(H,S,A)
                prob=np.exp(test_model.logp(t_sample_input))
                if H==1:
                    nom+=prob
                    
                denom+=prob
    
    print(nom/denom)

    preds[i,1]=(nom/denom)  
    preds[i,0]=1-(nom/denom) 

logloss=0
for i in range(my_X_test_3lvls.shape[0]):
    for my_y in range(2):
        similar_samples_in_test= (y_test==my_y) * np.all(X_test_3lvls==my_X_test_3lvls[i,:],axis=1)
        num_sim_samples_in_test=np.sum(similar_samples_in_test)

        if my_y == 1:
            logloss += -np.log(preds[i,1])*num_sim_samples_in_test
        else:
            logloss += -np.log(1-preds[i,1])*num_sim_samples_in_test
        
print('logloss of trained (d) on the data:',logloss/len(y_test))
model_comparison['trained (d)']={'logloss':logloss/len(y_test)}
    


# # Test best model (D-LR) on the data

# In[25]:


# test best model on the data   
shared_p_haveSibling=haveSibling_gt
shared_p_Anxiety=Anxiety_gt
shared_p_happyAlone_given_S_A=happyAlone_given_S_A
shared_p_ObsHaveSibling_given_haveSiblingT=ObsHaveSibling_given_HaveSiblingT_gt
shared_p_ObsHaveSibling_given_haveSiblingF=ObsHaveSibling_given_HaveSiblingF_gt
shared_p_ObsAnxiety_given_AnxietyT=ObsAnxiety_given_AnxietyT_gt
shared_p_ObsAnxiety_given_AnxietyF=ObsAnxiety_given_AnxietyF_gt

shared_ObsAnxiety=theano.shared(0)
shared_ObsHaveSibling=theano.shared(0)

test_model=pm.Model()
with test_model:
    haveSibling=pm.Bernoulli('haveSibling',shared_p_haveSibling)

    Anxiety=pm.Bernoulli('Anxiety',shared_p_Anxiety)

    happyAlone=pm.Bernoulli('happyAlone',shared_p_happyAlone_given_S_A[haveSibling,Anxiety])

    ObsHaveSibling_probs= pm.math.switch(haveSibling,shared_p_ObsHaveSibling_given_haveSiblingT,shared_p_ObsHaveSibling_given_haveSiblingF)
    ObsHaveSibling=pm.Categorical('ObsHaveSibling',ObsHaveSibling_probs,observed=shared_ObsHaveSibling)


    ObsAnxiety_probs= pm.math.switch(Anxiety,shared_p_ObsAnxiety_given_AnxietyT,shared_p_ObsAnxiety_given_AnxietyF)
    ObsAnxiety=pm.Categorical('ObsAnxiety',ObsAnxiety_probs,observed=shared_ObsAnxiety)
    

my_X_test_3lvls=np.zeros((9,2),dtype=np.int)
preds=np.zeros(my_X_test_3lvls.shape)

i=0
for S in range(3):
    for A in range(3):
        my_X_test_3lvls[i,0]=S
        my_X_test_3lvls[i,1]=A
        i+=1
        
for i in range(my_X_test_3lvls.shape[0]):
    nom=0
    denom=0
    print('------------',my_X_test_3lvls[i,:])
    for H in range(2):
        for S in range(2):
            for A in range(2):
                t_sample_input={}
                t_sample_input['happyAlone']=H
                t_sample_input['haveSibling']=S
                t_sample_input['Anxiety']=A
                
                
                shared_ObsHaveSibling.set_value(my_X_test_3lvls[i,0])
                shared_ObsAnxiety.set_value(my_X_test_3lvls[i,1])
                
#                 print(H,S,A)
                prob=np.exp(test_model.logp(t_sample_input))
                if H==1:
                    nom+=prob
                    
                denom+=prob
    
    print(nom/denom)

    preds[i,1]=(nom/denom)  
    preds[i,0]=1-(nom/denom) 

logloss=0
for i in range(my_X_test_3lvls.shape[0]):
    for my_y in range(2):
        similar_samples_in_test= (y_test==my_y) * np.all(X_test_3lvls==my_X_test_3lvls[i,:],axis=1)
        num_sim_samples_in_test=np.sum(similar_samples_in_test)

        if my_y == 1:
            logloss += -np.log(preds[i,1])*num_sim_samples_in_test
        else:
            logloss += -np.log(1-preds[i,1])*num_sim_samples_in_test
        
print('logloss of (d)-LR best params on the data:',logloss/len(y_test))
model_comparison['(d)-LR best params']={'logloss':logloss/len(y_test)}


# In[27]:


import json
print(model_comparison['lR+/-']['logloss'],
      model_comparison['CategoricalNB_3lvls']['logloss'],
      model_comparison['trained (c)']['logloss'],
      model_comparison['trained (d)']['logloss'],
      model_comparison['classicalNB']['logloss'],
     )

final_comparison_results={'ground truth':'DLR',
                          'ground truth probabilities':{
                                'haveSibling_gt':haveSibling_gt,

                                'Anxiety_gt':Anxiety_gt,

                                'happyAlone_given_S_A':happyAlone_given_S_A.eval().tolist(),

                                'ObsHaveSiblingT_given_HaveSiblingT_gt':ObsHaveSibling_given_HaveSiblingT_gt.tolist(),
                                'ObsHaveSiblingT_given_HaveSiblingF_gt':ObsHaveSibling_given_HaveSiblingF_gt.tolist(),

                                'ObsAnxietyT_given_AnxietyT_gt':ObsAnxiety_given_AnxietyT_gt.tolist(),
                                'ObsAnxietyT_given_AnxietyF_gt':ObsAnxiety_given_AnxietyF_gt.tolist(),
                          },
                          'model_loglosses':{},
                         }
for m in model_comparison:
    final_comparison_results['model_loglosses'][m]=model_comparison[m]['logloss']

with open('./results/dLR_%d.json'%(run_id),'w') as f:
    json.dump(final_comparison_results,f)
    
# with open('./x.json','w') as f:
#     json.dump(final_comparison_results,f)


# In[ ]:





# In[ ]:





# In[ ]:




