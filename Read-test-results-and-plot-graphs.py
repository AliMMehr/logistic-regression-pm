#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import os
import matplotlib.pyplot as plt


# In[91]:


mypath='results'
all_files = [f for f in listdir(mypath)]

jsonfiles=[]
for f in all_files:
    if f.endswith('.json'):
        jsonfiles.append(f)


# In[ ]:





# In[92]:


# len(jsonfiles)


# In[93]:


columns=['ground truth','model_loglosses classicalNB','model_loglosses lR+/-','model_loglosses trained (c)','model_loglosses trained (d)','model_loglosses CategoricalNB_3lvls']
logloss_to_drop=['model_loglosses CategoricalNB','model_loglosses BernoulliNB','model_loglosses MultinomialNB']
df=pd.DataFrame(columns=columns)
for fname in jsonfiles:
    with open(join(mypath,fname),'r') as f:
        run_details=json.load(f)
#         print(run_details['model_loglosses'])
        new_run_details={}
        for k,v in run_details['ground truth probabilities'].items():
            if isinstance(v,list):
                new_run_details['ground_truth '+k]=v
            else:
                new_run_details['ground_truth '+k]=round(v,3)
        for k,v in run_details['model_loglosses'].items():
            if k in logloss_to_drop:
                continue
            new_run_details['model_loglosses '+k]=round(v,6)  
#             print(new_run_details['model_loglosses '+k])
            
        new_run_details['ground truth']=run_details['ground truth']
        
        df=df.append(new_run_details,ignore_index=True)
df.to_csv('compare_LR+-_with_others.csv')
# new_run_details


# In[94]:


# df.columns


# In[95]:


df=df.set_index('ground truth')


# In[96]:


# df


# In[97]:


# df.index.unique()


# In[106]:


plt.figure()
gt2bestmodel={'C':'model_loglosses (c) best param','D':'model_loglosses best param(d)','DLR':'model_loglosses (d)-LR best params'}
all_models=['model_loglosses classicalNB','model_loglosses trained (c)', 'model_loglosses trained (d)','model_loglosses lR+/-']
xticklabels=['Naive Bayes','trained\nmodel (c)','trained\nmodel (d)-full CPT','LR$\pm$']

for gt in df.index.unique():
    print('-----------------------',gt)
    fig, ax = plt.subplots()
    ax.plot([0.6,3.4],[0,0],linewidth=0.5,alpha=0.5)
    
    
    data=[]
    
    gt_df=df.loc[gt]
    gt_bestmodel=gt2bestmodel[gt]
    for m in all_models:
        m_compared_to_best=(gt_df[m]-gt_df[gt_bestmodel])/gt_df[gt_bestmodel]*100
        m_compared_to_best=m_compared_to_best.tolist()
        data.append(m_compared_to_best)
        print(m,np.mean(m_compared_to_best))
        
        
    ax.boxplot(data) 
    gt2name={'C':'C','D':'D4', 'DLR':'DLR'}
    ax.set_title(gt2name[gt]+' Datasets')
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel('percent increase in (worsening of) logloss\ncompared to ground truth model')
    ax.set_xlabel('models that are compared to the ground truth model')
#     plt.add_axes([0,0.1,0,0])
    plt.savefig('compare_LR+-_to_others-%s.png'%(gt),dpi=250,bbox_inches='tight')
    plt.ylim(-2,30)
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




