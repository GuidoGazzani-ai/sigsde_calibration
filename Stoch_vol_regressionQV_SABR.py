

# # <font color=”darkblue”> Path Calibration </font>

# * Simulate a SABR-type model under $P$:
# \begin{align*}
# dX_t&= X_t\mu dt +X_t V_tdB_t\\
# dV_t&= \kappa(\theta-V_t) dt+ \alpha V_t dW_{t}
# \end{align*}
# with $d\langle B, W\rangle_t = \rho dt$.
# 
# * Estimate the trajectories of $(B_t^Q, W_t^Q)$ by estimating $\Sigma_{11,t}=d\langle X, X \rangle_t$ and $\Sigma_{22,t}=d\langle V, V \rangle_t$ and computing  
# \begin{equation*}
# dB_t^Q = \frac{dX_t}{\sqrt{\Sigma_{11,t}}}, \quad dW_t^Q = \frac{dV_t}{\sqrt{\Sigma_{22,t}}},
# \end{equation*}
# * Compute $\mathbb{\widehat{Y}}^Q_t$ and find $\ell_I$ such that
# \begin{equation*}
# \sqrt{\Sigma_{11,t}}\approx \sum_{0 \leq |I|\leq n} \ell_I \langle e_I,\mathbb{\widehat{Y}}^Q_t \rangle 
# \end{equation*}
# * Compare then 
# \begin{equation*}
# X_t(\ell)= X_0 + \int_0^t\sum_{0 < |I|\leq n} \ell_I \langle e_I,\mathbb{\widehat{Y}}^Q_s \rangle dB^Q_{s} 
# \end{equation*}
# with the orginally simulated trajectory $X$.


# In[1]:
#Call packages
import itertools as itt  
import iisignature
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import esig
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from functools import partial
from bokeh.io import show, output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure, show
from bokeh.layouts import column
import pandas as pd
from sklearn.metrics import mean_squared_error 
from bokeh.embed import file_html
import chart_studio.plotly as py
import time as tm
import signatory



# In[2]:


output_notebook()


# In[70]:


hours=8
minutes=60*5
days=1095
N, initial_price, vol = days*hours*minutes, 1, 0.25
mu=0.001 #change this to add drift to price path
t_0=0
t_final=1
initial_vol=0.08
initial_price=1

alpha=0.25
theta=0.1
kappa=0.17
rho=-0.5


# In[71]:


def number_of_parameters_gen(n,comp_of_path):
    'Inputs: n  (integer>0), comp_of_path (integer>0)'
    'Output: integer >0, i.e., number of the parameters of the signature'
    def sum_of_powers(x,y):
        if y<=0: return 1
        return x**y + sum_of_powers(x,y-1)
    return sum_of_powers(comp_of_path,n)

def zero_list_maker(n):
    'Produce a list of zeros'
    listofzeros = [0] * n
    return listofzeros

def correlated_bms_correct(N,rho,t_final,t_0):
    'Simulate two one dimensional \rho-correlated Brownian motions from t_0 to t_final with N steps'
    time_grid=np.linspace(0,t_final,num=N,retstep=True)[0]
    dt=np.abs(time_grid[0]-time_grid[1])
    dB = np.random.normal(0, np.sqrt(dt), N)
    dB[0]=0
    dW = rho*dB+np.sqrt(1-(rho)**2)*np.random.normal(0,np.sqrt(dt),N)
    dW[0]=0
    B = np.cumsum(dB)
    W = np.cumsum(dW)
    return time_grid, B, W


# ## <font color=”darkblue”> Simulate a SABR trajectory </font>

# In[72]:


def Simulate_SABR(N,initial_price,mu,alpha,kappa,theta,initial_vol,rho,t_final,t_0):
    'Simulate a SABR model with N points from t_0 to t_final and the chosen parameters'
    X=np.zeros(N)
    V=np.zeros(N)
    X[0], V[0]= initial_price, initial_vol
    t,B,W=correlated_bms_correct(N,rho,t_final,t_0)
    noises=np.stack([t,B,W],axis=0).transpose()
    for k in range(N-1):
        V[k+1]=V[k]+kappa*(theta-V[k])*(1/N)+(alpha)*(V[k])*(W[k+1]-W[k]) 
        X[k+1]=X[k]+(X[k])*(mu)*(1/N)+(V[k])*(X[k])*(B[k+1]-B[k])
    return X,V,B,W,t,noises

X,V,B,W,time,noises=Simulate_SABR(N,initial_price,mu,alpha,kappa,theta,initial_vol,rho,t_final,t_0)

p1 = figure(width=500, height=350, title='Simulation of the SABR type process')
p1.line(time, X, color="royalblue", legend_label='Price')

p2 = figure(width=500, height=350, title='Simulation of the underlying volatility process')
p2.line(time, V, color="tomato", legend_label='Volatility')


# put all the plots in an HBox
p = row(p1,p2)

show(p)


# ## <font color=”darkblue”> Estimate the QV of the volatility and get its Brownian motion: </font>

# In[73]:


def get_QV_vol(V,W,days,hours,minutes,alpha):
    'Estimate the Quadratic variation of the volatility process'
    QV_hat_vol=[]
    for k in range(days):
        QV_hat_vol.append(days*np.sum(np.diff(V[k*(hours*minutes):(k+1)*(hours*minutes)])**2))
    QV_hat_vol=np.array(QV_hat_vol).flatten()
    V_daily=np.array([V[hours*minutes*k] for k in range(days)])
    W_daily=np.array([W[hours*minutes*k] for k in range(days)])
    QV_real_vol=(alpha**2)*(V_daily**2)
    time_days=np.array(range(days))
    time_days=time_days/len(time_days)
    p = figure(width=500, height=350,title='Quadratic variation (QV) estimation')
    p1 = p.line(time_days,QV_real_vol, legend_label='QV real of Vol', line_color='royalblue')
    p2 = p.line(time_days,QV_hat_vol,legend_label='QV estimated of Vol',line_color='tomato')
    
    return QV_hat_vol, QV_real_vol, V_daily, W_daily, time_days, p

def get_BM_vol(V_daily, W_daily, QV_hat_vol ,QV_real_vol,time_days):
    'Extract the BM driving the volatility process; it is important that a sufficient number of points for the estimation 
    'are provided, see the examples'
    sqrt_QV_hat=np.sqrt(QV_hat_vol)
    new_increments_BM=[np.diff(V_daily)[k]/sqrt_QV_hat[k] for k in range(len(sqrt_QV_hat)-1)]
    W_daily_ext=np.cumsum(new_increments_BM)
    W_daily_ext=np.insert(W_daily_ext,0,0)
    q = figure(width=500, height=350,title='Compare Brownian motions')
    q1 = q.line(time_days,W_daily, legend_label='Real BM', line_color='royalblue')
    q2 = q.line(time_days,W_daily_ext,legend_label='Estimated BM',line_color='tomato')
    
    return W_daily, W_daily_ext,q

QV_hat_vol, QV_real_vol, V_daily, W_daily, time_days, p = get_QV_vol(V,W,days,hours,minutes,alpha)
W_daily, W_daily_ext,q = get_BM_vol(V_daily, W_daily, QV_hat_vol ,QV_real_vol,time_days)

f=row(p,q)
show(f)


# ## <font color=”darkblue”> Estimate the QV of the price and get its Brownian motion: </font>

# In[74]:


def get_QV_price(X,B,V_daily,days,hours,minutes):
    'Estimate the Quadratic variation of the price process'
    QV_hat_price=[]
    for k in range(days):
        QV_hat_price.append(days*np.sum(np.diff(X[k*(hours*minutes):(k+1)*(hours*minutes)])**2))
    QV_hat_price=np.array(QV_hat_price).flatten()
    X_daily=np.array([X[hours*minutes*k] for k in range(days)])
    B_daily=np.array([B[hours*minutes*k] for k in range(days)])
    QV_real_price=(X_daily*V_daily)**2
    time_days=np.array(range(days))
    time_days=time_days/len(time_days)
    r = figure(width=500, height=350,title='Quadratic variation (QV) estimation')
    r1 = r.line(time_days,QV_real_price,legend_label='QV real of price', line_color='royalblue')
    r2 = r.line(time_days,QV_hat_price,legend_label='QV estimated of price',line_color='tomato')
    
    return QV_hat_price, QV_real_price, X_daily, B_daily, time_days, r

def get_BM_price(X_daily, B_daily, QV_hat_price,QV_real_price,time_days):
    'Extract the BM driving the price process; it is important that a sufficient number of points for the estimation'
    'are provided, see the examples'
    sqrt_QV_hat=np.sqrt(QV_hat_price)
    new_increments_BM=[np.diff(X_daily)[k]/sqrt_QV_hat[k] for k in range(len(sqrt_QV_hat)-1)]
    B_daily_ext=np.cumsum(new_increments_BM)
    B_daily_ext=np.insert(B_daily_ext,0,0)
    s = figure(width=500, height=350,title='Compare Brownian motions')
    s1 = s.line(time_days,B_daily, legend_label='Real BM', line_color='royalblue')
    s2 = s.line(time_days,B_daily_ext,legend_label='Estimated BM',line_color='tomato')
    
    return B_daily, B_daily_ext,s

QV_hat_price, QV_real_price, X_daily, B_daily, time_days, r = get_QV_price(X,B,V_daily,days,hours,minutes)
B_daily, B_daily_ext,s= get_BM_price(X_daily, B_daily, QV_hat_price,QV_real_price,time_days)

f1=row(r,s)
show(f1)


# ## <font color=”darkblue”> Compute $\mathbb{\widehat{Y}}_{t}$ </font>

# In[75]:


def augment_noises(B,W):
    'Augment the two Brownian motions'
    t_scaled=[k/len(W) for k in range(len(W))]
    noises=np.array([[t_scaled[k],B[k],W[k]] for k in range(len(W))])
    return noises
def build_sig_df_sv(order_model,noises,N):
    'This function builds a data frame (sig_df) with columns keys; namely the signature at all N grid points of the noises'
    order_Signature=order_model+1
    nbr_components=noises.shape[1]
    noises_torchified=torch.from_numpy(noises).unsqueeze(0)
    sig=signatory.signature(noises_torchified,order_Signature,stream=True,basepoint=True,scalar_term=True)
    keys=esig.sigkeys(nbr_components,order_Signature).strip().split(" ")
    sig_df=pd.DataFrame(sig.squeeze(0).numpy(), columns=keys)
    return sig_df,keys,sig   

def tilde_transformation(word):
    'This simple function computes the tilde transformation for a word (list) of integers e.g. [1,2]'
    word_aux=word.copy()
    word_aux_2=word.copy()
    if word[-1]==1:
        word.append(2)
        word_aux.append(3)
        return word, word_aux
    if word[-1]!=1:
        word_aux.append(2)
        word_aux_2.append(3)
        word[-1]=1
        return word_aux, word_aux_2, word

def e_tilde_part2_new(words_as_lists):
    'Auxiliary function for the tilde tranformation'
    tilde=[list(tilde_transformation(words_as_lists[k])) for k in np.array(range(len(words_as_lists)))[1:]] #we skip the empty word
    return tilde

def from_tilde_to_strings_new(tilde): 
    'This function returns the tilde transformation components in the form of strings as needed for the data frame'
    for k in range(len(tilde)):
        if len(tilde[k])==2:
            tilde[k][0]=str(tuple(tilde[k][0])).replace(" ","")
            tilde[k][1]=str(tuple(tilde[k][1])).replace(" ","")
        elif (len(tilde[k])==3 and len(tilde[k][-1])==1):
            tilde[k][0]=str(tuple(tilde[k][0])).replace(" ","")
            tilde[k][1]=str(tuple(tilde[k][1])).replace(" ","")
            tilde[k][-1]=str(tuple(tilde[k][-1])).replace(",","")
        elif len(tilde[k])==3:
            tilde[k][0]=str(tuple(tilde[k][0])).replace(" ","")
            tilde[k][1]=str(tuple(tilde[k][1])).replace(" ","")
            tilde[k][2]=str(tuple(tilde[k][2])).replace(" ","")
    return tilde

def get_tilde_df_debug(Sig_data_frame,new_tilde,keys_n,keys_n1,comp_of_path,rho,y):
    'Implementation of the tilde-tranformation applied to the sig_df in the case of (t,B,W) as augmented noises'
    'It returns a dataframe similar to sig_df but where the components are the result of the tilde-transformation'
    aus_B=[]
    y=[[eval(key)] if isinstance(eval(key), int) else list(eval(key)) for key in keys_n]
    for k in range(len(y)):
        if k==0:
            aus_B.insert(0,Sig_data_frame['(2)'])
        if (k>0 and y[k][-1]==1):
            aus_B.append(Sig_data_frame[new_tilde[k-1][0]])
           
        if (k>0 and y[k][-1]==2):
            aus_B.append(Sig_data_frame[new_tilde[k-1][0]]-0.5*Sig_data_frame[new_tilde[k-1][2]])
        
        if (k>0 and y[k][-1]==3):
            aus_B.append(Sig_data_frame[new_tilde[k-1][0]]-rho*0.5*Sig_data_frame[new_tilde[k-1][2]])
           
    new_keys_B=[keys_n1[k]+str('~B') for k in range(len(y))]
    new_dictionary_B={key:series for key,series in zip(new_keys_B,aus_B)}        
    transformed_data_frame_B=pd.DataFrame(new_dictionary_B)
    return transformed_data_frame_B, new_keys_B


# In[76]:


order_model=2

noises_x=augment_noises(B_daily_ext,W_daily_ext) #x is the extrapolated one
noises_y=augment_noises(B_daily,W_daily)         #y the real one


Sig_data_frame_x, keys, Sig_x=build_sig_df_sv(order_model,noises_x,days)
Sig_data_frame_y, keys, Sig_y=build_sig_df_sv(order_model,noises_y,days)


# In[77]:


Sig_data_frame_x


# In[78]:



#Regression task x
reg_x=Lasso(alpha=0.00001).fit(Sig_data_frame_x,np.sqrt(QV_hat_price))
predictions_x = reg_x.predict(Sig_data_frame_x) 
#Regression task y

reg_y=Lasso(alpha=0.00001).fit(Sig_data_frame_y,np.sqrt(QV_hat_price))
predictions_y = reg_y.predict(Sig_data_frame_y) 


q = figure(width=500, height=350,title='Extrapolated BM')
q1 = q.line(time_days,predictions_x, legend_label='Sig-SDE', line_color='royalblue')
q2 = q.line(time_days,np.sqrt(QV_hat_price), legend_label='QV', line_color='tomato')

p = figure(width=500, height=350,title='Real BM')
p1 = p.line(time_days,predictions_y, legend_label='Sig-SDE', line_color='royalblue')
p2 = p.line(time_days,np.sqrt(QV_hat_price), legend_label='QV', line_color='tomato')


g = row(p,q)

show(g)


# In[79]:


reg_x.coef_


# In[80]:


reg_x.intercept_


# In[81]:


print('MSE of the QV with Extrapolated BMs ',mean_squared_error(predictions_x,np.sqrt(QV_hat_price)))
print('MSE of the QV with Real BMs ',mean_squared_error(predictions_y,np.sqrt(QV_hat_price)))


# In[82]:


X_approx=np.zeros(days)
X=np.zeros(days)
X_approx_true=np.zeros(days)
X_approx_true[0]=initial_price
X_approx[0], X[0]=initial_price, initial_price
for k in range(days-1):
    X_approx[k+1]=X_approx[k]+(predictions_x[k])*(B_daily_ext[k+1]-B_daily_ext[k])
    X_approx_true[k+1]=X_approx_true[k]+(predictions_x[k])*(B_daily[k+1]-B_daily[k])
    X[k+1]=X[k]+(np.sqrt(QV_real_price[k]))*(B_daily[k+1]-B_daily[k])

f1 = figure(width=700, height=500,title='Calibration on path')
f2 = f1.line(time_days[:int(days*(1/1))],X_approx[:int(days*(1/1))], legend_label='Sig-SDE ex BM', line_color='royalblue') #change where to slice the
f3 = f1.line(time_days[:int(days*(1/1))],X_approx_true[:int(days*(1/1))], legend_label='Sig-SDE true BM', line_color='cornflowerblue')
f4 = f1.line(time_days[:int(days*(1/1))],X[:int(days*(1/1))],legend_label='SABR', line_color='tomato')          #time series accordingly 
show(f1)


# In[100]:


print('MSE with Extrapolated BMs ',mean_squared_error(X_approx[:int(days*(1/1))],X[:int(days*(1/1))]))
print('MSE with True BMs ',mean_squared_error(X_approx_true[:int(days*(1/1))],X[:int(days*(1/1))]))


# ## <font color=”darkblue”> Let's now test this on a new realizations </font>

# In[108]:


from tqdm.auto import tqdm

mse_mc_extrapolated_bms=[]
mse_mc_true_bms=[]

hours=8
minutes=60*5
days=1095
N, initial_price, vol = days*hours*minutes, 1, 0.25
mu=0.001 
t_0=0
t_final=1
initial_vol=0.08

alpha=0.25
theta=0.1
kappa=0.17
rho=-0.5

nbr_test_trials=3
test_window=int(time_days.shape[0]/2)


# In[111]:


for j in tqdm(range(nbr_test_trials)):
    
    X_new,V_new,B_new,W_new,time,noises_new=Simulate_SABR(N,initial_price,mu,alpha,kappa,theta,initial_vol,rho,t_final,t_0)

    #p1 = figure(width=500, height=350, title='Simulation of the SABR type process')
    #p1.line(time, S_new, color="royalblue", legend_label='Price')

    #p2 = figure(width=500, height=350, title='Simulation of the underlying volatility process')
    #p2.line(time, V_new, color="tomato", legend_label='Volatility')


    # put all the plots in an HBox
    #p = row(p1,p2)

    #show(p)
    
    QV_hat_vol_new, QV_real_vol_new, V_daily_new, W_daily_new, time_days, p = get_QV_vol(V_new,W_new,days,hours,minutes,alpha)
    W_daily_new, W_daily_ext_new,q = get_BM_vol(V_daily_new, W_daily_new, QV_hat_vol_new ,QV_real_vol_new,time_days)
    
    #f=row(p,q)
    #show(f)
    
    QV_hat_price_new, QV_real_price_new, X_daily_new, B_daily_new, time_days, r = get_QV_price(X_new,B_new,V_daily_new,days,hours,minutes)
    B_daily_new, B_daily_ext_new,s= get_BM_price(X_daily_new, B_daily_new, QV_hat_price_new,QV_real_price_new,time_days)
    
    #f1=row(r,s)
    #show(f1)
    
    noises_x_new=augment_noises(B_daily_ext_new,W_daily_ext_new) #x is the extrapolated one
    noises_y_new=augment_noises(B_daily_new,W_daily_new) #y is the real one

    Sig_data_frame_x_new, keys, Sig_x=build_sig_df_sv(order_model,noises_x_new,days)
    Sig_data_frame_y_new, keys, Sig_y=build_sig_df_sv(order_model,noises_y_new,days)

    keys_n=esig.sigkeys(noises_x_new.shape[1],order_model).strip().split(" ")
    y=[[eval(key)] if isinstance(eval(key), int) else list(eval(key)) for key in keys_n]
    first_step=e_tilde_part2_new(y)
    new_tilde=from_tilde_to_strings_new(first_step)

    transformed_df_x, new_keys_B=get_tilde_df_debug(Sig_data_frame_x_new,new_tilde,keys_n,keys,3,rho,y)
    transformed_df_y, new_keys_B=get_tilde_df_debug(Sig_data_frame_y_new,new_tilde,keys_n,keys,3,rho,y)
    
    coeff_multi_x=reg_x.coef_[:13]
    coeff_multi_x[0]=reg_x.intercept_

    coeff_multi_y=reg_y.coef_[:13]
    coeff_multi_y[0]=reg_y.intercept_

    pred_x=np.dot(transformed_df_x[new_keys_B],coeff_multi_x)+initial_price
    pred_y=np.dot(transformed_df_y[new_keys_B],coeff_multi_y)+initial_price



    
    mse_mc_extrapolated_bms.append(mean_squared_error(pred_x[:test_window],X_daily_new[:test_window]))
    mse_mc_true_bms.append(mean_squared_error(pred_y[:test_window],X_daily_new[:test_window]))


# In[112]:


print(mse_mc_extrapolated_bms)
print(mse_mc_true_bms)


# In[113]:


from bokeh.models import Legend
f1 = figure(width=800, height=490,title='Out of sample performance of the calibration to the volatility')
f2 = f1.line(time_days[:test_window],pred_x[:test_window], line_color='royalblue') #change where to slice the
f3 = f1.line(time_days[:test_window],pred_y[:test_window], line_color='cornflowerblue') #change where to slice the
f4 = f1.line(time_days[:test_window],X_daily_new[:test_window], line_color='tomato')          #time series accordingly 
legend = Legend(items=[
    ("Sig-SDE ex",   [f2]),
    ("Sig-SDE true",   [f3]),
    ("SABR",[f4])
], location=(-120,360))
f1.add_layout(legend, 'right')
show(f1)  


# In[149]:


import os
os.getcwd()
os.chdir(r'C:\Users\Guido Gazzani\ucloud\Shared\Ripristinati\cluster_sabr1')

traj_outsample_sabr=np.load('traj_outsample_sabr.npy')
learned_traj_sabr=np.load('learned_traj_sabr.npy')
mse_true_bms_insample_sabr=np.load('mse_true_bms_insample_sabr.npy')
mse_true_bms_outsample_sabr=np.load('mse_true_bms_outsample_sabr.npy')
mse_ex_bms_insample_sabr=np.load('mse_ex_bms_insample_sabr.npy')
mse_ex_bms_outsample_sabr=np.load('mse_ex_bms_outsample_sabr.npy')

print(np.mean(mse_ex_bms_outsample_sabr))
print(np.mean(mse_true_bms_outsample_sabr))

sel=np.random.randint(0,10)
traj_1=traj_outsample_sabr[sel]
learned_traj_HE_1=learned_traj_sabr[sel]

p = figure(width=800, height=490,title='Out of sample performance of the calibration to the price path')
r = p.line(time_days[:test_window],traj_1[:test_window], line_color='royalblue')
q = p.line(time_days[:test_window],learned_traj_HE_1[:test_window], line_color='tomato')
legend = Legend(items=[
    ("Sig-SDE",   [r]),
    ("SABR",[q])
], location=(-600,360))

p.add_layout(legend, 'right')
show(p)


# In[150]:


print(mse_ex_bms_insample_sabr) #are switched, my mistake


# In[151]:


print(mse_true_bms_insample_sabr)


# In[116]:


test_window


# In[154]:


p = figure(width=800, height=490,title='Out of sample trajectories from the path calibration')
for k in range(7,8):
    traj_1=traj_outsample_sabr[k]
    learned_traj_HE_1=learned_traj_sabr[k]
    r = p.line(time_days[:int(time_days.shape[0]/2)],traj_1[:int(time_days.shape[0]/2)], line_color='royalblue')
    r1= p.scatter(time_days[:int(time_days.shape[0]/2)],traj_1[:int(time_days.shape[0]/2)], color='royalblue',size=3)
    q = p.line(time_days[:int(time_days.shape[0]/2)],learned_traj_HE_1[:int(time_days.shape[0]/2)], line_color='tomato')
legend = Legend(items=[
    ("Sig-SDE",   [r]),
    ("SABR",[q])
], location=(-600,360))

p.add_layout(legend, 'right')
show(p)  







# # Test on the single traj

# In[86]:


X_new,V_new,B_new,W_new,time,noises_new=Simulate_SABR(N,initial_price,mu,alpha,kappa,theta,initial_vol,rho,t_final,t_0)

p1 = figure(width=500, height=350, title='Simulation of the SABR type process')
p1.line(time, X_new, color="royalblue", legend_label='Price')

p2 = figure(width=500, height=350, title='Simulation of the underlying volatility process')
p2.line(time, V_new, color="tomato", legend_label='Volatility')


# put all the plots in an HBox
p = row(p1,p2)

show(p)


# In[87]:


QV_hat_vol_new, QV_real_vol_new, V_daily_new, W_daily_new, time_days, p = get_QV_vol(V_new,W_new,days,hours,minutes,alpha)
W_daily_new, W_daily_ext_new,q = get_BM_vol(V_daily_new, W_daily_new, QV_hat_vol_new ,QV_real_vol_new,time_days)

f=row(p,q)
show(f)


# In[88]:


QV_hat_price_new, QV_real_price_new, X_daily_new, B_daily_new, time_days, r = get_QV_price(X_new,B_new,V_daily_new,days,hours,minutes)
B_daily_new, B_daily_ext_new,s= get_BM_price(X_daily_new, B_daily_new, QV_hat_price_new,QV_real_price_new,time_days)

f1=row(r,s)
show(f1)


# In[89]:


noises_x_new=augment_noises(B_daily_ext_new,W_daily_ext_new) #x is the extrapolated one
noises_y_new=augment_noises(B_daily_new,W_daily_new) 


# In[90]:


Sig_data_frame_x_new, keys, Sig_x=build_sig_df_sv(order_model,noises_x_new,days)
Sig_data_frame_y_new, keys, Sig_y=build_sig_df_sv(order_model,noises_y_new,days)


# In[91]:


keys_n=esig.sigkeys(noises_x_new.shape[1],order_model).strip().split(" ")
y=[[eval(key)] if isinstance(eval(key), int) else list(eval(key)) for key in keys_n]
first_step=e_tilde_part2_new(y)
new_tilde=from_tilde_to_strings_new(first_step)


# In[92]:


transformed_df_x, new_keys_B=get_tilde_df_debug(Sig_data_frame_x_new,new_tilde,keys_n,keys,3,rho,y)
transformed_df_y, new_keys_B=get_tilde_df_debug(Sig_data_frame_y_new,new_tilde,keys_n,keys,3,rho,y)


# In[93]:


transformed_df_x[new_keys_B]


# In[94]:


reg_x.coef_


# In[95]:


coeff_multi_x=reg_x.coef_[:13]
coeff_multi_x[0]=reg_x.intercept_


# In[96]:


coeff_multi_y=reg_y.coef_[:13]
coeff_multi_y[0]=reg_y.intercept_


# In[97]:


pred_x=np.dot(transformed_df_x[new_keys_B],coeff_multi_x)+initial_price
pred_y=np.dot(transformed_df_y[new_keys_B],coeff_multi_y)+initial_price



# In[98]:


from bokeh.models import Legend


# In[101]:


test_window=int(time_days.shape[0]/2)


# In[ ]:





# In[102]:


from bokeh.models import Legend
f1 = figure(width=800, height=490,title='Out of sample performance of the calibration to the volatility')
f2 = f1.line(time_days[:test_window],pred_x[:test_window], line_color='royalblue') #change where to slice the
f3 = f1.line(time_days[:test_window],pred_y[:test_window], line_color='cornflowerblue') #change where to slice the
f4 = f1.line(time_days[:test_window],X_daily_new[:test_window], line_color='tomato')          #time series accordingly 
legend = Legend(items=[
    ("Sig-SDE ex",   [f2]),
    ("Sig-SDE true",   [f3]),
    ("SABR",[f4])
], location=(-120,360))
f1.add_layout(legend, 'right')
show(f1)  


# In[103]:


print('MSE with Extrapolated BMs ',mean_squared_error(pred_x[:test_window],X_daily_new[:test_window]))
print('MSE with True BMs ',mean_squared_error(pred_y[:test_window],X_daily_new[:test_window]))

