

# # <font color=”darkblue”> Path Calibration </font>

# * Simulate a Heston-type model under $P$:
# \begin{align*}
# dS_t&= S_t\mu dt + S_t \sqrt{V_t}dB_t\\
# dV_t&= \kappa(\theta-V_t) dt+ \alpha \sqrt{V_t} dW_{t}
# \end{align*}
# with $d\langle B, W\rangle_t = \rho dt$.
# 
# * Estimate the trajectories of $(B_t^Q, W_t^Q)$ by estimating $\Sigma_{11,t}=d\langle S, S \rangle_t$ and $\Sigma_{22,t}=d\langle V, V \rangle_t$ and computing  
# \begin{equation*}
# dB_t^Q = \frac{dS_t}{\sqrt{\Sigma_{11,t}}}, \quad dW_t^Q = \frac{dV_t}{\sqrt{\Sigma_{22,t}}},
# \end{equation*}
# * Compute $\mathbb{\widehat{Y}}^Q_t$ and find $\ell_I$ such that for all $t\ge0$
# \begin{equation*}
# S_{t}\approx \ell_{\emptyset}+\sum_{0 < |I|\leq n} \ell_I \langle \tilde{e}_I,\mathbb{\widehat{Y}}^Q_t \rangle 
# \end{equation*}
# where $Y^{Q}=(B^{Q},W^{Q})$.

# Fo

# #### <font color=”darkblue”> Load packages </font>

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


# In[3]:


def correlated_bms_correct(N,rho,t_final,t_0):
    time_grid=np.linspace(0,t_final,num=N,retstep=True)[0]
    dt=np.abs(time_grid[0]-time_grid[1])
    dB = np.random.normal(0, np.sqrt(dt), N)
    dB[0]=0
    dW = rho*dB+np.sqrt(1-(rho)**2)*np.random.normal(0,np.sqrt(dt),N)
    dW[0]=0
    B = np.cumsum(dB)
    W = np.cumsum(dW)
    return time_grid, B, W


# In[4]:


N=365
rho=-0.5
t_final, t_0= 1, 0
t,B,W=correlated_bms_correct(N,rho,t_final,t_0)
figure1 = figure(width=700, height=500,title='Correlated Brownians')
figure1.line(t, B, legend_label="B",color='royalblue')
figure1.line(t, W, legend_label="W",color='tomato')
figure1.legend.location = 'bottom_right'
show(figure1)


# #### <font color=”darkblue”> Additional auxiliar functions </font>

# In[5]:


def number_of_parameters_gen(n,comp_of_path):
    def sum_of_powers(x,y):
        if y<=0: return 1
        return x**y + sum_of_powers(x,y-1)
    return sum_of_powers(comp_of_path,n)

def zero_list_maker(n):
    listofzeros = [0] * n
    return listofzeros


# In[6]:


def tilde_transformation(word):
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
    tilde=[list(tilde_transformation(words_as_lists[k])) for k in np.array(range(len(words_as_lists)))[1:]] #we skip the empty word
    return tilde

def from_tilde_to_strings_new(tilde): 
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
    return transformed_data_frame_B,new_keys_B


# # <font color=”darkblue”> Regression on the price </font>
# 
# 
# 

# In[7]:


def Simulate_Heston(N,initial_price,mu,alpha,kappa,theta,initial_vol,rho,t_final,t_0):
    S=np.zeros(N)
    V=np.zeros(N)
    S[0], V[0]= initial_price, initial_vol
    t,B,W=correlated_bms_correct(N,rho,t_final,t_0)
    noises=np.stack([t,B,W],axis=0).transpose()
    for k in range(N-1):
        V[k+1]=V[k]+kappa*(theta-V[k])*(1/N)+(alpha)*(np.sqrt(V[k]))*(W[k+1]-W[k]) 
        S[k+1]=S[k]+(S[k])*(mu)*(1/N)+(np.sqrt(V[k]))*(S[k])*(B[k+1]-B[k])
    return S,V,B,W,t,noises


# In[8]:


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
theta=0.15
kappa=0.5
rho=-0.5




X,V,B,W,time,noises=Simulate_Heston(N,initial_price,mu,alpha,kappa,theta,initial_vol,rho,t_final,t_0)

p1 = figure(width=500, height=350, title='Simulation of the Heston type process')
p1.line(time, X, color="royalblue", legend_label='Price')

p2 = figure(width=500, height=350, title='Simulation of the underlying volatility process')
p2.line(time, V, color="tomato", legend_label='Volatility')

# put all the plots in an HBox
p = row(p1,p2)

show(p)


# In[9]:


if 2*kappa*(theta)>alpha**2:
    print('Feller condition is satisfied')
else:
    print('Feller condition is not satisfied')


# In[10]:


def get_QV_vol(V,W,days,hours,minutes,alpha):
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
    sqrt_QV_hat=np.sqrt(QV_hat_vol)
    new_increments_BM=[np.diff(V_daily)[k]/sqrt_QV_hat[k] for k in range(len(sqrt_QV_hat)-1)]
    W_daily_ext=np.cumsum(new_increments_BM)
    W_daily_ext=np.insert(W_daily_ext,0,0)
    q = figure(width=500, height=350,title='Compare Brownian motions')
    q1 = q.line(time_days,W_daily, legend_label='Real BM', line_color='royalblue')
    q2 = q.line(time_days,W_daily_ext,legend_label='Estimated BM',line_color='tomato')
    
    return W_daily, W_daily_ext,q


# In[11]:


def get_QV_vol_Heston(V,W,days,hours,minutes,alpha):
    QV_hat_vol=[]
    for k in range(days):
        QV_hat_vol.append(days*np.sum(np.diff(V[k*(hours*minutes):(k+1)*(hours*minutes)])**2))
    QV_hat_vol=np.array(QV_hat_vol).flatten()
    V_daily=np.array([V[hours*minutes*k] for k in range(days)])
    W_daily=np.array([W[hours*minutes*k] for k in range(days)])
    QV_real_vol=(alpha**2)*(V_daily)
    time_days=np.array(range(days))
    time_days=time_days/len(time_days)
    p = figure(width=500, height=350,title='Quadratic variation (QV) estimation')
    p1 = p.line(time_days,QV_real_vol, legend_label='QV real of Vol', line_color='royalblue')
    p2 = p.line(time_days,QV_hat_vol,legend_label='QV estimated of Vol',line_color='tomato')
    
    return QV_hat_vol, QV_real_vol, V_daily, W_daily, time_days, p


# In[12]:


QV_hat_vol, QV_real_vol, V_daily, W_daily, time_days, p = get_QV_vol_Heston(V,W,days,hours,minutes,alpha)
W_daily, W_daily_ext,q = get_BM_vol(V_daily, W_daily, QV_hat_vol ,QV_real_vol,time_days)

f=row(p,q)
show(f)


# In[13]:


def get_QV_price(X,B,V_daily,days,hours,minutes):
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
    sqrt_QV_hat=np.sqrt(QV_hat_price)
    new_increments_BM=[np.diff(X_daily)[k]/sqrt_QV_hat[k] for k in range(len(sqrt_QV_hat)-1)]
    B_daily_ext=np.cumsum(new_increments_BM)
    B_daily_ext=np.insert(B_daily_ext,0,0)
    s = figure(width=500, height=350,title='Compare Brownian motions')
    s1 = s.line(time_days,B_daily, legend_label='Real BM', line_color='royalblue')
    s2 = s.line(time_days,B_daily_ext,legend_label='Estimated BM',line_color='tomato')
    
    return B_daily, B_daily_ext,s


# In[14]:


def get_QV_price_Heston(X,B,V_daily,days,hours,minutes):
    QV_hat_price=[]
    for k in range(days):
        QV_hat_price.append(days*np.sum(np.diff(X[k*(hours*minutes):(k+1)*(hours*minutes)])**2))
    QV_hat_price=np.array(QV_hat_price).flatten()
    X_daily=np.array([X[hours*minutes*k] for k in range(days)])
    B_daily=np.array([B[hours*minutes*k] for k in range(days)])
    QV_real_price=V_daily*(X_daily)**2
    time_days=np.array(range(days))
    time_days=time_days/len(time_days)
    r = figure(width=500, height=350,title='Quadratic variation (QV) estimation')
    r1 = r.line(time_days,QV_real_price,legend_label='QV real of price', line_color='royalblue')
    r2 = r.line(time_days,QV_hat_price,legend_label='QV estimated of price',line_color='tomato')
    
    return QV_hat_price, QV_real_price, X_daily, B_daily, time_days, r

def get_BM_price(X_daily, B_daily, QV_hat_price,QV_real_price,time_days):
    sqrt_QV_hat=np.sqrt(QV_hat_price)
    new_increments_BM=[np.diff(X_daily)[k]/sqrt_QV_hat[k] for k in range(len(sqrt_QV_hat)-1)]
    B_daily_ext=np.cumsum(new_increments_BM)
    B_daily_ext=np.insert(B_daily_ext,0,0)
    s = figure(width=500, height=350,title='Compare Brownian motions')
    s1 = s.line(time_days,B_daily, legend_label='Real BM', line_color='royalblue')
    s2 = s.line(time_days,B_daily_ext,legend_label='Estimated BM',line_color='tomato')
    
    return B_daily, B_daily_ext,s


# In[15]:


QV_hat_price, QV_real_price, X_daily, B_daily, time_days, r = get_QV_price_Heston(X,B,V_daily,days,hours,minutes)
B_daily, B_daily_ext,s= get_BM_price(X_daily, B_daily, QV_hat_price,QV_real_price,time_days)

f1=row(r,s)
show(f1)


# In[16]:


def augment_noises(B,W):
    t_scaled=[k/len(W) for k in range(len(W))]
    noises=np.array([[t_scaled[k],B[k],W[k]] for k in range(len(W))])
    return noises
def build_sig_df_sv(order_model,noises,N):
    order_Signature=order_model+1
    nbr_components=noises.shape[1]
    noises_torchified=torch.from_numpy(noises).unsqueeze(0)
    sig=signatory.signature(noises_torchified,order_Signature,stream=True,basepoint=True,scalar_term=True)
    keys=esig.sigkeys(nbr_components,order_Signature).strip().split(" ")
    sig_df=pd.DataFrame(sig.squeeze(0).numpy(), columns=keys)
    return sig_df,keys,sig    


# In[17]:


noises_x=augment_noises(B_daily_ext,W_daily_ext) #x is the extrapolated one
noises_y=augment_noises(B_daily,W_daily)         #y the real one

order_model=2
Sig_data_frame_x, keys, Sig_x=build_sig_df_sv(order_model,noises_x,days)
Sig_data_frame_y, keys, Sig_y=build_sig_df_sv(order_model,noises_y,days)


# In[18]:


keys_n=esig.sigkeys(noises_x.shape[1],order_model).strip().split(" ")
y=[[eval(key)] if isinstance(eval(key), int) else list(eval(key)) for key in keys_n]
first_step=e_tilde_part2_new(y)
new_tilde=from_tilde_to_strings_new(first_step)


# In[19]:


transformed_df_x, new_keys_B=get_tilde_df_debug(Sig_data_frame_x,new_tilde,keys_n,keys,3,rho,y)
transformed_df_y, new_keys_B=get_tilde_df_debug(Sig_data_frame_y,new_tilde,keys_n,keys,3,rho,y)


# In[20]:


transformed_df_x


# In[25]:



#Regression task for noises_x (t,B_t,W_t) with B_t, W_t extrapolated from the market
reg_sv_x=Lasso(alpha=0.00001).fit(transformed_df_x[new_keys_B], X_daily)
pred_sv_x=reg_sv_x.predict(transformed_df_x[new_keys_B])
#Regression task for noises_y (t,B_t,W_t) with B_t, W_t used in the simulation of the Stoch-Vol
reg_sv_y=Lasso(alpha=0.00001).fit(transformed_df_y[new_keys_B], X_daily)
pred_sv_y=reg_sv_y.predict(transformed_df_y[new_keys_B])


q = figure(width=500, height=350,title='Extrapolated BM')
q1 = q.line(time_days,pred_sv_x, legend_label='Sig-SDE', line_color='royalblue')
q2 = q.line(time_days,X_daily, legend_label='Stoch-Vol', line_color='tomato')

p = figure(width=500, height=350,title='Real BM')
p1 = p.line(time_days,pred_sv_y, legend_label='Sig-SDE', line_color='royalblue')
p2 = p.line(time_days,X_daily, legend_label='Stoch-Vol', line_color='tomato')
p3 = p.line()


g = row(p,q)

show(g)

print('MSE of the traj with Extrapolated BMs ',mean_squared_error(pred_sv_x,X_daily))
print('MSE of the traj with Real BMs ',mean_squared_error(pred_sv_y,X_daily))


# ## <font color=”darkblue”> Let's now test this on new realizations </font>

# In[28]:


from tqdm.auto import tqdm


# In[38]:


mse_mc_extrapolated_bms=[]
mse_mc_true_bms=[]

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
theta=0.15
kappa=0.5
rho=-0.5


# In[34]:


nbr_test_trials=10
test_window=int(time_days.shape[0]/3)


# In[35]:


for j in tqdm(range(nbr_test_trials)):
    
    S_new,V_new,B_new,W_new,time,noises_new=Simulate_Heston(N,initial_price,mu,alpha,kappa,theta,initial_vol,rho,t_final,t_0)

    #p1 = figure(width=500, height=350, title='Simulation of the SABR type process')
    #p1.line(time, S_new, color="royalblue", legend_label='Price')

    #p2 = figure(width=500, height=350, title='Simulation of the underlying volatility process')
    #p2.line(time, V_new, color="tomato", legend_label='Volatility')


    # put all the plots in an HBox
    #p = row(p1,p2)

    #show(p)

    QV_hat_vol_new, QV_real_vol_new, V_daily_new, W_daily_new, time_days, p = get_QV_vol_Heston(V_new,W_new,days,hours,minutes,alpha)
    W_daily_new, W_daily_ext_new,q = get_BM_vol(V_daily_new, W_daily_new, QV_hat_vol_new ,QV_real_vol_new,time_days)


    QV_hat_price_new, QV_real_price_new, X_daily_new, B_daily_new, time_days, r = get_QV_price_Heston(S_new,B_new,V_daily_new,days,hours,minutes)
    B_daily_new, B_daily_ext_new,s= get_BM_price(X_daily_new, B_daily_new, QV_hat_price_new,QV_real_price_new,time_days)

    #f=row(p,q)
    #show(f)

    noises_x_new=augment_noises(B_daily_ext_new,W_daily_ext_new) #x is the extrapolated one
    noises_y_new=augment_noises(B_daily_new,W_daily_new)         #y the real one

    Sig_data_frame_x_new, keys, Sig_x_new=build_sig_df_sv(order_model,noises_x_new,days)
    Sig_data_frame_y_new, keys, Sig_y_new=build_sig_df_sv(order_model,noises_y_new,days)

    keys_n=esig.sigkeys(noises_x.shape[1],order_model).strip().split(" ")
    y=[[eval(key)] if isinstance(eval(key), int) else list(eval(key)) for key in keys_n]
    first_step=e_tilde_part2_new(y)
    new_tilde=from_tilde_to_strings_new(first_step)


    transformed_df_x_new, new_keys_B=get_tilde_df_debug(Sig_data_frame_x_new,new_tilde,keys_n,keys,3,rho,y)
    transformed_df_y_new, new_keys_B=get_tilde_df_debug(Sig_data_frame_y_new,new_tilde,keys_n,keys,3,rho,y)

    pred_x_new=reg_sv_x.predict(transformed_df_x_new[new_keys_B])
    pred_y_new=reg_sv_y.predict(transformed_df_y_new[new_keys_B])
    
    mse_mc_extrapolated_bms.append(mean_squared_error(pred_x_new[:test_window],X_daily_new[:test_window]))
    mse_mc_true_bms.append(mean_squared_error(pred_y_new[:test_window],X_daily_new[:test_window]))


# In[36]:


mse_mc_extrapolated_bms


# In[37]:


mse_mc_true_bms


# In[42]:


from bokeh.models import Legend


# In[46]:


p = figure(width=800, height=490,title='Out of sample performance of the calibration to the price path')
r = p.line(time_days[:test_window],pred_x_new[:test_window], line_color='royalblue')
q = p.line(time_days[:test_window],X_daily_new[:test_window], line_color='tomato')
legend = Legend(items=[
    ("Sig-SDE",   [r]),
    ("Heston",[q])
], location=(-120,360))

p.add_layout(legend, 'right')
show(p)


# As follows we load the data computed from the cluster for the MSE out of sample on 1000 trajectories

# In[71]:


from bokeh.models import Legend
import os
os.getcwd()
os.chdir(r'C:\Users\Guido Gazzani\ucloud\Shared\Ripristinati\cluster_HE1')


# In[72]:


traj_outsample_HE=np.load('traj_outsample_HE.npy')
learned_traj_HE=np.load('learned_traj_HE.npy')
mse_true_bms_insample_HE=np.load('mse_true_bms_insample_HE.npy')
mse_true_bms_outsample_HE=np.load('mse_true_bms_outsample_HE.npy')
mse_ex_bms_insample_HE=np.load('mse_ex_bms_insample_HE.npy')
mse_ex_bms_outsample_HE=np.load('mse_ex_bms_outsample_HE.npy')


# In[73]:


mse_true_bms_insample_HE


# In[74]:


mse_ex_bms_insample_HE


# In[75]:


np.mean(mse_ex_bms_outsample_HE)


# In[76]:


np.mean(mse_true_bms_outsample_HE)


# In[77]:


sel=np.random.randint(0,10)
traj_1=traj_outsample_HE[sel]
learned_traj_HE_1=learned_traj_HE[sel]

p = figure(width=800, height=490,title='Out of sample performance of the calibration to the price path')
r = p.line(time_days[:int(time_days.shape[0]/2)],traj_1[:int(time_days.shape[0]/2)], line_color='royalblue')
r1 = p.line(time_days[:int(time_days.shape[0]/2)],traj_1[:int(time_days.shape[0]/2)], line_color='royalblue')
q = p.line(time_days[:int(time_days.shape[0]/2)],learned_traj_HE_1[:int(time_days.shape[0]/2)], line_color='tomato')
legend = Legend(items=[
    ("Sig-SDE",   [r]),
    ("Heston",[q])
], location=(-600,360))

p.add_layout(legend, 'right')
show(p)






# In[99]:


os.getcwd()
os.chdir(r'C:\Users\Guido Gazzani\ucloud\Shared\Ripristinati\cluster_sabr1')
traj_outsample_sabr=np.load('traj_outsample_sabr.npy')
learned_traj_sabr=np.load('learned_traj_sabr.npy')


# In[105]:


p = figure(width=800, height=490,title='Out of sample trajectories from the path calibration')
for k in range(7,8):
    traj_1=traj_outsample_HE[k]
    learned_traj_HE_1=learned_traj_HE[k]
    r = p.line(time_days[:int(time_days.shape[0]/2)],traj_1[:int(time_days.shape[0]/2)], line_color='royalblue')
    r1= p.scatter(time_days[:int(time_days.shape[0]/2)],traj_1[:int(time_days.shape[0]/2)], color='royalblue',size=3)
    q = p.line(time_days[:int(time_days.shape[0]/2)],learned_traj_HE_1[:int(time_days.shape[0]/2)], line_color='tomato')
    
#    traj_2=traj_outsample_sabr[k]
#    learned_traj_sabr_2=learned_traj_sabr[k]
#    r = p.line(time_days[:int(time_days.shape[0]/2)],traj_2[:int(time_days.shape[0]/2)], line_color='royalblue')
#    r2= p.scatter(time_days[:int(time_days.shape[0]/2)],traj_2[:int(time_days.shape[0]/2)], color='royalblue',size=3)
#    s = p.line(time_days[:int(time_days.shape[0]/2)],learned_traj_sabr_2[:int(time_days.shape[0]/2)], line_color='mediumvioletred')
    
legend = Legend(items=[
    ("Sig-SDE",   [r]),
    ("Heston",[q]),
  #("SABR",[s])
], location=(-600,360))

p.add_layout(legend, 'right')
show(p)    

