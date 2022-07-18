# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 21:59:32 2022

@author: Guido Gazzani
"""

# # Joint Calibration time-series data and option prices 

# In[105]:


import signatory
import torch
#Call packages
import itertools as itt  
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import esig
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from functools import partial
from bokeh.io import show, output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure, show
from bokeh.layouts import column
import pandas as pd
from bokeh.embed import file_html
import chart_studio.plotly as py
from scipy.integrate import quad
from bokeh.models import Legend
import cmath
from scipy.optimize import bisect, least_squares, minimize_scalar
from scipy.stats import norm
import os
from py_vollib.ref_python.black_scholes_merton.implied_volatility import implied_volatility
from py_vollib.ref_python.black_scholes_merton.implied_volatility import black_scholes_merton



# ###  Load some of the functions from the path calibration </font>

# In[107]:


import path_calibration as pc
import stochastic_processes as sp


#  Regression on the price 



order_Signature=2
comp_of_path=3

x, y=pc.create_list_words_SV(order_Signature,comp_of_path)
print(y)
print(x)


# In[109]:


print('Example of transformation of the word [1,2,1] :',pc.tilde_transformation([1,2])) #All possible suffixes!
print('Example of transformation of the word [1,2,2] :',pc.tilde_transformation([1,1]))
print('Example of transformation of the word [1,2,3] :',pc.tilde_transformation([1,2,3]))


# In[110]:


tilde = pc.e_tilde_part2_new(y) #Memo: run it only once otherwise it iterates the transformation as input constraints are still satisfied
#print(tilde)
new_tilde=pc.from_tilde_to_strings_new(tilde)
#print(new_tilde)


# # <font color=”darkblue”> As follows some auxiliar function </font>

# In[111]:


def duplicate(testList, n):
    x=[list(testList) for _ in range(n)]
    flat_list = []
    for sublist in x:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def Heston_P_Value(hestonParams,r,T,s0,K,typ):
    kappa, theta, sigma, rho, v0 = hestonParams
    return 0.5+(1./np.pi)*quad(lambda xi: Int_Function_1(xi,kappa,theta, sigma,rho,v0,r,T,s0,K,typ),0.,500.)[0]

def Int_Function_1(xi,kappa,theta,sigma,rho,v0,r,T,s0,K,typ):
    return (cmath.e**(-1j*xi*np.log(K))*Int_Function_2(xi,kappa,theta,sigma,rho,v0,r,T,s0,typ)/(1j*xi)).real

def Int_Function_2(xi,kappa,theta,sigma,rho,v0,r,T,s0,typ):
    if typ == 1:
        w = 1.
        b = kappa - rho*sigma
    else:
        w = -1.
        b = kappa
    ixi = 1j*xi
    d = cmath.sqrt((rho*sigma*ixi-b)*(rho*sigma*ixi-b) - sigma*sigma*(w*ixi-xi*xi))
    g = (b-rho*sigma*ixi-d) / (b-rho*sigma*ixi+d)
    ee = cmath.e**(-d*T)
    C = r*ixi*T + kappa*theta/(sigma*sigma)*((b-rho*sigma*ixi-d)*T - 2.*cmath.log((1.0-g*ee)/(1.-g)))
    D = ((b-rho*sigma*ixi-d)/(sigma*sigma))*(1.-ee)/(1.-g*ee)
    return cmath.e**(C + D*v0 + ixi*np.log(s0))

def phi(x): ## Gaussian density
    return np.exp(-x*x/2.)/np.sqrt(2*np.pi)

#### Black Sholes Vega
def BlackScholesVegaCore(DF,F,X,T,v):   #S=F*DF
    vsqrt=v*np.sqrt(T)
    d1 = (np.log(F/X)+(vsqrt*vsqrt/2.))/vsqrt
    return F*phi(d1)*np.sqrt(T)/DF

#### Black Sholes Function
def BlackScholesCore(CallPutFlag,DF,F,X,T,v):
    ## DF: discount factor
    ## F: Forward
    ## X: strike
    vsqrt=v*np.sqrt(T)
    d1 = (np.log(F/X)+(vsqrt*vsqrt/2.))/vsqrt
    d2 = d1-vsqrt
    if CallPutFlag:
        return DF*(F*norm.cdf(d1)-X*norm.cdf(d2))
    else:
        return DF*(X*norm.cdf(-d2)-F*norm.cdf(-d1))
    
##  Black-Scholes Pricing Function
def BlackScholes(CallPutFlag,S,X,T,r,d,v):
    ## r, d: continuous interest rate and dividend
    return BlackScholesCore(CallPutFlag,np.exp(-r*T),np.exp((r-d)*T)*S,X,T,v)

def heston_EuropeanCall(hestonParams,r,T,s0,K):
    a = s0*Heston_P_Value(hestonParams,r,T,s0,K,1)
    b = K*np.exp(-r*T)*Heston_P_Value(hestonParams,r,T,s0,K,2)
    return a-b

def heston_Vanilla(hestonParams,r,T,s0,K,flag):
    a_call = s0*Heston_P_Value(hestonParams,r,T,s0,K,1)
    b_call = K*np.exp(-r*T)*Heston_P_Value(hestonParams,r,T,s0,K,2)
    a_put = s0*(1-Heston_P_Value(hestonParams,r,T,s0,K,1))
    b_put = K*np.exp(-r*T)*(1-Heston_P_Value(hestonParams,r,T,s0,K,2))
    if flag=='call':
        return a_call-b_call
    if flag=='put':
        return b_put-a_put
    else:
        return print('You have chosen a flag which is not a Vanilla Option')
    

def heston_Impliedvol(hestonParams,r,T,s0,K):
    myPrice = heston_EuropeanCall(hestonParams,r,T,s0,K)
    ## Bisection algorithm when the Lee-Li algorithm breaks down
    def smileMin(vol, *args):
        K, s0, T, r, price = args
        return price - BlackScholes(True, s0, K, T, r, 0., vol)
    vMin = 0.000001
    vMax = 10.
    return bisect(smileMin, vMin, vMax, args=(K, s0, T, r, myPrice), rtol=1e-15, full_output=False, disp=True)

def phi(x): ## Gaussian density
    return np.exp(-x*x/2.)/np.sqrt(2*np.pi)


#### Black Sholes Vega
def BlackScholesVegaCore(DF,F,X,T,v):   #S=F*DF
    vsqrt=v*np.sqrt(T)
    d1 = (np.log(F/X)+(vsqrt*vsqrt/2.))/vsqrt
    return F*phi(d1)*np.sqrt(T)/DF


# ### <font color=”darkblue”> Load maturities, set the strikes (varying per maturity here) and chose parameters of Heston </font>

# In[113]:


maturities=np.load('maturities_SPX_Bloomberg.npy')
moneynesses=[0.08,0.1,0.3,.3,0.3,.3,.3]
strikes=[]
nbr_strikes=9
S0=1
for moneyness in moneynesses:
    strikes.append(np.linspace(np.exp(-moneyness)*S0, np.exp(moneyness)*S0, nbr_strikes))
strikes=np.array(strikes)
set_params2={'alpha':0.55,'kappa':0,'theta':0,'rho':-0.5,'v0':0.12}


# In[114]:


def get_iv_Heston_by_params(set_params,strikes,maturities,S0,nbr_strikes):
    'Function that generates the IV surface of Heston for the given parameters, maturities and strikes'
    r=0.
    index_selected_mat=range(len(maturities))

    hestonParams = set_params['kappa'], set_params['theta'], set_params['alpha'], set_params['rho'], set_params['v0'] 
    
    Heston_prices_calib_call=[]
    iv_call=[]
    for j in index_selected_mat:
        for strike in strikes[j]:
            he_p_call=heston_Vanilla(hestonParams,r,maturities[j],S0,strike,'call')
            Heston_prices_calib_call.append(he_p_call)
            iv_call.append(implied_volatility(he_p_call, S0, strike, maturities[j], 0, 0, 'c'))

    Heston_prices_calib_put=[]
    iv_put=[]
    for j in index_selected_mat:
        for strike in strikes[j]:
            he_p_put=heston_Vanilla(hestonParams,r,maturities[j],S0,strike,'put')
            Heston_prices_calib_put.append(he_p_put)
            iv_put.append(implied_volatility(he_p_put, S0, strike, maturities[j], 0, 0, 'p'))
            
    element_to_substitute=0

    Heston_prices_calib=[]

    for j in index_selected_mat:
        for k in range(len(strikes[j])):
            if (k<len(strikes[0])-element_to_substitute):
                Heston_prices_calib.append(heston_Vanilla(hestonParams,r,maturities[j],S0,strikes[j][k],'call'))
            else:
                Heston_prices_calib.append(heston_Vanilla(hestonParams,r,maturities[j],S0,strikes[j][k],'put'))
                
    fig1 = plt.figure(figsize=(12, 8))
    for j in range(len(maturities)):
        #plt.subplot(2, 2, j+1)
        plt.plot(strikes[j], iv_call[j*nbr_strikes:(j+1)*nbr_strikes],label='Maturity T={}'.format(round(maturities[j],4)),marker='o')
        plt.legend()
    plt.title('Generated Smiles')
    plt.show()
    return np.array(Heston_prices_calib), np.array(iv_call)


# In[115]:


prices, iv= get_iv_Heston_by_params(set_params2,strikes,maturities,S0,nbr_strikes)


# In[220]:


set_params={'alpha':0.55,'kappa':0.8,'theta':0.1,'rho':-0.5,'v0':0.12}
#set_params={'alpha':0.4,'kappa':0.4,'theta':0.1,'rho':-0.5,'v0':0.12}


# In[221]:


hours=8
minutes=60*10
days=721
N, initial_price = days*hours*minutes, S0
mu=0.001 #drift
t_0=0
t_final=1
kappa,theta,alpha=set_params['kappa'], set_params['theta'], set_params['alpha']
rho,initial_vol=set_params['rho'], set_params['v0'] 

X,V,B,W,time,noises=sp.Simulate_Heston(N,S0,mu,alpha,kappa,theta,initial_vol,rho,t_final,t_0)


# In[222]:


# Of the vol
QV_hat_vol, QV_real_vol, V_daily, W_daily, time_days, p = sp.get_QV_vol_Heston(V,W,days,hours,minutes,alpha)
W_daily, W_daily_ext,q = sp.get_BM_vol(V_daily, W_daily, QV_hat_vol ,QV_real_vol,time_days)

f=row(p,q)
show(f)

#Of the price

QV_hat_price, QV_real_price, X_daily, B_daily, time_days, r = sp.get_QV_price_Heston(X,B,V_daily,days,hours,minutes)
B_daily, B_daily_ext,s= sp.get_BM_price(X_daily, B_daily, QV_hat_price,QV_real_price,time_days)

f1=row(r,s)
show(f1)


# In[223]:


noises_x=pc.augment_noises(B_daily_ext,W_daily_ext) #x is the extrapolated one
noises_y=pc.augment_noises(B_daily,W_daily)         #y the real one

Sig_data_frame_x, keys, Sig_x=pc.build_Sig_data_frame_sv(order_Signature,noises_x,days,comp_of_path,time_days)
Sig_data_frame_y, keys, Sig_y=pc.build_Sig_data_frame_sv(order_Signature,noises_y,days,comp_of_path,time_days)


transformed_df_x, new_keys_B=pc.transform_data_frame_sv(Sig_data_frame_x,new_tilde,keys,comp_of_path,rho,order_Signature)
transformed_df_y, new_keys_B=pc.transform_data_frame_sv(Sig_data_frame_y,new_tilde,keys,comp_of_path,rho,order_Signature)


# In[224]:


#os.chdir(r'C:\Users\Guido Gazzani\ucloud\Shared\SigSDEs_ucloud\Code_joint_calibration\replicate_correct_plots\may')
#np.save('set_calibrated_params.npy',set_calibrated_params_arr)
#np.save('transformed_df_x_B.npy',transformed_df_x[new_keys_B])
#np.save('X_daily.npy',X_daily)
#np.save('iv.npy',iv)

#os.chdir(r'C:\Users\Guido Gazzani\ucloud\Shared\SigSDEs_ucloud\Code_joint_calibration')


# In[225]:


def dimension_coef(order_Signature,d,D):
    return int(((d+1)**(order_Signature)-1)*D/d)
d,D=2,1
dim=dimension_coef(order_Signature+1,d,D)
print(dim)


# In[226]:


l_initial=np.random.uniform(-0.2,0.2,dim)


# Fix a maturity $T$>0

# In[227]:


J=2
T=maturities[J]
trunc=int(T*365.25)
print(trunc)


# In[228]:


Sig_data_frame_x, keys, Sig_x=pc.build_Sig_data_frame_sv(order_Signature,noises_x,trunc,comp_of_path,time_days)


# In[229]:


transformed_df_x, new_keys_B=pc.transform_data_frame_sv(Sig_data_frame_x,new_tilde,keys,comp_of_path,rho,order_Signature)


# In[230]:


transformed_df_x[new_keys_B]


# In[231]:


Sig, price_path=transformed_df_x[new_keys_B], X_daily[:trunc]


# #### <font color=”darkblue”> Define the first (for $T_{1}$) objective functional for the path, $L_{\text{path}}$ 

# In[232]:


def objective_path_lq(l):
    diff = np.dot(Sig,l)+price_path[0] - price_path
    return diff


# In[233]:


iv_calibrated=[]


# ## <font color=”darkblue”> Calibration to option </font>

# Compute the vega weights

# In[234]:


iv=np.array(np.split(iv,len(maturities)))


# In[235]:


def get_vegas(maturities, strikes, initial_price, iv_market, flag_truncation):
    vega=[]
    for i in range(len(maturities)):
        for j in range(len(strikes[i])):
            if flag_truncation==True:
                vega.append(min(1/(BlackScholesVegaCore(1,initial_price,strikes[i][j],maturities[i],iv_market[i,j])),1))
            else:
                vega.append(1/(BlackScholesVegaCore(1,initial_price,strikes[i][j],maturities[i],iv_market[i,j])))
    vega=np.array(vega)

    vega_by_mat=np.array(np.split(vega,len(maturities)))
    sums_each_strike=np.sum(vega_by_mat, axis=1)
    normalized_vega=np.array([vega_by_mat[j]/sums_each_strike[j] for j in range(len(maturities))])
    flat_normal_weights=normalized_vega.flatten()
    return flat_normal_weights, normalized_vega


flag_truncation=False
flat_normal_weights, norm_vegas=get_vegas(maturities, strikes, S0, iv, flag_truncation)


# In[236]:


print('We are interested in the maturity\n T={}'.format(maturities[J]))
print('with strike prices given by\n K={}'.format(strikes[J]))


# In[16]:


os.chdir(r'C:\Users\Guido Gazzani\ucloud\Shared\SigSDEs_ucloud\Code_real_data\Cluster_files')


# In[17]:


arr_dfs_by_mat=np.load('arr_dfs_by_mat((7, 1000000, 13)).npy')
arr_dfs_by_mat.shape



sel_maturities=[maturities[J],maturities[J+2]]
index_sel_maturities=[J,J+2]

Vega_W=np.array([np.split(flat_normal_weights,len(maturities))[idx] for idx in index_sel_maturities]).flatten()
Premium1=np.array([np.split(prices,len(maturities))[idx] for idx in index_sel_maturities]).flatten()
arr_dfs_to_optimize=[]
adjusted_scalar_products=[]



initial_price=1


# In[11]:


def time_varying_model(l,idx_maturity,arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,initial_price):
    if len([idx_maturity])==1:
        tensor_sigsde_at_mat_aux=np.tensordot(arr_dfs_by_mat,l,axes=1)+initial_price
        tensor_sigsde_at_mat=tensor_sigsde_at_mat_aux[idx_maturity,:]
    else:
        tensor_prod_last_maturity=np.tensordot(arr_dfs_to_optimize,l,axes=1)
        tensor_prod_last_maturity=np.expand_dims(tensor_prod_last_maturity,1)
        tensor_sigsde_at_mat_aux=np.concatenate((adjusted_scalar_products,tensor_prod_last_maturity),axis=1)
        tensor_sigsde_at_mat=initial_price+np.sum(tensor_sigsde_at_mat_aux,axis=1)
    return tensor_sigsde_at_mat


def get_mc_sel_mat_tensor_TV(tensor_sigsde_at_mat,index_sel_maturities,strikes):

        '''
        Input: tensor_sigsde_mat (np.array): model at selected maturities

        If len(index_sel_maturities)>1 then dim(tensor_sigsde_at_mat)=(nbr_MC_sim, len(index_sel_maturities))
        If len(index_sel_maturities)=1 then dim(tensor_sigsde_at_mat)=(nbr_MC_sim,)

        index_sel_maturities (list): list of integers corresponding to the selected maturities
        strikes (np.array 2D): array of arrays where each of the sub-array stores the strikes

        Output: Monte Carlo prices of the model for the selected maturities and strikes
        '''
        pay=[]
        if len(index_sel_maturities)==1:

            for K in strikes[index_sel_maturities[0]]:
                matrix=np.maximum(0,tensor_sigsde_at_mat-K)
                pay.append(np.mean(matrix))
            mc_payoff_arr=np.array(pay)
        else:
            for K in strikes[0]:
                matrix_big=[]
                for j in index_sel_maturities:
                    payff=np.maximum(0, tensor_sigsde_at_mat[:,j] - K)
                    matrix_big.append(payff)
                matrix=np.array(matrix_big)
                pay.append(np.mean(matrix,axis=1))
            mc_payoff_arr=np.array(pay).transpose().flatten()
        return mc_payoff_arr


def objective_options(l):
    tensor_sigsde=time_varying_model(l,index_sel_maturities[0],arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,S0)
    tensor_sigsde2=time_varying_model(l,index_sel_maturities[1],arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,S0)
    mc_payoff_arr=get_mc_sel_mat_tensor_TV(tensor_sigsde,[index_sel_maturities[0]],strikes)  
    mc_payoff_arr2=get_mc_sel_mat_tensor_TV(tensor_sigsde2,[index_sel_maturities[1]],strikes)  
    mc_payoff_arr_conc=np.array([mc_payoff_arr,mc_payoff_arr2]).flatten()
    print('One optimization step completed!')
    return np.sqrt(Vega_W)*(mc_payoff_arr_conc-Premium1)


# In[ ]:



lambd=0.95
# The following is the so called L_joint
def objective_mix_lq(l):
    conc_=np.concatenate([np.sqrt(1-lambd)*objective_path_lq(l),np.sqrt(lambd)*objective_options2(l)],axis=0)
    return conc_
    
    
n=order_Signature
l_initial=np.random.uniform(-1,1,int(((d+1)**(n+1)-1)*D/d))


# In[242]:


for t in tqdm(range(1)): 
    l_initial=np.random.uniform(-0.8,0.8,dim)
    res1 = least_squares(objective_options, l_initial,loss='linear')


# In[12]:


tensor_sigsde_at_mat=time_varying_model(res1['x'],index_sel_maturities[0],arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,S0)  
tensor_sigsde_at_mat2=time_varying_model(res1['x'],index_sel_maturities[1],arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,S0)  
calibrated_prices=get_mc_sel_mat_tensor_TV(tensor_sigsde_at_mat,[index_sel_maturities[0]],strikes)
calibrated_prices2=get_mc_sel_mat_tensor_TV(tensor_sigsde_at_mat2,[index_sel_maturities[1]],strikes)
calibrated_prices_both=np.array([calibrated_prices,calibrated_prices2])
#tensor_sigsde_at_mat=np.tensordot(arr_dfs_by_mat,res1['x'],axes=1)+initial_price
#calibrated_prices=get_mc_sel_mat_tensor(res1['x'])

def implied_vol_minimize( price, S0, K, T, r, payoff="call", disp=True ):
    """ Returns Implied volatility by minimization"""
    
    n = 2     # must be even
    def obj_fun(vol):
        return ( BlackScholes(True, S0, K, T, r, 0., vol) - price)**n
        
    res = minimize_scalar( obj_fun, bounds=(1e-15, 8), method='bounded')
    if res.success == True:
        return res.x       
    if disp == True:
        print("Strike", K)
    return -1

def get_iv_from_calib_onemat(calibrated_prices,strikes,maturity,idx_mat,S0):
    iv_calib_mc=[]
    
    for k in range(len(strikes[idx_mat])):
         iv_calib_mc.append(implied_vol_minimize(calibrated_prices[k], S0, strikes[idx_mat][k], maturity, 0, payoff="call", disp=True))
    return np.array(iv_calib_mc)
    

def get_iv_from_calib(calibrated_prices,strikes,maturities,S0):
    sig_prices_mc_arr=[]
    iv_calib_mc=[]
    
    sig_prices_mc_arr=np.array(np.split(calibrated_prices,len(maturities)))
        
    for j in range(len(maturities)):
        for k in range(len(strikes[j])):
                iv_calib_mc.append(implied_vol_minimize(sig_prices_mc_arr[j,k], S0, strikes[j][k], maturities[j], 0, payoff="call", disp=True))

    iv_calib_arr_mc=np.array([np.array(iv_calib_mc[k*len(strikes[0]):(k+1)*len(strikes[0])]) for k in range(len(maturities))])
    return iv_calib_arr_mc, sig_prices_mc_arr


# In[244]:



plt.plot(strikes[J],Premium1[:9],marker='*',color='b',label='Market')
plt.plot(strikes[J],calibrated_prices,marker='o',color='r',label='SigSDE')
plt.legend()
plt.xlabel('Strike')
#plt.ylabel('Price')
plt.title('Option prices')
plt.show()

plt.plot(strikes[J],np.abs(Premium1[:9]-calibrated_prices),marker='o',color='b')
plt.title('Absolute Error on option prices')
plt.xlabel('Strike')
plt.show()


plt.plot(strikes[J],Premium1[9:18],marker='*',color='b',label='Market')
plt.plot(strikes[J],calibrated_prices2,marker='o',color='r',label='SigSDE')
plt.legend()
plt.xlabel('Strike')
#plt.ylabel('Price')
plt.title('Option prices')
plt.show()

plt.plot(strikes[J],np.abs(Premium1[9:18]-calibrated_prices2),marker='o',color='b')
plt.title('Absolute Error on option prices')
plt.xlabel('Strike')
plt.show()


# In[245]:


iv_mc=[]
iv_market=np.array(np.split(iv,len(maturities)))

iv_calib_arr_mc_slice1=get_iv_from_calib_onemat(calibrated_prices,strikes,maturities[index_sel_maturities[0]],index_sel_maturities[0],S0)
iv_calib_arr_mc_slice2=get_iv_from_calib_onemat(calibrated_prices2,strikes,maturities[index_sel_maturities[1]],index_sel_maturities[1],S0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle('Maturity T={}'.format(round(maturities[index_sel_maturities[0]],4)))
ax1.plot(strikes[index_sel_maturities[0]], iv_calib_arr_mc_slice1,marker='o',color='r',alpha=0.4,label=f'SigSDE IV (n={n})')
ax1.plot(strikes[index_sel_maturities[0]], iv[index_sel_maturities[0]],marker='*',alpha=0.4,color='b',label='Market IV')
ax1.set_xlabel('Strikes')
ax1.set_title('Implied volatilities - Heston')
ax2.scatter(strikes[index_sel_maturities[0]],np.abs(iv_calib_arr_mc_slice1-iv[index_sel_maturities[0]])*10000,color='slateblue') 
ax2.set_xlabel('Strikes')
ax2.set_ylabel('Bps')
ax2.set_title('Absolute Error in Basepoints')
ax1.legend()
#plt.savefig('Fit_MC_TV2(T={}'.format(round(maturities[j],4))+', with n=2).png',dpi=500)
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle('Maturity T={}'.format(round(maturities[index_sel_maturities[1]],4)))
ax1.plot(strikes[index_sel_maturities[1]], iv_calib_arr_mc_slice2,marker='o',color='r',alpha=0.4,label=f'SigSDE IV (n={n})')
ax1.plot(strikes[index_sel_maturities[1]], iv[index_sel_maturities[1]],marker='*',alpha=0.4,color='b',label='Market IV')
ax1.set_xlabel('Strikes')
ax1.set_title('Implied volatilities - Heston')
ax2.scatter(strikes[index_sel_maturities[1]],np.abs(iv_calib_arr_mc_slice2-iv[index_sel_maturities[1]])*10000,color='slateblue') 
ax2.set_xlabel('Strikes')
ax2.set_ylabel('Bps')
ax2.set_title('Absolute Error in Basepoints')
ax1.legend()
#plt.savefig('Fit_MC_TV2(T={}'.format(round(maturities[j],4))+', with n=2).png',dpi=500)
plt.show()


# In[246]:


for t in tqdm(range(1)): 
    l_initial=np.random.uniform(-1,1,dim)
    res2= least_squares(objective_path_lq, l_initial,loss='linear')
    


# In[247]:


Sig_data_frame_xx, keysx, Sig_xx=pc.build_Sig_data_frame_sv(order_Signature,noises_x,days,comp_of_path,time_days)
transformed_df_xx, new_keys_Bx=pc.transform_data_frame_sv(Sig_data_frame_xx,new_tilde,keys,comp_of_path,rho,order_Signature)

ww=trunc
q = figure(width=500, height=350,title='Path calibration')
q1 = q.line(time_days[:trunc+ww],(np.dot(transformed_df_xx[new_keys_B],res2['x'])+S0)[:trunc+ww], legend_label='Sig-SDE', line_color='tomato')
q2 = q.line(time_days[:trunc+ww],X_daily[:trunc+ww], legend_label='Heston', line_color='royalblue')
q3= q.line([time_days[trunc],time_days[trunc]], [min((np.dot(transformed_df_xx[new_keys_B],res2['x'])+S0)[:trunc+ww]),max((np.dot(transformed_df_xx[new_keys_B],res2['x'])+S0)[:trunc+ww])], line_width=1,line_color='orange',line_dash='dashed')
show(q)
print('MSE:',np.sum(((np.dot(transformed_df_xx[new_keys_B],res2['x'])+S0)[:trunc+ww]-X_daily[:trunc+ww])**2))

q = figure(width=500, height=350,title='Path calibration with options parameters')
q1 = q.line(time_days[:trunc+ww],(np.dot(transformed_df_xx[new_keys_B],res1['x'])+S0)[:trunc+ww], legend_label='Sig-SDE', line_color='tomato')
q2 = q.line(time_days[:trunc+ww],X_daily[:trunc+ww], legend_label='Heston', line_color='royalblue')
q3= q.line([time_days[trunc],time_days[trunc]], [min((np.dot(transformed_df_xx[new_keys_B],res1['x'])+S0)[:trunc+ww]),max((np.dot(transformed_df_xx[new_keys_B],res1['x'])+S0)[:trunc+ww])], line_width=1,line_color='orange',line_dash='dashed')
show(q)
print('MSE:',np.sum(((np.dot(transformed_df_xx[new_keys_B],res1['x'])+S0)[:trunc+ww]-X_daily[:trunc+ww])**2))

q = figure(width=500, height=350,title='Path calibration with options parameters with a minus in front')
q1 = q.line(time_days[:trunc+ww],(-np.dot(transformed_df_xx[new_keys_B],res1['x'])+S0)[:trunc+ww], legend_label='Sig-SDE', line_color='tomato')
q2 = q.line(time_days[:trunc+ww],X_daily[:trunc+ww], legend_label='Heston', line_color='royalblue')
q3= q.line([time_days[trunc],time_days[trunc]], [min((-np.dot(transformed_df_xx[new_keys_B],res1['x'])+S0)[:trunc+ww]),max((-np.dot(transformed_df_xx[new_keys_B],res1['x'])+S0)[:trunc+ww])], line_width=1,line_color='orange',line_dash='dashed')
show(q)
print('MSE:',np.sum(((-np.dot(transformed_df_xx[new_keys_B],res1['x'])+S0)[:trunc+ww]-X_daily[:trunc+ww])**2))


# In[248]:


res2['x']


# In[249]:


res1['x']


# In[250]:


plt.scatter(new_keys_B,res1['x'],label='Options')
plt.scatter(new_keys_B,res2['x'],label='Path')
plt.axhline(y=0, color='r', linestyle='--')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# In[251]:


l_aux=res1['x']+np.random.normal(0,0.001,dim)
#l_initial=np.random.uniform(-0.4,0.4,dim)


# In[252]:


exp=300
def rho1(x,exp):
    return (1+x)**exp
def rho2(x,exp):
    return exp*(1+x)**(exp-1)
def rho3(x,exp):
    return (exp)*(exp-1)*(1+x)**(exp-2)

def rho_all(z):
    return np.array([rho1(z,exp),rho2(z,exp),rho3(z,exp)])


# In[253]:


abs_err1=100000*np.abs(iv[index_sel_maturities[0]]-iv_calib_arr_mc_slice1)
abs_err2=100000*np.abs(iv[index_sel_maturities[1]]-iv_calib_arr_mc_slice2)
abs_err=np.array([abs_err1,abs_err2]).flatten()


# In[254]:


def objective_options(l):
    tensor_sigsde=time_varying_model(l,index_sel_maturities[0],arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,S0)
    tensor_sigsde2=time_varying_model(l,index_sel_maturities[1],arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,S0)
    mc_payoff_arr=get_mc_sel_mat_tensor_TV(tensor_sigsde,[index_sel_maturities[0]],strikes)  
    mc_payoff_arr2=get_mc_sel_mat_tensor_TV(tensor_sigsde2,[index_sel_maturities[1]],strikes)  
    mc_payoff_arr_conc=np.array([mc_payoff_arr,mc_payoff_arr2]).flatten()
    print('One optimization step completed!')
    return np.sqrt(Vega_W+abs_err)*(mc_payoff_arr_conc-Premium1)

lambd=0.9
# The following is the so called L_joint
def objective_mix_lq(l):
    conc_=np.concatenate([np.sqrt(1-lambd)*objective_path_lq(l),np.sqrt(lambd)*objective_options(l)],axis=0)
    return conc_
    


# In[255]:


for t in tqdm(range(1)):
    sol_mix= least_squares(objective_mix_lq, l_initial,loss=rho_all)


# In[256]:


for t in tqdm(range(1)):
    sol_mix2= least_squares(objective_mix_lq, l_initial,loss='linear')


# In[257]:


plt.scatter(new_keys_B,res1['x'],label='Options')
plt.scatter(new_keys_B,res2['x'],label='Path')
plt.scatter(new_keys_B,sol_mix['x'],label='Joint')
plt.scatter(new_keys_B,sol_mix2['x'],label='Joint2')
plt.axhline(y=0, color='r', linestyle='--')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# In[258]:


sol_mix['x']-sol_mix2['x']


# In[259]:


print('Loss of the mixed problem with the joint param:',np.sum((objective_mix_lq(sol_mix['x']))**2))
print('Loss of the mixed problem with the joint param2:',np.sum((objective_mix_lq(sol_mix2['x']))**2))
print('Loss of the mixed problem with path param:',np.sum((objective_mix_lq(res1['x']))**2))
print('Loss of the mixed problem with option param:',np.sum((objective_mix_lq(res2['x']))**2))


# In[260]:


tensor_sigsde_at_mat=time_varying_model(sol_mix['x'],index_sel_maturities[0],arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,S0)  
tensor_sigsde_at_mat2=time_varying_model(sol_mix['x'],index_sel_maturities[1],arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,S0)  
calibrated_prices=get_mc_sel_mat_tensor_TV(tensor_sigsde_at_mat,[index_sel_maturities[0]],strikes)
calibrated_prices2=get_mc_sel_mat_tensor_TV(tensor_sigsde_at_mat2,[index_sel_maturities[1]],strikes)
calibrated_prices_both=np.array([calibrated_prices,calibrated_prices2])

tensor_sigsde_at_mat=time_varying_model(sol_mix2['x'],index_sel_maturities[0],arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,S0)  
tensor_sigsde_at_mat2=time_varying_model(sol_mix2['x'],index_sel_maturities[1],arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,S0)  
calibrated_prices3=get_mc_sel_mat_tensor_TV(tensor_sigsde_at_mat,[index_sel_maturities[0]],strikes)
calibrated_prices4=get_mc_sel_mat_tensor_TV(tensor_sigsde_at_mat2,[index_sel_maturities[1]],strikes)
calibrated_prices_both2=np.array([calibrated_prices3,calibrated_prices4])


# In[261]:


iv_mc_MIX=[]
#iv_market=np.array(np.split(iv,len(maturities)))
j=index_sel_maturities[0]

iv_calib_arr_mc_slice1=get_iv_from_calib_onemat(calibrated_prices,strikes,maturities[index_sel_maturities[0]],index_sel_maturities[0],S0)
iv_calib_arr_mc_slice2=get_iv_from_calib_onemat(calibrated_prices2,strikes,maturities[index_sel_maturities[1]],index_sel_maturities[1],S0)
iv_calib_arr_mc_slice3=get_iv_from_calib_onemat(calibrated_prices3,strikes,maturities[index_sel_maturities[0]],index_sel_maturities[0],S0)
iv_calib_arr_mc_slice4=get_iv_from_calib_onemat(calibrated_prices4,strikes,maturities[index_sel_maturities[1]],index_sel_maturities[1],S0)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle('Maturity T={}'.format(round(maturities[j],4)))
ax1.plot(strikes[j], iv_calib_arr_mc_slice1,marker='o',color='r',alpha=0.4,label=f'SigSDE IV (n={n})')
ax1.plot(strikes[j], iv_calib_arr_mc_slice3,marker='o',color='g',alpha=0.4,label=f'SigSDE IV (n={n})')
ax1.plot(strikes[j], iv[j],marker='*',alpha=0.4,color='b',label='Market IV')
ax1.set_xlabel('Strikes')
ax1.set_title('Implied volatilities - Heston')
ax2.scatter(strikes[j],np.abs( iv_calib_arr_mc_slice1-iv[j])*10000,color='slateblue') 
ax2.scatter(strikes[j],np.abs( iv_calib_arr_mc_slice3-iv[j])*10000,color='forestgreen') 
ax2.set_xlabel('Strikes')
ax2.set_ylabel('Bps')
ax2.set_title('Absolute Error in Basepoints')
ax1.legend()
#plt.savefig('Fit_HE_joint(T={}'.format(round(maturities[j],4))+', with n=2).png',dpi=250)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle('Maturity T={}'.format(round(maturities[index_sel_maturities[1]],4)))
ax1.plot(strikes[index_sel_maturities[1]], iv_calib_arr_mc_slice2,marker='o',color='r',alpha=0.4,label=f'SigSDE IV (n={n})')
ax1.plot(strikes[index_sel_maturities[1]], iv_calib_arr_mc_slice4,marker='o',color='g',alpha=0.4,label=f'SigSDE IV (n={n})')
ax1.plot(strikes[index_sel_maturities[1]], iv[index_sel_maturities[1]],marker='*',alpha=0.4,color='b',label='Market IV')
ax1.set_xlabel('Strikes')
ax1.set_title('Implied volatilities - Heston')
ax2.scatter(strikes[j],np.abs( iv_calib_arr_mc_slice2-iv[index_sel_maturities[1]])*10000,color='slateblue') 
ax2.scatter(strikes[j],np.abs( iv_calib_arr_mc_slice4-iv[index_sel_maturities[1]])*10000,color='forestgreen') 
ax2.set_xlabel('Strikes')
ax2.set_ylabel('Bps')
ax2.set_title('Absolute Error in Basepoints')
ax1.legend()
#plt.savefig('Fit_HE_joint(T={}'.format(round(maturities[j],4))+', with n=2).png',dpi=250)
plt.show()


# In[262]:


noises_x=pc.augment_noises(B_daily_ext,W_daily_ext) #x is the extrapolated one

Sig_data_frame_x, keys, Sig_x=pc.build_Sig_data_frame_sv(order_Signature,noises_x,days,comp_of_path,time_days)

transformed_df_x, new_keys_B=pc.transform_data_frame_sv(Sig_data_frame_x,new_tilde,keys,comp_of_path,rho,order_Signature)


# In[263]:


year=365.25
ww=trunc#-1
grid_in_and_out_sample=np.array(range(trunc+ww))/year
grid_in_sample=np.array(range(trunc+1))/year
path_approximated_1=np.dot(transformed_df_x[new_keys_B],sol_mix['x'])+S0
path_approximated_2=np.dot(transformed_df_x[new_keys_B],sol_mix2['x'])+S0

q = figure(width=500, height=400,title='Joint calibration with weight {}'.format(lambd) + ' on the options')
q1 = q.line(grid_in_and_out_sample,path_approximated_1[:trunc+ww], legend_label='Sig-SDE', line_color='tomato')
q2 = q.circle(grid_in_and_out_sample,path_approximated_1[:trunc+ww], fill_color='tomato')

q3 = q.line(grid_in_and_out_sample,path_approximated_2[:trunc+ww], legend_label='Sig-SDE', line_color='tomato')
q4 = q.circle(grid_in_and_out_sample,path_approximated_2[:trunc+ww], fill_color='forestgreen')

q5 = q.line(grid_in_and_out_sample,X_daily[:trunc+ww], legend_label='Heston', line_color='royalblue')
q6= q.line([grid_in_sample[-1],grid_in_sample[-1]], [min(path_approximated_2[:trunc+ww]),max(path_approximated_2[:trunc+ww])], line_width=1,line_color='orange',line_dash='dashed')
q.legend.location = "top_left"
#show(q)


b = figure(width=500, height=400,title='IV at maturity T={}'.format(round(maturities[index_sel_maturities[0]],4)))

b1 = b.circle(strikes[j],iv_calib_arr_mc_slice1,legend_label='Sig IV',fill_alpha=0.2,size=7,color="red")
b2 = b.circle(strikes[j],iv_calib_arr_mc_slice3,legend_label='Sig IV',fill_alpha=0.2,size=7,color="forestgreen")
b3 = b.star(strikes[j],iv[index_sel_maturities[0]],legend_label='Heston IV',fill_alpha=0.2,size=7,color="navy")
b.xaxis.axis_label = 'Strike'
b.yaxis.axis_label = 'IV'
b.legend.location = "top_right"



k = figure(width=500, height=400,title='IV at maturity T={}'.format(round(maturities[index_sel_maturities[1]],4)))

k.circle(strikes[j],iv_calib_arr_mc_slice2,legend_label='Sig IV',fill_alpha=0.2,size=7,color="red")
k.circle(strikes[j],iv_calib_arr_mc_slice4,legend_label='Sig IV',fill_alpha=0.2,size=7,color="forestgreen")
k.star(strikes[j],iv[index_sel_maturities[1]],legend_label='Heston IV',fill_alpha=0.2,size=7,color="navy")
k.xaxis.axis_label = 'Strike'
k.yaxis.axis_label = 'IV'
k.legend.location = "top_right"
show(row(q,b,k))


# In[278]:


year=365.25
ww=trunc+int(0.8*trunc)

grid_in_and_out_sample=np.array(range(trunc+ww))/year
grid_in_sample=np.array(range(trunc+1))/year
q = figure(width=500, height=400,title='Joint calibration with weight {}'.format(lambd) + ' on the options')

q3 = q.line(grid_in_and_out_sample,path_approximated_2[:trunc+ww], legend_label='SigSDE', line_color='tomato')
q4 = q.circle(grid_in_and_out_sample,path_approximated_2[:trunc+ww], fill_color='tomato')

q5 = q.line(grid_in_and_out_sample,X_daily[:trunc+ww], legend_label='Heston', line_color='navy')
q6= q.line([grid_in_sample[-1],grid_in_sample[-1]], [min(path_approximated_1[:trunc+ww]),max(path_approximated_1[:trunc+ww])], line_width=1,line_color='black',line_dash='dashed')
q.legend.location = "bottom_left"
#show(q)


bb = figure(width=600, height=500)

#b1 = b.circle(strikes[j],iv_calib_arr_mc_slice1,legend_label='Sig IV',fill_alpha=0.2,size=7,color="red")
bb.circle(strikes[j],iv_calib_arr_mc_slice3,legend_label='Sig IV',fill_alpha=0.2,size=7,color="forestgreen")
bb.star(strikes[j],iv[j],legend_label='Heston IV',fill_alpha=0.2,size=7,color="navy")
bb.xaxis.axis_label = 'Strike'
bb.yaxis.axis_label = 'IV'
bb.legend.location = "top_right"



kk = figure(width=600, height=500)

#k.circle(strikes[j],iv_calib_arr_mc_slice2,legend_label='Sig IV',fill_alpha=0.2,size=7,color="red")
kk.circle(strikes[j],iv_calib_arr_mc_slice4,legend_label='Sig IV',fill_alpha=0.2,size=7,color="forestgreen")
kk.star(strikes[j],iv[index_sel_maturities[1]],legend_label='Heston IV',fill_alpha=0.2,size=7,color="navy")
kk.xaxis.axis_label = 'Strike'
kk.yaxis.axis_label = 'IV'
kk.legend.location = "top_right"
show(row(q,bb,kk))

