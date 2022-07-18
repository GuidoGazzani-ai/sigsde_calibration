# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:21:54 2021

@author: Guido Gazzani
"""


#Call packages
import itertools as itt  
import iisignature

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import esig
import random
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
from sklearn.metrics import mean_squared_error 
from bokeh.embed import file_html
import chart_studio.plotly as py
import time as tm

def brownian_generator(N):
    time_step=1/(N)**(0.5)
    W_random=np.random.randn(N)
    W_random[0]=0
    W_values=np.cumsum(time_step*W_random)
    t_scaled=[k/N for k in range(N)]
    W=[[t_scaled[k],W_values[k]] for k in range(N)] #augmented Brownians to have uniqueness of the Sig
    return t_scaled, np.array(W), W_values




def Simulate_Heston(N,initial_price,mu,alpha,kappa,theta,initial_vol,rho,t_final,t_0):
    X=np.zeros(N)
    V=np.zeros(N)
    X[0], V[0]= initial_price, initial_vol
    dt=float((t_final-t_0)/N)
    time_grid=np.arange(t_0,t_final,dt)
    dB = np.random.normal(0, np.sqrt(dt), N)
    dB[0]=0
    dW = rho*dB+np.sqrt(1-(rho)**2)*np.random.normal(0,np.sqrt(dt),N)
    dW[0]=0
    B = np.cumsum(dB)
    W = np.cumsum(dW)
    noises=np.array([[time_grid[k],B[k],W[k]] for k in range(N)]) #(t, B_t, W_t)
    for k in range(N-1):
        V[k+1]=V[k]+kappa*(theta-V[k])*(1/N)+(alpha)*(np.sqrt(np.abs(V[k])))*(W[k+1]-W[k]) 
        X[k+1]=X[k]+(X[k])*mu*(1/N)+np.sqrt(np.abs(V[k]))*(X[k])*(B[k+1]-B[k])
    t_scaled=[k/len(X) for k in range(len(X))]
    return X,V,B,W,t_scaled,noises




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