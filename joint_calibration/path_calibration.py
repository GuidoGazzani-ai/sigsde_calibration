# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:20:41 2021

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


from scipy.integrate import quad
import cmath
import numpy as np
import matplotlib.pyplot as plt
from math import isnan
from scipy.optimize import bisect
from scipy.stats import norm
import os
from scipy.optimize import minimize





def brownian_generator(N):
    time_step=1/(N)**(0.5)
    W_random=np.random.randn(N)
    W_random[0]=0
    W_values=np.cumsum(time_step*W_random)
    t_scaled=[k/N for k in range(N)]
    W=[[t_scaled[k],W_values[k]] for k in range(N)] #augmented Brownians to have uniqueness of the Sig
    return t_scaled, np.array(W), W_values

def number_of_parameters_gen(n,comp_of_path):
    def sum_of_powers(x,y):
        if y<=0: return 1
        return x**y + sum_of_powers(x,y-1)
    return sum_of_powers(comp_of_path,n)

def zero_list_maker(n):
    listofzeros = [0] * n
    return listofzeros



def create_list_words_SV(order_Signature,comp_of_path):
    "Input: int, order of the Signature and int, number of components of the path"
    "Example: for words reflecting order 2 of the path t--->(t,B_t,W_t), use create_list_words_SV(2,3)"
    "Returns a list of the possible words (as strings) with respect to the signature order in input"
    list_of_components=list(range(1,comp_of_path+1))
    l=[]
    aux_final=[]
    for k in range(0,order_Signature+1):
        aux=[p for p in itt.product(list_of_components,repeat=k)]
        l.append(aux)
    last_list=[]
    for k in range(len(l)):
        for j in range(len(l[k])):
            last_list.append(list(l[k][j]))
    x_auxiliary=last_list
    last_list=tuple([tuple(k) for k in last_list])
    aux_final=[str(last_list[k]).replace(" ","") for k in range(len(last_list))]
    for i in list_of_components:
        aux_final[i]=aux_final[i].replace(",","")
    return aux_final, x_auxiliary   #aux_final is the list of strings, x_auxiliary the list of words as lists




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


def build_Sig_data_frame_sv(order_Signature,noises,N,comp_of_path,time):
    order_Signature=order_Signature+1
    Sig=np.array([iisignature.sig(noises[:k+1],order_Signature) for k in range(N)])
    Sig=np.insert(Sig,0,1,axis=1)
    keys=esig.sigkeys(comp_of_path,order_Signature).strip().split(" ")
    L=zero_list_maker(number_of_parameters_gen(order_Signature,comp_of_path))
    my_dictionary={key: value for key, value in zip(keys, L)}
    names=list(my_dictionary.keys())
    Sig_data_frame = pd.DataFrame(Sig, columns=names)
    return Sig_data_frame, keys, Sig

def augment_noises(B,W):
    t_scaled=[k/len(W) for k in range(len(W))]
    noises=np.array([[t_scaled[k],B[k],W[k]] for k in range(len(W))])
    return noises

def transform_data_frame_sv(Sig_data_frame,new_tilde,keys,comp_of_path,rho,order_Signature):
    aus_B=[] #auxiliary empty lists
    aus_W=[]
    x, y=create_list_words_SV(order_Signature,comp_of_path)
    for k in range(len(y)):
        if k==0:
            aus_B.insert(0,Sig_data_frame['(2)'])
            aus_W.insert(0,Sig_data_frame['(3)'])
        if (k>0 and y[k][-1]==1):
            aus_B.append(Sig_data_frame[new_tilde[k-1][0]])
            aus_W.append(Sig_data_frame[new_tilde[k-1][1]])
        if (k>0 and y[k][-1]==2):
            aus_B.append(Sig_data_frame[new_tilde[k-1][0]]-0.5*Sig_data_frame[new_tilde[k-1][2]])
            aus_W.append(Sig_data_frame[new_tilde[k-1][1]]-rho*0.5*Sig_data_frame[new_tilde[k-1][2]])
        if (k>0 and y[k][-1]==3):
            aus_B.append(Sig_data_frame[new_tilde[k-1][0]]-rho*0.5*Sig_data_frame[new_tilde[k-1][2]])
            aus_W.append(Sig_data_frame[new_tilde[k-1][1]]-0.5*Sig_data_frame[new_tilde[k-1][2]])

    new_keys_W=[esig.sigkeys(3,order_Signature).strip().split(" ")[k]+str('~W') for k in range(len(x))]
    new_keys_B=[esig.sigkeys(3,order_Signature).strip().split(" ")[k]+str('~B') for k in range(len(x))]

    new_dictionary_B={key:series for key,series in zip(new_keys_B,aus_B)}        
    new_dictionary_W={key:series for key,series in zip(new_keys_W,aus_W)}                           
    transformed_data_frame_W=pd.DataFrame(new_dictionary_W)
    transformed_data_frame_B=pd.DataFrame(new_dictionary_B)

    df_concat_new = pd.concat([transformed_data_frame_B, transformed_data_frame_W], axis=1)
    return df_concat_new,new_keys_B