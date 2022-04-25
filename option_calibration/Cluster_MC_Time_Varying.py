import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import least_squares




def duplicate(testList, n):
    x=[list(testList) for _ in range(n)]
    flat_list = []
    for sublist in x:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def multi_maturities(maturities,k):
    mat=list(maturities)
    new_multi_mat=[]
    for element in mat:
        for j in range(k):
            new_multi_mat.append(element)
    return np.array(new_multi_mat)

def phi(x): ## Gaussian density
    return np.exp(-x*x/2.)/np.sqrt(2*np.pi)    
#### Black Sholes Vega
def BlackScholesVegaCore(DF,F,X,T,v):   #S=F*DF
    vsqrt=v*np.sqrt(T)
    d1 = (np.log(F/X)+(vsqrt*vsqrt/2.))/vsqrt
    return F*phi(d1)*np.sqrt(T)/DF



moneyness=np.load('strikes_SPX_Bloomberg.npy')
maturities=np.load('maturities_SPX_Bloomberg.npy')
iv_market=np.load('iv_spx_170321_bymat.npy')
market_prices=np.load('prices_optionsSPX_170321.npy')

initial_price=100
strikes_all=duplicate(moneyness*initial_price,len(maturities))
strikes=np.array([np.array(strikes_all[j*(len(moneyness)):(j+1)*(len(moneyness))]) for j in range(len(maturities))])
maturities_ext=multi_maturities(maturities,len(strikes[0]))
strike_flat=strikes.flatten()
option_prices_splitted=np.array(np.split(market_prices,len(maturities)))



prices_scaled=option_prices_splitted/100
print(prices_scaled)


initial_price=1


if initial_price==100:
    Premium=market_prices
elif initial_price==1:
    Premium=prices_scaled.flatten()

strikes=strikes/100



flag_truncation=False
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

flat_normal_weights, norm_vegas=get_vegas(maturities, strikes, initial_price, iv_market, flag_truncation)




def count_param(m,nb_components):
    return sum([nb_components**k for k in range(m+1)])

# Recall that the object arr_dfs_by_mat has to be computed by the user; functions are given in the other files
# Observe that it only depends on maturities and that can be easily parallelized



MC_number, N, rho, n= 1000000, 365, -0.5, 2
d,D=2,1
rho_matrix=[[1,rho],[rho,1]]

arr_dfs_by_mat=np.load('arr_dfs_by_mat((7, 1000000, 13)).npy')
arr_dfs_by_mat=np.swapaxes(arr_dfs_by_mat,0,1)
print('Collected successfully with shape: ',arr_dfs_by_mat.shape)
ell_first_maturity=np.load('ell_MC(2,2,-0.5,1000000,1,[0]).npy')
#the first maturity needs no adjustment therefore we fit it with the classical model (here we load the found set of parameters)

l_calibrated=[]
l_calibrated.append(ell_first_maturity)


for j in range(1,len(maturities)):
    sel_maturities=[maturities[j]]  
    index_sel_maturities=[j] 
    
    Vega_W=np.array([np.split(flat_normal_weights,len(maturities))[idx] for idx in index_sel_maturities]).flatten()
    Premium1=np.array([np.split(Premium,len(maturities))[idx] for idx in index_sel_maturities]).flatten()
    
    
    adjusted_arr_dfs=[arr_dfs_by_mat[:,j,:]-arr_dfs_by_mat[:,idx,:] for idx in range(0,j-1)]
    adjusted_arr_dfs.insert(0,arr_dfs_by_mat[:,j,:])
    adjusted_arr_dfs=np.array(adjusted_arr_dfs)
    arr_dfs_to_optimize=arr_dfs_by_mat[:,j,:]-arr_dfs_by_mat[:,j-1,:]
    adjusted_scalar_products=np.array([np.tensordot(adjusted_arr_dfs[k],l_calibrated[k],axes=1) for k in range(j)])
    adjusted_scalar_products=adjusted_scalar_products.transpose()

       
    def time_varying_model(l,idx_maturity,arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,initial_price):
        if idx_maturity==0:
            tensor_sigsde_at_mat_aux=np.tensordot(arr_dfs_by_mat,l,axes=1)+initial_price
            tensor_sigsde_at_mat=tensor_sigsde_at_mat_aux[:,0]
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
        If len(index_sel_maturities)>1 then dim(tensor_sigsde_at_mat)=(nbr_MC_sim,)

        index_sel_maturities (list): list of integers corresponding to the selected maturities
        strikes (np.array 2D): array of arrays where each of the sub-array stores the strikes

        Output: Monte Carlo prices of the model for the selected maturities and strikes
        '''
        pay=[]
        if len(index_sel_maturities)==1:

            for K in strikes[0]:
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



    def obj_MC_tensor_selected_mat_TV(l):
        tensor_sigsde=time_varying_model(l,index_sel_maturities[0],arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,initial_price)
        mc_payoff_arr=get_mc_sel_mat_tensor_TV(tensor_sigsde,index_sel_maturities,strikes)   
        return np.sqrt(Vega_W)*(mc_payoff_arr-Premium1)

        
   
    for t in tqdm(range(1),desc='Fit the maturity nbr. '+str(j+1)): 
        l_initial=np.random.uniform(-0.1,0.1,int(((d+1)**(n+1)-1)*D/d))
        res1 = least_squares(obj_MC_tensor_selected_mat_TV, l_initial,loss='linear')
    
    l_calibrated.append(res1['x'])
    
    np.save(f'ell_MC_TV4({n},{d},{rho},{MC_number},{initial_price},{index_sel_maturities}).npy',res1['x'])
    
    tensor_sigsde_at_mat=time_varying_model(res1['x'],index_sel_maturities[0],arr_dfs_to_optimize,adjusted_scalar_products,arr_dfs_by_mat,initial_price)
    
    calibrated_prices=get_mc_sel_mat_tensor_TV(tensor_sigsde_at_mat,index_sel_maturities,strikes)
    np.save(f'calibrated_prices_MC_TV4({MC_number},{initial_price},{index_sel_maturities},{n})',calibrated_prices)
    

    
