import pandas as pd
import numpy as np
from sys import path
path.append("..")
from src.utils import *
from plot_settings import *
from markov_funcs import *
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

data = pd.read_parquet("../data/NSW/nsw_df.parquet")
df = data.loc['2019-08-01':'2020-02-14']
df

# plot_correlation_heatmap(data)
# plot_rolling_correlations(data, span=48*252, portfolio='TOTALDEMAND')

#### Markov Model
from sklearn.preprocessing import StandardScaler
df.columns
df
df['month'] += 1
df['doy'] += 1
df['hour'] += 1

#Backward Elimination Linear Regression
def stepwise_backwards_regression(y, X, p_thres=0.05):
    cols = list(X.columns)
    pmax = 1
    while (len(cols)>0):
        p = []
        X_1 = X[cols]
        # X_1 = sm.add_constant(X_1)
        X_1['const'] = 1
        model = sm.OLS(y, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)    
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if(pmax>p_thres):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    # pred = model.predict(X_1)
    print(selected_features_BE)
    print(model.summary())
    return selected_features_BE

## Choose reponse and predictor
# cols = ['TOTALDEMAND', 'rrp', 'dow', 'TEMPERATURE']
cols = ['TOTALDEMAND', 
        'TEMPERATURE', 'rrp',
        'dow',
        'minutes_past_midnight']
start_0 = '2020-01-01'
# tmpdf = np.log(df[cols]).dropna().loc[start_0:]
# tmpdf = tmpdf[(tmpdf != 0) & (tmpdf != -np.inf)].dropna() ## fails standard scaler otherwise
tmpdf = np.log(df[cols]).loc[start_0:]
tmpdf = tmpdf[(tmpdf != 0) & (tmpdf != -np.inf)] ## fails standard scaler otherwise
tmpdf.index.name = 'Date'
tmpdf.index = pd.to_datetime(tmpdf.index)
tmpdf = tmpdf.interpolate(method='time').bfill()
tmpdf.info()

## Add lags
# tmpdf['lag1'] = tmpdf['TOTALDEMAND'].shift(1).fillna(0)
# tmpdf['lag2'] = tmpdf['TOTALDEMAND'].shift(2).fillna(0)

plot_correlation_heatmap(tmpdf)

y = tmpdf['TOTALDEMAND']
X = tmpdf.drop('TOTALDEMAND',axis=1)
features = stepwise_backwards_regression(y, X)
features.append('TOTALDEMAND')

## Scale data for model
tmpdf_scaled = StandardScaler().set_output(transform='pandas').fit_transform(tmpdf[features])
tmpdf_scaled.plot()

## Smooth out data with kalman filter
tmpdf_scaled_smoothed = tmpdf_scaled.apply(lambda x: KalmanFilterAverage(x))
tmpdf_scaled_smoothed.plot()

## Train/ Test split
days = 7
X_test = tmpdf_scaled_smoothed.iloc[-days*48:][[cols[0]]]
X_train = tmpdf_scaled_smoothed.iloc[:-days*48]
print(X_train.shape[0], X_test.shape[0])

## Smooth out data with kalman filter
endog = X_train[cols[0]]
exog = X_train.drop(cols[0],axis=1)

#################### Markov Regression Model
#########################################
k_regimes = 2
np.random.seed(k_regimes+1)
model = sm.tsa.MarkovRegression(endog=endog, 
                                k_regimes=k_regimes, 
                                trend='c', 
                                switching_trend=True,
                                switching_exog=True,
                                switching_variance=True,
                                exog=exog)
model_res = model.fit(search_reps=10)
model_res.summary()
print(model_res.expected_durations) # 30 minute blocks

def plot_regimes(endog, model_res, exogs, X_test=None, k_regimes=2, plot_exogs=False, title=None):
    if not isinstance(endog, pd.DataFrame):
        endog = pd.DataFrame(endog)
    
    # model_res, exogs = model_ar, exog
    
    cols = endog.columns
    ## Get exog labels
    exog_labels = exogs.columns
        
    ## Get params of differing states and find highest const state
    const = model_res.params['const[0]']
    prob_ind = 0
    regime_params = {}
    for regime in range(k_regimes):
        params = [i for i in model_res.params.index if f'[{regime}]' in i]
        if model_res.params[params[0]] > const:
            const = model_res.params[params[0]]
            prob_ind += 1
        regime_params[regime] = params
        
    ## Build plot
    if X_test is not None:
        if 'p[0->0].tvtp0' in model_res.params:
            fig, axs = plt.subplots(k_regimes+3,1,figsize=(18,8))
        else:
            fig, axs = plt.subplots(k_regimes+2,1,figsize=(18,8))
    else:
        if 'p[0->0].tvtp0' in model_res.params:
            fig, axs = plt.subplots(k_regimes+2,1,figsize=(18,8))
        else:
            fig, axs = plt.subplots(k_regimes+1,1,figsize=(18,8))
    
    fig.subplots_adjust(hspace=1)
    
    #### Plot the Traing Set & Fitted Values
    # if plot_exogs:
    #     for exog in exogs: 
    #         exogs[exog].plot(ax=axs[0], linewidth=2)
    endog.plot(ax=axs[0], linewidth=4)
    model_res.fittedvalues.plot(ax=axs[0], label='Fitted Values', linewidth=4, linestyle='-.')

    ## Add in the means of each regime
    cs = ['r', 'g', 'm', 'b']
    mu_val = 0
    for key in regime_params.keys():
        # labels = model_res.params[regime_params[key]].index[:-1]
        for ind, val in enumerate(model_res.params[regime_params[key]][:-1]):
            if ind == 0:
                mu_label = rf'$\mu_{mu_val}$'
                axs[0].axhline(y=val, label=mu_label, linestyle='dotted', c=cs[mu_val])
                mu_val += 1
    axs[0].legend()
        
    #### Plot of marginal probabilities
    model_res.smoothed_marginal_probabilities[prob_ind].plot(ax=axs[1], linewidth=3)
    if prob_ind==0:
        axs[1].set_title('Smoothed Marginal Probability of low-mean regime')
    else:
        axs[1].set_title('Smoothed Marginal Probability of high-mean regime')
        
    #### Plot Feature Importance
    if 'ar.L1' in model_res.params:
        ## For Markov AutoRegression
        ar_vals = []
        ar_params = [i for i in model_res.params.index if 'ar' in i]
        for ar_param in ar_params:
            ar_vals.append(model_res.params[ar_param])
            
        ## Plot of Autoregressive coefficients 
        axs[2].bar(range(len(ar_vals)), ar_vals)
        axs[2].set_xlabel("Lag")
        # axs[2].set_ylabel("AR Coefficient")
        axs[2].set_title("Autoregressive Coefficients")
        axs[2].set_xticks(range(len(ar_vals)))  # Set x-axis ticks for each lag
        axs[2].grid(axis='y')
    else:
        #### Plot the effect of exogenous variables on ax[1]
        for key in regime_params.keys():
            # labels = model_res.params[regime_params[key]].index[:-1]
            for ind, val in enumerate(model_res.params[regime_params[key]][:-1]):
                label_ = str(exog_labels[ind-1]) + f"_{key}"
                if ind != 0:
                    axs[2].bar(label_, val, alpha=0.5)
        axs[2].set_title("Feature Importance for each state")
                
    # axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=-45)
    if title is not None:
        axs[0].set_title(f'{endog.columns[0]} {title}')
    else:
        axs[0].set_title(f'{endog.columns[0]}')

    ##### Plot of prediction vs actual
    if X_test is not None:
        predict = model_res.predict(start=0, end=len(X_test) - 1)
        axs[3].plot(X_test.index, X_test[cols[0]].values, label='Actual', linewidth=3)
        axs[3].plot(X_test.index, predict.values, label='Predicted', linewidth=3)
        axs[3].set_title("Test Set vs Predicted")
        axs[3].set_xticks(X_test.index[::len(X_test) // 7])
        axs[3].legend()
    
    if 'p[0->0].tvtp0' in model_res.params:
        ## For Markov AutoRegression
        tvtp_vals = []
        tvtp_params = [i for i in model_res.params.index if 'tvtp' in i]
        for tvtp_param in tvtp_params:
            tvtp_vals.append(model_res.params[tvtp_param])
        
        xrange = np.linspace(np.min(tvtp_vals), np.max(tvtp_vals), 100)
        
        # Extract probabilities method 1
        p00_0 = model_res.params['p[0->0].tvtp0']
        p00_1 = model_res.params['p[0->0].tvtp1']
        p10_0 = model_res.params['p[1->0].tvtp0']
        p10_1 = model_res.params['p[1->0].tvtp1']
        p_stay_0 = np.exp(p00_0 + p00_1 * xrange) / (1 + np.exp(p00_0 + p00_1 * xrange))
        p_stay_1 = np.exp(p10_0 + p10_1 * xrange) / (1 + np.exp(p10_0 + p10_1 * xrange))

        # # Extract probabilities method 2
        # p_stay_vals = {}
        # for i in range(1, int(len(tvtp_vals)/2+2), 2):
        #     # i = 3
        #     val = np.exp(tvtp_vals[i-1] + tvtp_vals[i] * xrange) / (1 + np.exp(tvtp_vals[i-1] + tvtp_vals[i] * xrange))
        #     p_stay_vals[i-1] = val
            
        # # Plot transition probabilities vs temperature
        # for key in p_stay_vals.keys():
        #     plt.plot(xrange, p_stay_vals[key], label=f'Regime {key} -> Regime {key}')
        
        axs[4].plot(xrange, p_stay_0, label='Regime 0 -> Regime 0')
        axs[4].plot(xrange, p_stay_1, label='Regime 1 -> Regime 1')
        # axs[4].xlabel('Temperature')
        # axs[4].ylabel('Transition Probability')
        axs[4].set_title('Effect of Exog on Regime Transition Probabilities')
        axs[4].legend()
        axs[4].grid(True)
        
        # plt.plot(xrange, p_stay_0, label='Regime 0 -> Regime 0')
        # plt.plot(xrange, p_stay_1, label='Regime 1 -> Regime 1')
        # # plt.xlabel('Temperature')
        # # plt.ylabel('Transition Probability')
        # plt.title('Effect of Exog on Regime Transition Probabilities')
        # plt.legend()
        # plt.grid(True)
        
plot_regimes(endog, model_res, exog, X_test)

## Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
predict_res = model_res.predict(start=0, end=len(X_test) - 1)
print('MAE:', mean_absolute_error(X_test['TOTALDEMAND'], predict_res).round(2))
print('RMSE:', np.sqrt(mean_squared_error(X_test['TOTALDEMAND'], predict_res)).round(2))


#################### Markov Auto-Regression Model
#########################################
k_regimes = 2
np.random.seed(k_regimes+1)
model_ar = sm.tsa.MarkovAutoregression(endog=endog, 
                                       order=3, 
                                       k_regimes=k_regimes, 
                                       switching_ar=False, 
                                       switching_trend=True,
                                       switching_exog=True,
                                       switching_variance=True,
                                       exog_tvtp=sm.add_constant(exog['TEMPERATURE'])
                                       )
model_ar = model_ar.fit(search_reps=10)
model_ar.summary()
print(model_ar.expected_durations) # 30 minute blocks

plot_regimes(endog, model_ar, exog, X_test)

## Metrics
predict_ar = model_ar.predict(start=0, end=len(X_test) - 1)
print('MAE:', mean_absolute_error(X_test['TOTALDEMAND'], predict_ar).round(2))
print('RMSE:', np.sqrt(mean_squared_error(X_test['TOTALDEMAND'], predict_ar)).round(2))




























######## Hidden Markov Model
# def hiddenMarkovModel(prices, col, hidden_states=2, d=0.94, start='2024-01-01', gaussianHMM=False, train_test_split=True, plot=True, remove_direction_and_mean=True, verbose=True,
#                       algorithm='viterbi', cov_type='diag'):
#     from hmmlearn import hmm
#     #algorithm could be 'map'
#     """
#     Initial hidden state probabilities 𝝅 = [𝝅₀, 𝝅₁, 𝝅₂, …]ᵀ. This vector describes
#     the initial probabilities of the system being in a particular hidden state.
    
#     Hidden state transition matrix A. Each row in A corresponds to a particular
#     hidden state, and the columns for each row contain the transition probabilities
#     from the current hidden state to a new hidden state. For example, A[1, 2] contains
#     the transition probability from hidden state 1 to hidden state 2.
    
#     Observable emission probabilities 𝜽 = [𝜽₀, 𝜽₁, 𝜽₂, …]ᵀ. This vector describes
#     the emission probabilities for the observable process Xᵢ given some hidden
#     state Zᵢ.
#     """
#     # prices = tmpdf
#     if isinstance(col, str):
#         col = [col] 
#     tmpcol = prices[col].dropna()
#     tmpcol = tmpcol[tmpcol != 0].dropna()
#     # tmpret = np.log(tmpcol).diff().fillna(0)
#     tmpret = tmpcol.pct_change().fillna(0).astype(float)
#     tmpret = tmpret[tmpret != -np.inf]
#     if train_test_split:
#         ## Train test split
#         X_train, X_test = subset_data(tmpret, train_split_func=True, test_size=0.1)
#         x_train = X_train.values.reshape(-1,1)
#         x_test = X_test.values.reshape(-1,1)
        
#         if remove_direction_and_mean:
#             rollingVol = X_train.ewm(span=252*3).std()
#             rollingMu  = X_train.ewm(span=252*3).mean()

#             x_train = x_train - rollingVol.fillna(0).values.reshape(-1,1)
#             x_train = x_train - rollingMu.fillna(0).values.reshape(-1,1)
            
#             X_test = X_test - X_test.ewm(span=252*3).std().fillna(0)
#             X_test = X_test - X_test.ewm(span=252*3).mean().fillna(0)   
        
#     # plt.plot(x_train) # check if looking stationary as it makes the HMM more consistent
    
#     ## Build and fit model
#     if gaussianHMM:
#         model = hmm.GaussianHMM(n_components=hidden_states, covariance_type=cov_type,
#                                 n_iter=100, random_state=42, 
#                                 verbose=False, algorithm=algorithm)
#     else:
#         model = hmm.GMMHMM(n_components=hidden_states, covariance_type=cov_type,
#                                 n_iter=100, random_state=42, 
#                                 verbose=False, algorithm=algorithm)
        
#     if train_test_split:
#         x_train = (x_train - np.nanmean(x_train))/np.nanstd(x_train,ddof=0)
#         x_train = np.nan_to_num(x_train)
#         X_test = (X_test - np.nanmean(X_test))/np.nanstd(X_test,ddof=0)
#         # X_test = np.nan_to_num(X_test)
#         model.fit(x_train)
            
#         ## Predict the hidden states corresponding to the observed values
#         score, Z = model.decode(x_train)
#         states = pd.unique(Z)
        
#         if verbose:
#             print("\nLog Probability & States:")
#             print(score, states)
            
#             print("\nStarting probabilities:")
#             print(model.startprob_.round(2))
            
#             print("\nTransition matrix:")
#             print(model.transmat_.round(3))
            
#             print("\nGaussian distribution means:")
#             print(model.means_.round(4))
            
#             print("\nGaussian distribution covariances:")
#             print(model.covars_.round(4))

#         if plot:
#             if start is None:
#                 start = X_test.index[0]
#                 start_label = str(X_train.index[-1]).split(" ")[0]
#             else:
#                 start_label = start
#             x_test = X_test.loc[start:].values.reshape(-1,1)
#             score_test, Z_test = model.decode(x_test)
#             # Plot the price chart
#             plt.figure(figsize=(15, 8))
#             subplots = hidden_states + 2
#             colors = ['r', 'g', 'b']
#             plt.subplot(subplots, 1, 1)
#             for i in states:
#                 want = (Z == i)
#                 try:
#                     price = tmpcol.loc[:X_train.index[-1]]
#                     if price.shape[0] != len(want):
#                         raise ValueError('break')
#                 except:
#                     price = tmpcol.loc[:X_train.index[-1]].iloc[:-1]
#                 x = price[want].index
#                 y = price[want]
#                 plt.plot(x, y, '.', label=f'State: {i+1}', c=colors[i])
#             plt.title(f'{str(col[0])} up to {str(X_train.index[-1]).split(" ")[0]}')
#             plt.legend()
#             # Plot the smoothed marginal probabilities
#             plt.subplot(subplots, 1, 2)
#             for i in range(hidden_states):
#                 state_probs = pd.Series(model.predict_proba(x_train)[:, i], index=X_train.loc[:start].index)
#                 state_probs_smooth = state_probs.ewm(alpha=1-d).mean()
#                 plt.plot(state_probs_smooth, label=f'State {i+1}', alpha=0.5, c=colors[i])
#             plt.legend()
#             plt.title('Smoothed Marginal Probabilities (Train)')
#             # Plot the smoothed marginal probabilities for x_test
#             plt.subplot(subplots, 1, 3)
#             for i in states:
#                 want = (Z_test == i)
#                 price = tmpcol[1:].loc[start:]
#                 x = price[want].index
#                 y = price[want]
#                 plt.plot(x, y, '.', label=f'State: {i+1}', c=colors[i])
#             plt.title(f'{str(col[0])} from {start_label} onwards')
#             plt.legend()
#             plt.subplot(subplots, 1, 4)
#             for i in range(hidden_states):
#                 state_probs_test = pd.Series(model.predict_proba(x_test)[:, i], index=X_test.loc[start:].index)
#                 state_probs_smooth = state_probs_test.ewm(alpha=1-d).mean()
#                 plt.plot(state_probs_smooth, label=f'State {i+1}', c=colors[i])
#             plt.legend()
#             plt.title('Smoothed Marginal Probabilities (Test)')
            
#             plt.tight_layout()
#             plt.show()
    
#     else:
#         tmpret = (tmpret - np.nanmean(tmpret))/np.nanstd(tmpret,ddof=0)
#         model.fit(tmpret)
#         ## Predict the hidden states corresponding to the observed values
#         score, Z = model.decode(tmpret)
#         states = pd.unique(Z)
        
#         if verbose:
#             print("\nLog Probability & States:")
#             print(score, states)
            
#             print("\nStarting probabilities:")
#             print(model.startprob_.round(2))
            
#             print("\nTransition matrix:")
#             print(model.transmat_.round(3))
            
#             print("\nGaussian distribution means:")
#             print(model.means_.round(4))
            
#             print("\nGaussian distribution covariances:")
#             print(model.covars_.round(4))
        
#         if plot:
#             if start is None:
#                 start = tmpret.index[0]
#                 start_label = str(tmpret.index[-1]).split(" ")[0]
#             else:
#                 start_label = start
#             tmpret_ = tmpret.loc[start:].values.reshape(-1,1)
#             score_test, Z_test = model.decode(tmpret_)
#             # Plot the price chart
#             plt.figure(figsize=(15, 8))
#             subplots = 2
#             colors = ['r', 'g', 'b']
#             plt.subplot(subplots, 1, 1)
#             for i in states:
#                 want = (Z == i)
#                 try:
#                     price = tmpcol.loc[:tmpret.index[-1]]
#                     if price.shape[0] != len(want):
#                         raise ValueError('break')
#                 except:
#                     price = tmpcol.loc[:tmpret.index[-1]].iloc[:-1]
#                 x = price[want].index
#                 y = price[want]
#                 plt.plot(x, y, '.', label=f'State: {i+1}', c=colors[i])
#             plt.title(f'{str(col[0])} up to {str(tmpret.index[-1]).split(" ")[0]}')
#             plt.legend()
#             # Plot the smoothed marginal probabilities
#             plt.subplot(subplots, 1, 2)
#             for i in range(hidden_states):
#                 state_probs = pd.Series(model.predict_proba(tmpret)[:, i], index=tmpret.index)
#                 state_probs_smooth = state_probs.ewm(alpha=1-d).mean()
#                 plt.plot(state_probs_smooth, label=f'State {i+1}', alpha=0.5, c=colors[i])
#             plt.legend()
#             plt.title('Smoothed Marginal Probabilities')
#             plt.tight_layout()
#             plt.show()
    
#     return model

# col = 'rrp'
# tmpdf = np.log(df[[col]]).dropna()

# #### Build model
# # model = hiddenMarkovModel(tmpdf, col, hidden_states=hidden_states, d=d, start=start, gaussianHMM=gaussianHMM, train_test_split=train_test_split, remove_direction_and_mean=remove_direction_and_mean, verbose=verbose, algorithm=algorithm, cov_type=cov_type)
# from sklearn.preprocessing import StandardScaler

# ## Choose reponse and predictor
# cols = ['TOTALDEMAND', 'rrp']
# start_0 = '2020-01-01'
# tmpdf = np.log(df[cols]).dropna().loc[start_0:]
# tmpdf = tmpdf[(tmpdf != 0) & (tmpdf != -np.inf)].dropna() ## fails standard scaler otherwise
# interaction_feature = tmpdf[cols[0]] * tmpdf[cols[1]]
# interaction_feature.plot()

# ## Scale data for model
# scaled = StandardScaler().set_output(transform='pandas').fit_transform(interaction_feature.values.reshape(-1,1))
# scaled.index = interaction_feature.index
# scaled.plot()

# ## Train/ Test split
# days = 7
# X_test = scaled.iloc[-days*48:]
# X_train = scaled.iloc[:-days*48]

# # k_regimes, endog, exog = 2, tmpdf_scaled[cols[0]], tmpdf_scaled[cols[1]]
# # k_regimes, endog, exog = 2, tmpdf_scaled[cols[0]].rolling(48).mean().dropna(), tmpdf_scaled[cols[1]].rolling(48).mean().dropna()
# X_train, X_test = KalmanFilterAverage(X_train), KalmanFilterAverage(X_train)
# x_train, x_test = X_train.values.reshape(-1,1), X_test.values.reshape(-1,1)

# train_test_split, start = True, None
# remove_direction_and_mean, verbose = False, True
# gaussianHMM, hidden_states, d = True, 2, 0.94
# algorithm, cov_type = 'viterbi', 'diag'

# model = hmm.GMMHMM(n_components=hidden_states, covariance_type=cov_type,
#                         n_iter=100, random_state=42, 
#                         verbose=False, algorithm=algorithm)

# model.fit(x_train)
# ## Predict the hidden states corresponding to the observed values
# score, Z = model.decode(x_train)
# states = pd.unique(Z)

# if verbose:
#     print("\nLog Probability & States:")
#     print(score, states)
    
#     print("\nStarting probabilities:")
#     print(model.startprob_.round(2))
    
#     print("\nTransition matrix:")
#     print(model.transmat_.round(3))
    
#     print("\nGaussian distribution means:")
#     print(model.means_.round(4))
    
#     print("\nGaussian distribution covariances:")
#     print(model.covars_.round(4))

# def plot_hmm(): 
#     # if start is None:
#     start = X_test.index[0]
#     start_label = str(X_train.index[-1]).split(" ")[0]
#     # else:
#     #     start_label = start
#     x_test = X_test.loc[start:].values.reshape(-1,1)
#     score_test, Z_test = model.decode(x_test)
#     # Plot the price chart
#     plt.figure(figsize=(15, 8))
#     subplots = hidden_states + 2
#     colors = ['r', 'g', 'b']
#     plt.subplot(subplots, 1, 1)
#     for i in states:
#         want = (Z == i)
#         try:
#             price = tmpdf[cols[0]].loc[:X_train.index[-1]]
#             if price.shape[0] != len(want):
#                 raise ValueError('break')
#         except:
#             price = tmpdf[cols[0]].loc[:X_train.index[-1]].iloc[:-1]
#         x = price[want].index
#         y = price[want]
#         plt.plot(x, y, '.', label=f'State: {i+1}', c=colors[i])
#     plt.title(f'Up to {str(X_train.index[-1]).split(" ")[0]}')
#     plt.legend()
#     # Plot the smoothed marginal probabilities
#     plt.subplot(subplots, 1, 2)
#     for i in range(hidden_states):
#         state_probs = pd.Series(model.predict_proba(x_train)[:, i], index=X_train.loc[:start].index)
#         state_probs_smooth = state_probs.ewm(alpha=1-d).mean()
#         plt.plot(state_probs_smooth, label=f'State {i+1}', alpha=0.5, c=colors[i])
#     plt.legend()
#     plt.title('Smoothed Marginal Probabilities (Train)')
#     # Plot the smoothed marginal probabilities for x_test
#     plt.subplot(subplots, 1, 3)
#     for i in states:
#         want = (Z_test == i)
#         price = tmpdf[cols[0]][1:].loc[start:]
#         x = price[want].index
#         y = price[want]
#         plt.plot(x, y, '.', label=f'State: {i+1}', c=colors[i])
#     plt.title(f'From {start_label} onwards')
#     plt.legend()
#     plt.subplot(subplots, 1, 4)
#     for i in range(hidden_states):
#         state_probs_test = pd.Series(model.predict_proba(x_test)[:, i], index=X_test.loc[start:].index)
#         state_probs_smooth = state_probs_test.ewm(alpha=1-d).mean()
#         plt.plot(state_probs_smooth, label=f'State {i+1}', c=colors[i])
#     plt.legend()
#     plt.title('Smoothed Marginal Probabilities (Test)')

#     plt.tight_layout()
#     plt.show()

# plot_hmm()