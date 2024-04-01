import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tsa
import matplotlib.pyplot as plt
from plot_settings import *
from src.utils import *

nsw_df = pd.read_parquet("../data/NSW/nsw_df.parquet")

###################### 
##################################################
## pip install --upgrade --user hmmlearn
# from hmmlearn import hmm # ,vhmm

# def hiddenMarkovModel(prices, col, hidden_states=2, d=0.94, start='2024-01-01', gaussianHMM=False, plot=True, remove_direction_and_mean=True, normalise=True, verbose=True):
#     from hmmlearn import hmm
#     """
#     The reason for using 3 hidden states is that we expect at the very least 3
#     different regimes in the daily changes — low, medium and high votality.
    
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
#     # rollingVols, rollingCorrs = hcb.getRollingVol_and_Corrs(returns)
#     # prices = rollingTmp
#     # col = ['price']
#     tmpcol = prices[col].dropna()
#     tmpcol = tmpcol[tmpcol != 0].dropna()
#     # tmpret = np.log(tmpcol).diff().fillna(0)
#     tmpret = tmpcol.pct_change()[1:]
    
#     ## Train test split
#     X_train, X_test = subset_data(tmpret, train_split_func=True, test_size=0.1)
#     x_train = X_train.values.reshape(-1,1)
#     x_test = X_test.values.reshape(-1,1)

#     if remove_direction_and_mean:
#         rollingVol = X_train.ewm(span=252*3).std()
#         rollingMu  = X_train.ewm(span=252*3).mean()

#         x_train = x_train - rollingVol.fillna(0).values.reshape(-1,1)
#         x_train = x_train - rollingMu.fillna(0).values.reshape(-1,1)
        
#         X_test = X_test - X_test.ewm(span=252*3).std().fillna(0)
#         X_test = X_test - X_test.ewm(span=252*3).mean().fillna(0)   
        
#     # plt.plot(x_train) # check if looking stationary as it makes the HMM more consistent
    
#     ## Build and fit model
#     if gaussianHMM:
#         model = hmm.GaussianHMM(n_components=hidden_states, covariance_type='diag',
#                                 n_iter=100, random_state=42, verbose=False)
#     else:
#         model = hmm.GMMHMM(n_components=hidden_states, covariance_type='diag',
#                                 n_iter=100, random_state=42, verbose=False)
    
#     x_train = (x_train - np.nanmean(x_train, ))/np.nanstd(x_train,ddof=0)
#     X_test = (X_test - np.nanmean(X_test))/np.nanstd(X_test,ddof=0)
#     model.fit(x_train)
        
#     ## Predict the hidden states corresponding to the observed values
#     score, Z = model.decode(x_train)
#     states = pd.unique(Z)
    
#     if verbose:
#         print("\nLog Probability & States:")
#         print(score, states)
        
#         print("\nStarting probabilities:")
#         print(model.startprob_.round(2))
        
#         print("\nTransition matrix:")
#         print(model.transmat_.round(3))
        
#         print("\nGaussian distribution means:")
#         print(model.means_.round(4))
        
#         print("\nGaussian distribution covariances:")
#         print(model.covars_.round(4))

#     if plot:
#         if start is None:
#             start = X_test.index[0]
#             start_label = str(X_train.index[-1]).split(" ")[0]
#         else:
#             start_label = start
        
#         try:
#             x_test = X_test.loc[start:].values.reshape(-1,1)
#         except:
#             x_test = X_test.values.reshape(-1,1)
#         score_test, Z_test = model.decode(x_test)
#         # Plot the price chart
#         plt.figure(figsize=(15, 8))
#         subplots = 4
#         colors = ['r', 'g', 'b']
#         plt.subplot(subplots, 1, 1)
#         for i in states:
#             want = (Z == i)
#             try:
#                 price = tmpcol.loc[:X_train.index[-1]]
#                 if price.shape[0] != len(want):
#                     raise ValueError('break')
#             except:
#                 price = tmpcol.loc[:X_train.index[-1]].iloc[:-1]
#             x = price[want].index
#             y = price[want]
#             plt.plot(x, y, '.', label=f'State: {i+1}', c=colors[i])
#         plt.title(f'{col} up to {str(X_train.index[-1]).split(" ")[0]}')
#         plt.legend()
#         # Plot the smoothed marginal probabilities
#         plt.subplot(subplots, 1, 2)
#         for i in range(hidden_states):
#             state_probs = pd.Series(model.predict_proba(x_train)[:, i])
#             state_probs_smooth = state_probs.ewm(alpha=1-d).mean()
#             plt.plot(state_probs_smooth, label=f'State {i+1}', alpha=0.5, c=colors[i])
#         plt.legend()
#         plt.title('Smoothed Marginal Probabilities (Train)')
#         # Plot the smoothed marginal probabilities for x_test
#         plt.subplot(subplots, 1, 3)
#         for i in states:
#             want = (Z_test == i)
#             price = tmpcol[1:].loc[start:]
#             x = price[want].index
#             y = price[want]
#             plt.plot(x, y, '.', label=f'State: {i+1}', c=colors[i])
#         plt.title(f'{col} from {start_label} onwards')
#         plt.legend()
#         plt.subplot(subplots, 1, 4)
#         for i in range(hidden_states):
#             state_probs_test = pd.Series(model.predict_proba(x_test)[:, i])
#             state_probs_smooth = state_probs_test.ewm(alpha=1-d).mean()
#             plt.plot(state_probs_smooth, label=f'Test State {i+1}', c=colors[i])
#         plt.legend()
#         plt.title('Smoothed Marginal Probabilities (Test)')
        
#         plt.tight_layout()
#         plt.show()

#     return model

# data
# tmpcol = data[['TOTALDEMAND']]
# tmpcol.index = data['LASTCHANGED']
# tmpcol = tmpcol.iloc[:1_000_000]
# # tmpcol = tmpcol.iloc[:5_000]
# tmpcol.info()

# tmpcol.plot()

# rollingTmp = tmpcol.rolling(window=5000).mean()

# col = 'TOTALDEMAND'
# plot=True
# remove_direction_and_mean, verbose = False, True
# gaussianHMM, hidden_states, d = True, 2, 0
# model = hiddenMarkovModel(np.log(rollingTmp.dropna()), col, hidden_states=hidden_states, d=d, gaussianHMM=gaussianHMM, remove_direction_and_mean=remove_direction_and_mean, verbose=verbose, plot=plot)



## Hash rate

def getBitcoinHashData(path="dataHMM/hash-rate.json", plot=True):
    import datetime as dt
    hash = pd.read_json(path)
    hash = hash[['hash-rate', 'market-price']]

    data = pd.DataFrame(columns=['hash', 'price'])
    for col in hash.columns:
        for loc in hash.index:
            d = hash.loc[loc, col]
            keys = list(d.keys())
            if col == 'hash-rate':
                data.loc[d[keys[0]], 'hash'] = d[keys[1]]
            else:
                data.loc[d[keys[0]], 'price'] = d[keys[1]]

    data = data.astype(float)
    data['returns'] = np.log(data['price']).diff()
    data.index = [dt.datetime.fromtimestamp(ind/1000) for ind in data.index]
    print(data.info())
    if plot:
        # Create the dual axis plot
        fig, ax1 = plt.subplots()
        # Plot 'hash' data on the primary y-axis (left)
        ax1.plot(data['hash'], 'b-o', label='Hash Rate')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Hash Rate', color='b')

        # Create a twin axis for 'price' data on the secondary y-axis (right)
        ax2 = ax1.twinx()
        ax2.plot(data['price'], 'r-s', label='Price')
        ax2.set_ylabel('Price', color='r')

        # Add labels and title
        plt.title('Dual Axis Plot (Hash Rate vs. Price)')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.show()
    return data

path = r"/Users/reubenbowell/MEGA/VSCode/team-O/reuben/dataRB/hash-rate.json"
data = getBitcoinHashData(path)
data

inds = nsw_df['DATETIME'].drop_duplicates().index
tmp = nsw_df.loc[inds].set_index('DATETIME')
tmp.info()
cols = ['TOTALDEMAND',
       'FORECASTDEMAND', 
       'TEMPERATURE']

returns = tmp[cols].pct_change()[1:]
returns


plot_rolling_correlations(data, portfolio=['hash'])

tmp.columns
plot_rolling_correlations(returns, portfolio=['TOTALDEMAND'])


plot_correlation_heatmap(returns)





# remove_direction_and_mean, verbose = False, False
# gaussianHMM, hidden_states, d = False, 2, 0.5
# model = hiddenMarkovModel(data.loc['2015-01-01':], 'hash', hidden_states=hidden_states, d=d, gaussianHMM=gaussianHMM, remove_direction_and_mean=remove_direction_and_mean, verbose=verbose)
