import pandas as pd
import numpy as np
from sys import path
path.append("..")
from src.utils import *
from plot_settings import *
from markov_funcs import *
from hmmlearn import hmm

data = pd.read_csv("../data/NSW/final_df.csv", index_col=0)
df = data.loc['2019-08-01':'2020-02-14']

plot_correlation_heatmap(data)
# plot_rolling_correlations(data, span=48*252, portfolio='TOTALDEMAND')

#### Markov Model
from sklearn.preprocessing import StandardScaler

## Choose reponse and predictor
cols = ['TOTALDEMAND', 'rrp']
tmpdf = np.log(df[cols]).dropna()
tmpdf = tmpdf[(tmpdf != 0) & (tmpdf != -np.inf)].dropna() ## fails standard scaler otherwise

## Scale data for model
tmpdf_scaled = StandardScaler().set_output(transform='pandas').fit_transform(tmpdf)
tmpdf_scaled.plot()

## Train/ Test split
days = 7
X_test = tmpdf_scaled.iloc[-days*48:]
X_train = tmpdf_scaled.iloc[:-days*48]

# k_regimes, endog, exog = 2, tmpdf_scaled[cols[0]], tmpdf_scaled[cols[1]]
# k_regimes, endog, exog = 2, tmpdf_scaled[cols[0]].rolling(48).mean().dropna(), tmpdf_scaled[cols[1]].rolling(48).mean().dropna()
endog, exog = 2, KalmanFilterAverage(X_train[cols[0]]), KalmanFilterAverage(X_train[cols[1]])

k_regimes = 3
model = sm.tsa.MarkovRegression(endog=endog, k_regimes=k_regimes, trend='c', switching_variance=True, exog=exog)
model_res = model.fit(search_reps=10)
model_res.summary()
plot_regimes(endog, model_res, prob_ind=1)
print(model_res.expected_durations) # 30 minute blocks

## Model predict (not proper)
X_test_reset_index = X_test.reset_index(drop=True)
predictions = model_res.predict(start=0, end=len(X_test_reset_index) - 1)

## Plot
predictions.plot()
X_test[cols[0]].plot()

######## Hidden Markov Model

def hiddenMarkovModel(prices, col, hidden_states=2, d=0.94, start='2024-01-01', gaussianHMM=False, train_test_split=True, plot=True, remove_direction_and_mean=True, verbose=True,
                      algorithm='viterbi', cov_type='diag'):
    from hmmlearn import hmm
    #algorithm could be 'map'
    """
    Initial hidden state probabilities 𝝅 = [𝝅₀, 𝝅₁, 𝝅₂, …]ᵀ. This vector describes
    the initial probabilities of the system being in a particular hidden state.
    
    Hidden state transition matrix A. Each row in A corresponds to a particular
    hidden state, and the columns for each row contain the transition probabilities
    from the current hidden state to a new hidden state. For example, A[1, 2] contains
    the transition probability from hidden state 1 to hidden state 2.
    
    Observable emission probabilities 𝜽 = [𝜽₀, 𝜽₁, 𝜽₂, …]ᵀ. This vector describes
    the emission probabilities for the observable process Xᵢ given some hidden
    state Zᵢ.
    """
    # prices = tmpdf
    if isinstance(col, str):
        col = [col] 
    tmpcol = prices[col].dropna()
    tmpcol = tmpcol[tmpcol != 0].dropna()
    # tmpret = np.log(tmpcol).diff().fillna(0)
    tmpret = tmpcol.pct_change().fillna(0).astype(float)
    tmpret = tmpret[tmpret != -np.inf]
    if train_test_split:
        ## Train test split
        X_train, X_test = subset_data(tmpret, train_split_func=True, test_size=0.1)
        x_train = X_train.values.reshape(-1,1)
        x_test = X_test.values.reshape(-1,1)
        
        if remove_direction_and_mean:
            rollingVol = X_train.ewm(span=252*3).std()
            rollingMu  = X_train.ewm(span=252*3).mean()

            x_train = x_train - rollingVol.fillna(0).values.reshape(-1,1)
            x_train = x_train - rollingMu.fillna(0).values.reshape(-1,1)
            
            X_test = X_test - X_test.ewm(span=252*3).std().fillna(0)
            X_test = X_test - X_test.ewm(span=252*3).mean().fillna(0)   
        
    # plt.plot(x_train) # check if looking stationary as it makes the HMM more consistent
    
    ## Build and fit model
    if gaussianHMM:
        model = hmm.GaussianHMM(n_components=hidden_states, covariance_type=cov_type,
                                n_iter=100, random_state=42, 
                                verbose=False, algorithm=algorithm)
    else:
        model = hmm.GMMHMM(n_components=hidden_states, covariance_type=cov_type,
                                n_iter=100, random_state=42, 
                                verbose=False, algorithm=algorithm)
        
    if train_test_split:
        x_train = (x_train - np.nanmean(x_train))/np.nanstd(x_train,ddof=0)
        x_train = np.nan_to_num(x_train)
        X_test = (X_test - np.nanmean(X_test))/np.nanstd(X_test,ddof=0)
        # X_test = np.nan_to_num(X_test)
        model.fit(x_train)
            
        ## Predict the hidden states corresponding to the observed values
        score, Z = model.decode(x_train)
        states = pd.unique(Z)
        
        if verbose:
            print("\nLog Probability & States:")
            print(score, states)
            
            print("\nStarting probabilities:")
            print(model.startprob_.round(2))
            
            print("\nTransition matrix:")
            print(model.transmat_.round(3))
            
            print("\nGaussian distribution means:")
            print(model.means_.round(4))
            
            print("\nGaussian distribution covariances:")
            print(model.covars_.round(4))

        if plot:
            if start is None:
                start = X_test.index[0]
                start_label = str(X_train.index[-1]).split(" ")[0]
            else:
                start_label = start
            x_test = X_test.loc[start:].values.reshape(-1,1)
            score_test, Z_test = model.decode(x_test)
            # Plot the price chart
            plt.figure(figsize=(15, 8))
            subplots = hidden_states + 2
            colors = ['r', 'g', 'b']
            plt.subplot(subplots, 1, 1)
            for i in states:
                want = (Z == i)
                try:
                    price = tmpcol.loc[:X_train.index[-1]]
                    if price.shape[0] != len(want):
                        raise ValueError('break')
                except:
                    price = tmpcol.loc[:X_train.index[-1]].iloc[:-1]
                x = price[want].index
                y = price[want]
                plt.plot(x, y, '.', label=f'State: {i+1}', c=colors[i])
            plt.title(f'{str(col[0])} up to {str(X_train.index[-1]).split(" ")[0]}')
            plt.legend()
            # Plot the smoothed marginal probabilities
            plt.subplot(subplots, 1, 2)
            for i in range(hidden_states):
                state_probs = pd.Series(model.predict_proba(x_train)[:, i], index=X_train.loc[:start].index)
                state_probs_smooth = state_probs.ewm(alpha=1-d).mean()
                plt.plot(state_probs_smooth, label=f'State {i+1}', alpha=0.5, c=colors[i])
            plt.legend()
            plt.title('Smoothed Marginal Probabilities (Train)')
            # Plot the smoothed marginal probabilities for x_test
            plt.subplot(subplots, 1, 3)
            for i in states:
                want = (Z_test == i)
                price = tmpcol[1:].loc[start:]
                x = price[want].index
                y = price[want]
                plt.plot(x, y, '.', label=f'State: {i+1}', c=colors[i])
            plt.title(f'{str(col[0])} from {start_label} onwards')
            plt.legend()
            plt.subplot(subplots, 1, 4)
            for i in range(hidden_states):
                state_probs_test = pd.Series(model.predict_proba(x_test)[:, i], index=X_test.loc[start:].index)
                state_probs_smooth = state_probs_test.ewm(alpha=1-d).mean()
                plt.plot(state_probs_smooth, label=f'State {i+1}', c=colors[i])
            plt.legend()
            plt.title('Smoothed Marginal Probabilities (Test)')
            
            plt.tight_layout()
            plt.show()
    
    else:
        tmpret = (tmpret - np.nanmean(tmpret))/np.nanstd(tmpret,ddof=0)
        model.fit(tmpret)
        ## Predict the hidden states corresponding to the observed values
        score, Z = model.decode(tmpret)
        states = pd.unique(Z)
        
        if verbose:
            print("\nLog Probability & States:")
            print(score, states)
            
            print("\nStarting probabilities:")
            print(model.startprob_.round(2))
            
            print("\nTransition matrix:")
            print(model.transmat_.round(3))
            
            print("\nGaussian distribution means:")
            print(model.means_.round(4))
            
            print("\nGaussian distribution covariances:")
            print(model.covars_.round(4))
        
        if plot:
            if start is None:
                start = tmpret.index[0]
                start_label = str(tmpret.index[-1]).split(" ")[0]
            else:
                start_label = start
            tmpret_ = tmpret.loc[start:].values.reshape(-1,1)
            score_test, Z_test = model.decode(tmpret_)
            # Plot the price chart
            plt.figure(figsize=(15, 8))
            subplots = 2
            colors = ['r', 'g', 'b']
            plt.subplot(subplots, 1, 1)
            for i in states:
                want = (Z == i)
                try:
                    price = tmpcol.loc[:tmpret.index[-1]]
                    if price.shape[0] != len(want):
                        raise ValueError('break')
                except:
                    price = tmpcol.loc[:tmpret.index[-1]].iloc[:-1]
                x = price[want].index
                y = price[want]
                plt.plot(x, y, '.', label=f'State: {i+1}', c=colors[i])
            plt.title(f'{str(col[0])} up to {str(tmpret.index[-1]).split(" ")[0]}')
            plt.legend()
            # Plot the smoothed marginal probabilities
            plt.subplot(subplots, 1, 2)
            for i in range(hidden_states):
                state_probs = pd.Series(model.predict_proba(tmpret)[:, i], index=tmpret.index)
                state_probs_smooth = state_probs.ewm(alpha=1-d).mean()
                plt.plot(state_probs_smooth, label=f'State {i+1}', alpha=0.5, c=colors[i])
            plt.legend()
            plt.title('Smoothed Marginal Probabilities')
            plt.tight_layout()
            plt.show()
    
    return model

col = 'rrp'
tmpdf = np.log(df[[col]]).dropna()

#### Build model
# model = hiddenMarkovModel(tmpdf, col, hidden_states=hidden_states, d=d, start=start, gaussianHMM=gaussianHMM, train_test_split=train_test_split, remove_direction_and_mean=remove_direction_and_mean, verbose=verbose, algorithm=algorithm, cov_type=cov_type)
from sklearn.preprocessing import StandardScaler

## Choose reponse and predictor
cols = ['TOTALDEMAND', 'rrp']
start_0 = '2020-01-01'
tmpdf = np.log(df[cols]).dropna().loc[start_0:]
tmpdf = tmpdf[(tmpdf != 0) & (tmpdf != -np.inf)].dropna() ## fails standard scaler otherwise
interaction_feature = tmpdf[cols[0]] * tmpdf[cols[1]]
interaction_feature.plot()

## Scale data for model
scaled = StandardScaler().set_output(transform='pandas').fit_transform(interaction_feature.values.reshape(-1,1))
scaled.index = interaction_feature.index
scaled.plot()

## Train/ Test split
days = 7
X_test = scaled.iloc[-days*48:]
X_train = scaled.iloc[:-days*48]

# k_regimes, endog, exog = 2, tmpdf_scaled[cols[0]], tmpdf_scaled[cols[1]]
# k_regimes, endog, exog = 2, tmpdf_scaled[cols[0]].rolling(48).mean().dropna(), tmpdf_scaled[cols[1]].rolling(48).mean().dropna()
X_train, X_test = KalmanFilterAverage(X_train), KalmanFilterAverage(X_train)
x_train, x_test = X_train.values.reshape(-1,1), X_test.values.reshape(-1,1)

train_test_split, start = True, None
remove_direction_and_mean, verbose = False, True
gaussianHMM, hidden_states, d = True, 2, 0.94
algorithm, cov_type = 'viterbi', 'diag'

model = hmm.GMMHMM(n_components=hidden_states, covariance_type=cov_type,
                        n_iter=100, random_state=42, 
                        verbose=False, algorithm=algorithm)

model.fit(x_train)
## Predict the hidden states corresponding to the observed values
score, Z = model.decode(x_train)
states = pd.unique(Z)

if verbose:
    print("\nLog Probability & States:")
    print(score, states)
    
    print("\nStarting probabilities:")
    print(model.startprob_.round(2))
    
    print("\nTransition matrix:")
    print(model.transmat_.round(3))
    
    print("\nGaussian distribution means:")
    print(model.means_.round(4))
    
    print("\nGaussian distribution covariances:")
    print(model.covars_.round(4))

def plot_hmm(): 
    # if start is None:
    start = X_test.index[0]
    start_label = str(X_train.index[-1]).split(" ")[0]
    # else:
    #     start_label = start
    x_test = X_test.loc[start:].values.reshape(-1,1)
    score_test, Z_test = model.decode(x_test)
    # Plot the price chart
    plt.figure(figsize=(15, 8))
    subplots = hidden_states + 2
    colors = ['r', 'g', 'b']
    plt.subplot(subplots, 1, 1)
    for i in states:
        want = (Z == i)
        try:
            price = tmpdf[cols[0]].loc[:X_train.index[-1]]
            if price.shape[0] != len(want):
                raise ValueError('break')
        except:
            price = tmpdf[cols[0]].loc[:X_train.index[-1]].iloc[:-1]
        x = price[want].index
        y = price[want]
        plt.plot(x, y, '.', label=f'State: {i+1}', c=colors[i])
    plt.title(f'Up to {str(X_train.index[-1]).split(" ")[0]}')
    plt.legend()
    # Plot the smoothed marginal probabilities
    plt.subplot(subplots, 1, 2)
    for i in range(hidden_states):
        state_probs = pd.Series(model.predict_proba(x_train)[:, i], index=X_train.loc[:start].index)
        state_probs_smooth = state_probs.ewm(alpha=1-d).mean()
        plt.plot(state_probs_smooth, label=f'State {i+1}', alpha=0.5, c=colors[i])
    plt.legend()
    plt.title('Smoothed Marginal Probabilities (Train)')
    # Plot the smoothed marginal probabilities for x_test
    plt.subplot(subplots, 1, 3)
    for i in states:
        want = (Z_test == i)
        price = tmpdf[cols[0]][1:].loc[start:]
        x = price[want].index
        y = price[want]
        plt.plot(x, y, '.', label=f'State: {i+1}', c=colors[i])
    plt.title(f'From {start_label} onwards')
    plt.legend()
    plt.subplot(subplots, 1, 4)
    for i in range(hidden_states):
        state_probs_test = pd.Series(model.predict_proba(x_test)[:, i], index=X_test.loc[start:].index)
        state_probs_smooth = state_probs_test.ewm(alpha=1-d).mean()
        plt.plot(state_probs_smooth, label=f'State {i+1}', c=colors[i])
    plt.legend()
    plt.title('Smoothed Marginal Probabilities (Test)')

    plt.tight_layout()
    plt.show()

plot_hmm()