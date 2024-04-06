import pandas as pd
import numpy as np
import statsmodels.api as sm
# from datetime import datetime
import seaborn as sns
import sys
sys.path.append("..")
from plot_settings import *

def markov_regime_switching(endog, k_regimes, switching_variance=True, search_reps=0, print_summary=False, exog=None):
    if not isinstance(endog, pd.DataFrame):
        endog = pd.DataFrame(endog)
    # if endog.columns[0] == 'MSCI_US':
    #     model_msci = sm.tsa.MarkovRegression(endog=endog.dropna(), k_regimes=k_regimes, trend='c', switching_variance=switching_variance, exog=exog)
    # else:
    model_msci = sm.tsa.MarkovRegression(endog=endog.dropna(), k_regimes=k_regimes, trend='c', switching_variance=switching_variance, exog=None)
    
    res_msci = model_msci.fit(search_reps=search_reps)
    # res_msci.smoothed_marginal_probabilities
    if print_summary:
        print(res_msci.summary())
    return res_msci

def loop_rolling_realised_vol(returns, years, k_regimes=2, switching_variance=True, search_reps=0, plot=False, print_summary=False, exog=None):
    rolling_real_vol = (returns.rolling(window=years*12).std() * np.sqrt(12))
    
    results = {}
    for regime in rolling_real_vol.columns:
        if regime == 'MSCI_US':
            res_model = markov_regime_switching(rolling_real_vol[regime], k_regimes=k_regimes, print_summary=print_summary, switching_variance=switching_variance, search_reps=search_reps, exog=exog)
        else:
            res_model = markov_regime_switching(rolling_real_vol[regime], k_regimes=k_regimes, print_summary=print_summary, switching_variance=switching_variance, search_reps=search_reps)
        
        if plot:
            plot_regimes(rolling_real_vol[regime], res_model, title=f'{years}y rolling realised volatility')
        results[regime] = res_model
    return rolling_real_vol, results

def plot_regimes(endog, model_res, title=None, prob_ind=0, underlying=None):
    if not isinstance(endog, pd.DataFrame):
        endog = pd.DataFrame(endog)
    ## From the summary we can see the two regimes (const - mean, sigma2 - variance) # as well as transition probabilities.
    # Plot the spread price along with the probabilites of high-mean regime 
    fig, axs = plt.subplots(2,1,figsize=(18,8))
    fig.subplots_adjust(hspace=0.75)
    # if underlying is not None:
    #     underlying.plot(ax=axs[0], label=underlying.columns[0])
    endog.dropna().plot(ax=axs[0])
    axs[0].axhline(y=model_res.params['const[0]'], label=r'$\mu_0$', linestyle='dotted', c='r') 
    axs[0].axhline(y=model_res.params['const[1]'], label=r'$\mu_1$', linestyle='dotted', c='m') 
    if title is not None:
        axs[0].set_title(f'{endog.columns[0]} {title}')
    else:
        axs[0].set_title(f'{endog.columns[0]}')
    axs[0].legend()
    model_res.smoothed_marginal_probabilities[prob_ind].plot(ax=axs[1])
    
    # if endog.columns[0] == 'MSCI_US':
    #     if prob_ind==1:
    #         axs[1].set_title('Probability of high-mean regime')
    #     else:
    #         axs[1].set_title('Probability of low-mean regime')
    # else:
    #     if prob_ind==1:
    #         axs[1].set_title('Probability of high-mean regime')
    #     else:
    #         axs[1].set_title('Probability of low-mean regime')
            
    if prob_ind==0:
        axs[1].set_title('Probability of low-mean regime')
    else:
        axs[1].set_title('Probability of high-mean regime')

def plot_regime_switching_macro(endog, results_df, upper_label_years='#', prob_ind=0):
    if not isinstance(endog, pd.DataFrame):
        endog = pd.DataFrame(endog)
    fig, axs = plt.subplots(2, 1, figsize=(18, 8))
    fig.subplots_adjust(hspace=0.5)
    
    main = endog.columns[0]
    endog.dropna(inplace=True)

    ## First subplot
    endog.dropna().plot(ax=axs[0], label='Vol')
    axs[0].axhline(y=results_df[main].params['const[0]'], label=r'$\mu_0$', linestyle='dotted', c='r')
    axs[0].axhline(y=results_df[main].params['const[1]'], label=r'$\mu_1$', linestyle='dotted', c='m')
    axs[0].set_title(f'{main} {upper_label_years}y rolling realised volatility with regime means')

    # Create a twin y-axis for the top subplot
    # axs2 = axs[0].twinx()
    # axs2.set_ylabel('Equity Vol Regime Prob')

    results_df[main].smoothed_marginal_probabilities[prob_ind].loc[endog.index[0]:].plot(ax=axs[1], label='Equity Vol Regime Prob', linewidth=2, c='y')
    # Fill the area under the 'High Equity Vol Regime' line
    axs[1].fill_between(results_df[main].smoothed_marginal_probabilities[prob_ind].loc[endog.index[0]:].index,
                    results_df[main].smoothed_marginal_probabilities[prob_ind].loc[endog.index[0]:],
                    color='yellow', alpha=0.3, hatch='//')

    # Set the legend for the top subplot
    axs[0].legend(loc='upper left')

    # if prob_ind==0:
    #     axs[0].set_title('Probability of high-mean regime')
    # else:
    #     axs[0].set_title('Probability of low-mean regime')

    ## Second subplot
    results_df['CPI'].smoothed_marginal_probabilities[prob_ind].loc[endog.index[0]:].plot(ax=axs[1], label='CPI Vol Regime Prob', linestyle='--', linewidth=4, c='g')
    results_df['GDP'].smoothed_marginal_probabilities[prob_ind].loc[endog.index[0]:].plot(ax=axs[1], label='GDP Vol Regime Prob', linewidth=2, c='r')
    results_df['Deflator'].smoothed_marginal_probabilities[prob_ind].loc[endog.index[0]:].plot(ax=axs[1], label='Deflator Vol Regime Prob', linestyle='-.', linewidth=2, c='cyan')
    
    # Set the legend for the second subplot below the plot
    axs[1].legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.15))

    if prob_ind==0:
        axs[1].set_title('Probability of low-mean regime')
    else:
        axs[1].set_title('Probability of high-mean regime')

    plt.tight_layout()
    # plt.show()

def _calculate_metrics(cumret, plot_rets=False, window_monthly=3, annualiser=252):
    '''
    calculate performance metrics from cumulative returns
    
    results_df = pd.DataFrame(columns=['Total return', 'APR', 'Ann Vol', 'Sharpe', 'maxDD', 'maxDDD', 'Skewness', 'Kurtosis', 'Vol of ex-ante vol'])
    '''
    total_return = (cumret[-1] - cumret[0])/cumret[0]
    apr = (1+total_return)**(annualiser/len(cumret)) - 1
    rets = pd.DataFrame(cumret).pct_change()[1:]
    ann_vol = np.nanstd(rets,ddof=0) * np.sqrt(annualiser)
    if len(rets) < 12:
        sharpe = np.nan
    else:
        sharpe = np.sqrt(annualiser) * np.nanmean(rets) / np.nanstd(rets,ddof=0)
    skewness = rets.skew()[0]
    kurtosis = rets.kurtosis()[0]
    
    # ## Vol of ex ante volatility
    # rolling_cov = rets.rolling(window=window_monthly*12).cov()
    # volatility_ex_ante = np.sqrt(rolling_cov)
    # vol_of_vol = (volatility_ex_ante.std() * 100)[0]
    # # ex_ante['Vol of Ex Ante Volatility %'] = [round(x,4) for x in vol_of_vol]
    
    # maxdd and maxddd
    highwatermark=np.zeros(cumret.shape)
    drawdown=np.zeros(cumret.shape)
    drawdownduration=np.zeros(cumret.shape)
    for t in np.arange(1, cumret.shape[0]):
        highwatermark[t]=np.maximum(highwatermark[t-1], cumret[t])
        drawdown[t]=cumret[t]/highwatermark[t]-1
        if drawdown[t]==0:
            drawdownduration[t]=0
        else:
            drawdownduration[t]=drawdownduration[t-1]+1
    maxDD=np.min(drawdown)
    maxDDD=np.max(drawdownduration)
    
    if plot_rets:
        from seaborn import histplot
        sns.histplot(rets, kde=True, color='blue', bins=20)
        plt.title('Returns Histogram')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    
    return total_return, apr, ann_vol, sharpe, maxDD, maxDDD, skewness, kurtosis
