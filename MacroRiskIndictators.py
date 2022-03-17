import numpy as np
import pandas as pd
import datetime as datetime
import pandas_datareader.data as web
#import quandl
import blpapi
from xbbg import blp

import matplotlib.pyplot as plt
from cycler import cycler

import seaborn as sns
import scipy
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance

import CLO_SQL_Updated as cs

import copy


file = 'C:/Users/jknechtel/Documents/Equity Risk Modeling/FactorTSraw_hist.xlsx'


index_list = ['MID Index','RTY Index','RLG Index','RLV Index', 'NDX Index',
              'UKX Index','NKY Index','SPTSX Index','DAX Index','CAC Index','MXEA Index',
              'MXEF Index','MXCN Index','IBOV Index','HSI Index',
              'FNER Index','LPX50TR Index',
              'S5CPGS Index','S5CONS Index','S5UTIL Index','S5SEQX Index',
              'LF98TRUU Index','LCANTRDU Index','I165686 Index','LUACTRUU Index','LD08TRUU Index',
              'H2173EU Index','CSLLLTOT Index','JPCAUS3M Index',
              'US0003M Index', 'USSOC Curncy','GB03 Govt','USSP2 BGN Curncy',   
              'GT02 Govt','GT05 Govt','GT10 Govt','USYC2Y10 Index','USYC5Y10 Index',
              'SPGSCI Index','CO1 Comdty','XAU Curncy','DXY Curncy','JPY Curncy','EUR Curncy','GBP Curncy','CHF Curncy']  #'USOSFR20 Curncy',

yields =  ['US0003M Index', 'USSOC Curncy','GB03 Govt','USSP2 BGN Curncy',
              'GT02 Govt','GT05 Govt','GT10 Govt','USYC2Y10 Index','USYC5Y10 Index']

VIX = ['VIX Index']
#########################################################
def get_Index_rets(start_date='1998-12-31',call = True):


    if call:
        index_ntr = blp.bdh(tickers=index_list, flds=['PX_LAST'],start_date='1998-12-31', end_date=pd.Timestamp.today())
        index_ntr.to_csv('Z:/Shared/Risk Management and Investment Technology/Macro & Market Indicators/IndexReturns.csv')
    else:
        index_ntr = pd.read_csv('Z:/Shared/Risk Management and Investment Technology/Macro & Market Indicators/IndexReturns.csv')    
    
    try:
        index_ntr = index_ntr.droplevel(1,axis=1)
    except:
        pass
    
    index_ntr.index = pd.to_datetime(index_ntr.index)
    index_wkly = index_ntr.resample('W-FRI').last()
    index_rets = index_wkly.pct_change()
    for c in yields:
        index_rets[c] = index_wkly[c].diff()

    index_rets.index.name = 'Dates'
    #index_rets.drop(columns=['USOSFR20 Curncy'],inplace=True)
    index_rets.dropna(inplace=True)
    index_rets.reset_index(inplace=True)
    
    return index_rets, index_wkly

#########################################################
def exponential_smoother(raw_data, half_life):
    """
    Purpose: performs exponential smoothing on "raw_data". Begins recursion
    with the first data item (i.e. assumes that data in "raw_data" is listed
    in chronological order).

    Output: a list containing the smoothed values of "raw_data".

    "raw_data": iterable, the data to be smoothed.

    "half_life": float, the half-life for the smoother. The smoothing factor
    (alpha) is calculated as alpha = 1 - exp(ln(0.5) / half_life)
    """
    import math

    raw_data = list(raw_data)
    half_life = float(half_life)

    smoothing_factor = 1 - math.exp(math.log(0.5) / half_life)
    smoothed_values = [raw_data[0]]
    for index in range(1, len(raw_data)):
        previous_smooth_value = smoothed_values[-1]
        new_unsmooth_value = raw_data[index]
        new_smooth_value = ((smoothing_factor * new_unsmooth_value)
            + ((1 - smoothing_factor) * previous_smooth_value))
        smoothed_values.append(new_smooth_value)

    return(smoothed_values)

#########################################################
def append_recession_series(current_date, dataframe):
    """
    Purpose: determine if the "current_date" is in a recessionary time period.
    Appends the result (100 if in recession, 0 if not) to the "Recession"
    series of the "dataframe".
    """
    recession_value = 0
    for recession_instance in range(0, len(recession_end_dates)):
        recession_start = recession_start_dates[recession_instance]
        recession_end = recession_end_dates[recession_instance]
        after_start_date = current_date >= recession_start
        before_end_date = current_date <= recession_end

        if after_start_date and before_end_date:
            recession_value = 100
            break
    dataframe['Recession'].append(recession_value)

#########################################################
def calculate_turbulence(returns, initial_window_size=250):
    """
    Purpose: calculate the Turbulence of the asset pool.

    Output: a dictionary containing the Turbulence values and their date-stamps.

    "returns": dataframe, the returns of the asset pool (in reverse chronological
    order)

    "initial window_size": integer, the initial window size used to calculate
    the covariance matrix. This window size grows as the analysis proceeds
    across time.
    """
    import copy
    import numpy as np
    from scipy.spatial import distance

    window_size = copy.deepcopy(int(initial_window_size)) 
    turbulence = {'Dates': [], 'Raw Turbulence': []}  #, 'Recession': []
    chronological_returns = returns #.iloc[::-1]
    while window_size < len(chronological_returns):
        historical_sample = chronological_returns.iloc[:window_size, 1:]
        historical_sample_means = historical_sample.mean()
        inverse_covariance_matrix = np.linalg.pinv(historical_sample.cov())
        current_data = chronological_returns.iloc[[window_size]]
        current_date = current_data['Dates'].values[0]

        mahalanobis_distance = distance.mahalanobis(u=current_data.iloc[:, 1:],
                                                    v=historical_sample_means,
                                                    VI=inverse_covariance_matrix)
        turb = mahalanobis_distance**2

        turbulence['Raw Turbulence'].append(turb)
        turbulence['Dates'].append(current_date)
        #append_recession_series(current_date=current_date,
        #                             dataframe=self.turbulence)
        window_size += 1

    #turbulence['Turbulence'] = exponential_smoother(raw_data=self.turbulence['Raw Turbulence'],
    #                                                          half_life=12)
    turbulence = pd.DataFrame(turbulence)
    turbulence['Turbulence'] = exponential_smoother(raw_data=turbulence['Raw Turbulence'], half_life=12)
    return(turbulence)

#########################################################
def gini(values):
    """
    Purpose: calculate the Gini coefficient for a one-dimensional set of values.

    Output: the Gini coefficient, as a float object.

    "values": iterable, the values on which to calculate the Gini coefficient.
    The data in "values" does not have to be sorted in ascending or descending order.
    """

    values = list(values)
    values.sort()

    minimum_value = values[0]
    lorenz_curve_value = minimum_value
    average_input = sum(values)/len(values)
    line_of_equality = [average_input]
    gap_area = [line_of_equality[0] - lorenz_curve_value]

    for index in range(1, len(values)):
        lorenz_curve_value += values[index]
        line_of_equality.append(line_of_equality[index - 1] + average_input)
        gap_area.append(line_of_equality[index - 1] + average_input
                        - lorenz_curve_value)

    return(sum(gap_area)/sum(line_of_equality))

#########################################################
def calculate_systemic_risk(returns, window_size=250):
    """
    Purpose: calculate the Systemic Risk of the asset pool.

    Output: a dictionary containing the Systemic Risk values and their date-stamps.

    "returns": dataframe, the returns of the asset pool (in reverse chronological
    order)

    "window_size": integer, the window size used to calculate
    the covariance matrix. This window size shifts forward as the analysis proceeds
    across time.
    """

    import numpy as np
    import copy

    systemic_risk = {'Dates': [], 'Systemic Risk': []}  #, 'Recession': []
    chronological_returns = returns #.iloc[::-1]

    window_endpoint = copy.deepcopy(int(window_size))
    while window_endpoint < len(chronological_returns):
        covariance_matrix = chronological_returns.iloc[window_endpoint + 1 - window_size : window_endpoint + 1, 1:].cov()
        eigenvalues = np.sort(np.linalg.eig(covariance_matrix)[0])
        current_date = chronological_returns.iloc[window_endpoint, 0]
        systemic_risk['Systemic Risk'].append(gini(values=eigenvalues))
        systemic_risk['Dates'].append(current_date)
        #append_recession_series(current_date=current_date,
        #                             dataframe=self.systemic_risk)
        window_endpoint += 1
    systemic_risk = pd.DataFrame(systemic_risk)
    return(systemic_risk)


#########################################################
def cusip2bb(cusips):
    
    bbtkrs = blp.bdp(tickers=cusips+" Equity", flds=["TICKER_AND_EXCH_CODE"])
    bbtkrs.reset_index(inplace=True)
    #cusip2bb_map = dict(zip(cusips,bbtkrs))
    bbtkrs_map = dict(zip(bbtkrs['index'],bbtkrs['ticker_and_exch_code']))
    cusips_call = cusips+" Equity"
    miss_keys = np.setdiff1d(cusips_call,list(bbtkrs_map.keys()))

    fx_keys = [s for s in miss_keys if "BEN" in s]
    cash_keys = [s for s in miss_keys if "CASH" in s]
    oth_keys = [s for s in miss_keys if "X" in s[0]]
    key_stocks = np.setdiff1d(miss_keys,oth_keys)
    key_stocks = np.setdiff1d(key_stocks,cash_keys)
    key_stocks = np.setdiff1d(key_stocks,fx_keys)
    codes = [i.strip(' Equity')[1:8] for i in key_stocks]
    codes = [i + ' Equity' for i in codes]
    bbtkrs_o = blp.bdp(tickers=codes, flds=["TICKER_AND_EXCH_CODE"])
    bbtkrs_o.reset_index(inplace=True)
    bbtkrs_omap = dict(zip(bbtkrs_o['index'],bbtkrs_o['ticker_and_exch_code']))

    bbtkrs_map = {**bbtkrs_map,**bbtkrs_omap}
    
    return bbtkrs_map, cash_keys, oth_keys, fx_keys

#########################################################
def SimpleRets(sers):
    '''
    Calculates simple returns at various frequencies for different indices
    Arguments:
        r: input vector of index levesl
      
    Return value: pandas series with simple returns on 1wk, 1mo, 3mo and 6mo basis
    

    '''
    oneweek = (sers/sers.shift(5)-1)*100
    onemonth = (sers/sers.shift(21)-1)*100
    threemonth = (sers/sers.shift(63)-1 )*100 
    sixmonth = (sers/sers.shift(126)-1)*100    

    owsers = oneweek.loc[oneweek.index[-1]]
    omsers = onemonth.loc[onemonth.index[-1]]
    tmsers = threemonth.loc[threemonth.index[-1]]
    smsers = sixmonth.loc[sixmonth.index[-1]]
    data = pd.concat([owsers,omsers,tmsers,smsers],axis=1)
    RetStats = pd.DataFrame(data.values,index=sers.columns,columns=["1 Week %","1 Month %", "3 Month %",'6 Month %'])#\
                              #.apply(lambda x: '{:.2%}'.format(x))
    
    
    return(RetStats)
#########################################################
def TurbPlot(graph_data,plot_title,include_lines="",win=120,save=False):
    '''
    Standardized Line Plot with "onbrand" color scheme/theme
    Arguments:
        graph_data: needs to be a DF currently
        plot_title: string
        save: optional, names the file the plot_title
    Returns:
        fig, ax: for further customization if needed
    '''
    fig, ax = plt.subplots(3,figsize=(20,10))
    parameters = {'xtick.labelsize': 20,'ytick.labelsize': 20,'figure.titlesize': 25}
    plt.rcParams.update(parameters)
    fig.suptitle(plot_title,fontsize=25)    
#    ax.set_prop_cycle(cycler('color',['#4682b4','#b0c4de','#87cefa','#708090','#778899']))
    
    #ax.set_prop_cycle(cycler('color',['#4682b4','#b0c4de','#87cefa']))
    ax[0].plot(graph_data[['Raw Turbulence','Turbulence']],linewidth=4)  # [['Raw Turbulence','Turbulence']]
    ax[0].legend(graph_data[['Raw Turbulence','Turbulence']].columns.values)
    plt.grid()


    ax[1].plot(graph_data[['VIX Index']],linewidth=4)  # [['Raw Turbulence','Turbulence']]
    ax[1].legend(graph_data[['VIX Index']].columns.values)
    plt.grid()
#    plt.title(plot_title,fontsize=25)

    #absorbRatio = calculate_systemic_risk(returns, window_size=250)
#    absorbRatio.set_index('Dates',inplace=True)
    ax[2].plot(graph_data[['Systemic Risk']],linewidth=4)  # [['Raw Turbulence','Turbulence']]
    ax[2].legend(graph_data[['Systemic Risk']].columns.values)
    plt.grid()
    #plt.title(plot_title,fontsize=25)

    
    #ax.secondary_yaxis(graph_data[['VIX Index']],linewidth=4)
    
#    try:
#        plt.legend(graph_data.columns.values) 
#    except:
#        print("not a dataframe")
    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig(plot_title+'.png',format="png")
    return fig, ax
#########################################################

def get_AlladinFactorSeries(file,writeOutput=False):
    usfinds = pd.read_excel(file,sheet_name='USAM',index_col=[0],parse_dates=['date'])
    cafinds = pd.read_excel(file,sheet_name='CAND',index_col=[0],parse_dates=['date'])
    wldfinds = pd.read_excel(file,sheet_name='WRLD',index_col=[0],parse_dates=['date'])
    
    if writeOutput:
        cs.insertTable(usfinds,'EQ_USFactorIndices')
        cs.insertTable(cafinds,'EQ_CAFactorIndices')
        cs.insertTable(wldfinds,'EQ_WLDFactorIndices')
        
    usfrets = usfinds.pct_change()
    cafrets = cafinds.pct_change()
    wldfrets = wldfinds.pct_change()
    
    return usfinds, cafinds, wldfinds, usfrets, cafrets, wldfrets
#########################################################
def plot_corr(df,size=10):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    
    #
    # Compute the correlation matrix for the received dataframe
    corr = df.corr()
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='RdBu')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)