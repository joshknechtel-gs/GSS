'''
Basic utilities. Simple functions used often.
'''

#from IPython.core.debugger import set_trace      # Uncomment for debugging and set_trace() at the troublespot

import numpy as np
import pandas as pd
from scipy import stats
#import plotly
#import plotly.graph_objs as goankIC
#import plotly.express as px
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from cycler import cycler

AnnFactor = {'Daily':252, 'Weekly':52, 'Monthly':12, 'Annual':1}


SellOffLabels = [
    '2000 Tech bubble bursts',
    '2006 Hawkish Fed fears',
    '2007 Quant crisis',
    '2008 GFC',
    '2010 Euro debt crisis',
    '2010 Flash Crash',
    '2011 US Downgrade/Euro scare',
    '2012 Growth scare',
    '2015/16 China crisis fears',
    '2018 Fed hawkish surprise',
    'COVID-19']

# this will include the beginning month but that isn't how Matt did it in Excel

SellOffs = [ 
     ('2000-4-30','2002-10-9'),
     ('2006-5-31','2006-5-31'),
     ('2007-7-30','2007-7-31'),
     ('2007-11-30','2009-2-28'),
     ('2010-1-31','2010-1-31'),
     ('2010-4-30','2010-5-31'),
     ('2011-5-31','2011-9-30'),
     ('2012-4-30','2012-5-31'),
     ('2015-11-30','2016-2-28'),
     ('2018-9-30','2018-12-31'),
     ('2020-2-28','2020-3-31')           
     ]
SellOffDict = dict(zip(SellOffLabels,SellOffs))

# older regime bits
DateCutBins = pd.to_datetime(['19950101','20000324','20071009','20090309'])
DateCutLabels = ["Internet Bubble","Bubble to GFC","GFC","Post GFC"]

#########################################################
def CumRet(x):
    '''
    Computes the cumulative return for a vector of returns.

    Argument: vector of periodic returns

    Return Value: scalar, total cumulative return for all periods
    '''
    return (np.prod(1+x)-1)
#########################################################
def CumRetVec(x):
    '''
    Computes the cumulative return each period for a vector of returns.

    Argument: vector of periodic returns

    Return Value: vector of cumulative return at the end of each period
    '''    
    return (np.cumprod(1+x)-1)

#########################################################
def shiftReturn (indf, Date="Date", Return="Return"):
    '''
    Shifts returns in a vector down one period.

    Arguments: 
        indf: input DataFrame
        Date: Date column of the input DataFrame, needed for the ascending sort by Date
        Return: column whose values will be shifted. Last value in the column will become NaN

    Return Value: input DataFrame with a Return column shifted
    '''    
    indf = indf.sort_values(by=Date)
    indf[Return] = indf[Return].shift(-1)
    return(indf)
#########################################################
def RetStats(r,freq="Monthly"):
    '''
    Calculates return statistics
    Arguments:
        r: input vector of returns
        freq: annualization factor {"Daily":252, "Weekly": 52, "Monthly":12, "Annual":1}
             Default is "Monthly"
    
    Return value: pandas series with annualized return, volatility, Sharpe Ratio, Sortino Ratio, skewness, 
                  kurtosis, the maximum drawdown, worst and best return
    
    Notes: (1) Sharpe and Sortino ratios do not (currently) subtract risk-free rate in the numerator
           (2) Maximum Drawdown is expressed as a percent of the peak value: (Peak - Trough)/Peak    
    '''
    AnnFactor = {'Daily':252, 'Weekly':52, 'Monthly':12, 'Annual':1}
    QFactor = {'Daily':63, 'Weekly':13, 'Monthly':3}

    a = AnnFactor[freq]
    q = QFactor[freq]
    
    r = r[np.isfinite(r)]

    fiveyear = (1+r).rolling(window=5*a,min_periods=5*a).agg(lambda x : x.prod()) - 1
    quarter = (1+r).rolling(window=q,min_periods=q).agg(lambda x : x.prod()) - 1
    quarter = quarter[quarter<0]
    
    AnnRet = (1 + CumRet(r))**(a/len(r)) -1
    AnnVol = np.sqrt(a)*np.std(r)
    DownsideDev = np.sqrt(a)*np.std(r[r<0])
    SR = AnnRet / AnnVol
    Sortino = AnnRet / DownsideDev
    skew = stats.skew(r)
    kurtosis = stats.kurtosis(r)
    
    NAV = 1+CumRetVec(r)                 # Net asset value with initial investment=1
    RM = np.maximum.accumulate(NAV)      # Running max
    DD = (RM-NAV)/RM                      # Percent drawdown at point in time
    maxDD = np.max(DD)
    maxDDstd = maxDD/AnnVol
    
    PlusMonths = r[r>0].count()/r.count() 
    
    s1 = pd.Series([AnnRet,AnnVol,maxDD,-quarter.min(),-quarter.mean()],\
                              index=["AnnRet","AnnVol", "MaxDD",'MaxQDD','AvgQDD'])\
                              .apply(lambda x: '{:.2%}'.format(x))
    s2 = pd.Series([maxDDstd],\
                   index=["MaxDD/Vol"]).apply(lambda x: '{:.2}'.format(x))
    s3 = pd.Series([r.min(),r.max(),fiveyear.min(),fiveyear.max()],\
                              index=["Worst Return","Best Return","Worst 5y Return","Best 5y Return"])\
                              .apply(lambda x: '{:.2%}'.format(x))
    s4 = pd.Series([skew,kurtosis,SR,Sortino,PlusMonths],\
                   index=["Skewness", "Kurtosis","Sharpe Ratio","Sortino Ratio","PositivePeriods Ratio"])\
                  .apply(lambda x: '{:.2}'.format(x))
    RetStats = pd.concat([s1,s2,s3,s4])  # Returning a DataFrame was causing issues when this function was called with apply
    return(RetStats)

#########################################################
def DateCut(dvec,bins=DateCutBins,labels=DateCutLabels):
    '''
    Wrapper around pandas.cut to return preset labels for a date vector
    Arguments:
    dvec: vector of dates
    bins: bins argument of pandas.cut
          Defaults to pd.to_datetime(['19950101','20000324','20071009','20090309', max(dvec)])
    labels: labels argument of pandas.cut
            Defaults to ["Internet Bubble","Bubble to GFC","GFC","Post GFC"]
    Return value: pandas Series with labels
    
    '''
    nlabels = len(labels)
    nbins = len(bins)
    if nlabels == nbins: bins = bins.append(pd.Index([dvec.max()])) 
    return(pd.cut(dvec,bins,labels=labels))
#########################################################
def RunningMax(x):
    '''
    Function for tracking drawdown starts along with magnitudes.
    Arguments:
    x: pandas Series
    Return value: pandas DataFrame with original Series and
                  columns 'RunningMax' (value of largest x seen so far)
                  and 'MaxStart' (index of first instance of RunningMax)
    
    '''
    rmax = np.full_like(x,np.NaN)
    maxindex = np.full_like(x.index,np.nan)
    rmax[0],maxindex[0] = x[0],x.index[0]
    for i in range(1,len(x)):
        if (x[i] > rmax[i-1]):
            rmax[i],maxindex[i] = x[i],x.index[i]
        else:
            rmax[i],maxindex[i] = rmax[i-1],maxindex[i-1]
    RMdf = pd.DataFrame(x)
    RMdf["RunningMax"] = rmax
    RMdf["MaxStart"] = maxindex
    return(RMdf)
#########################################################
def DrawDownRecoverAnalysis(df):
    '''
    look at HudsonAA for examples on how I use these
    still need to consolidate into one function
    '''
    df['ret'] = df/df.loc[df.first_valid_index()]
    df['modMax'] = df.ret.cummax()
    df['modDD'] = 1-df.ret.div(df['modMax'])
    groups = df.groupby(df['modMax'])
    dd = groups[['modMax','modDD']].apply(lambda g: g[g['modDD'] == g['modDD'].max()])
    top10dd = dd[dd['modDD']>0].sort_values('modDD', ascending=False)
    return top10dd
#########################################################
def DrawDownGroup(df,index_list):
    '''
    look at HudsonAA for examples on how I use these
    '''
    group_max,dd_date = index_list
    ddGroup = df[df['modMax'] == group_max]
    group_length = len(ddGroup)
    group_dd = ddGroup['modDD'].max()
    group_dd_length = len(ddGroup[ddGroup.index <= dd_date])
    group_start = ddGroup[0:1].index[0]
    group_end = ddGroup.tail(1).index[0]
    group_rec = group_length - group_dd_length
    #print (group_start,group_end,group_dd,dd_date,group_dd_length,group_rec,group_length)
    return group_start,group_end,group_max,group_dd,dd_date,group_dd_length,group_rec,group_length
#########################################################
# DividendPayers has good examples on use
def StandardLinePlot(graph_data,plot_title,include_lines="",win=120,save=False):
    '''
    Standardized Line Plot with "onbrand" color scheme/theme
    Arguments:
        graph_data: needs to be a DF currently
        plot_title: string
        save: optional, names the file the plot_title
    Returns:
        fig, ax: for further customization if needed
    '''
    fig, ax = plt.subplots(figsize=(20,10))
    parameters = {'xtick.labelsize': 20,'ytick.labelsize': 20,'figure.titlesize': 25}
    plt.rcParams.update(parameters)
    
    if include_lines == 'hist':
        graph_data['Historical average'] = graph_data.expanding(min_periods=12).mean()
        # '#4682b4': steelblue  ,'#b0c4de': lightsteelblue ,'#87cefa':lightskyblue
        # '#708090': slategrey  ,'#778899': lightslategrey ,'#d3d3d3':lightgrey
        ax.set_prop_cycle(cycler('color',['#4682b4','#b0c4de','#87cefa','#708090','#778899']))
    elif include_lines == 'move':
        graph_data[[graph_data.columns[0]]]
        graph_data['Moving {} average'.format(win)] = graph_data[[graph_data.columns[0]]].rolling(win).mean()
        graph_data['-std'] = graph_data[[graph_data.columns[0]]].rolling(win).mean()-graph_data[[graph_data.columns[0]]].rolling(win).std()
        graph_data['+std'] = graph_data[[graph_data.columns[0]]].rolling(win).mean()+graph_data[[graph_data.columns[0]]].rolling(win).std()
        ax.set_prop_cycle(cycler('color',['#4682b4','#b0c4de','#b0c4de','#b0c4de']))
    else:
        ax.set_prop_cycle(cycler('color',['#4682b4','#b0c4de','#87cefa','#708090','#778899']))
    
    #ax.set_prop_cycle(cycler('color',['#4682b4','#b0c4de','#87cefa']))
    ax.plot(graph_data,linewidth=4)
    plt.grid()
    plt.title(plot_title,fontsize=25)
    try:
        plt.legend(graph_data.columns.values) 
    except:
        print("not a dataframe")
    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig(plot_title+'.png',format="png")
    return fig, ax
#########################################################    
def SellOffAnalysis(rets):
    '''
    Calculates performance during well known equity sell offs
    Arguments:
        rets: dataframe of returns
    Returns:
        SOdf: dataframe table of sell off performance 
    '''
    SOdf = pd.DataFrame(np.nan,columns = ["Performance"],index = list(SellOffDict.keys()))
    for key,value in SellOffDict.items():
        sm = value[0]
        em = value[1]
        #perf = np.prod(1+Port_ret[value[0]:value[1]])-1
        #print(key, ": ", perf)
        SOdf.loc[key] = np.prod(1+rets.loc[value[0]:value[1]])-1
        #print(key, ": ", perf)
    return SOdf
#########################################################  
def get_quandl_data(incodes,outcodes):   #,resample=""
    apiKey="BRvb9YQ9jG9AzxSxCVTo"
    
    # first call it to pass in the codes for the monthly
    qdata = quandl.get(incodes,api_key=apiKey) 
    qdata.rename(columns=dict(zip(qdata.columns.values,outcodes)),inplace=True)
    # this seems so strange but it is because quandl returns SM & BM irregularly
    qdata = (qdata.resample('M').last()).resample('BM').bfill()
    #if resample != "":
        #qdata = qdata.resample('BQ').last()
    
    # join the quarterly codes
    QRatios = ['MULTPL/SP500_BVPS_QUARTER','MULTPL/SP500_PBV_RATIO_QUARTER','MULTPL/SP500_SALES_QUARTER',
          'MULTPL/SP500_PSR_QUARTER']
    our_codes_q = ['spbvps','sppb','spsales','spps']
    qdata2 = quandl.get(QRatios,api_key=apiKey) 
    qdata2.rename(columns=dict(zip(qdata2.columns.values,our_codes_q)),inplace=True)
    qdata2 = (qdata2.resample('M').last()).resample('BM').ffill()
    qdata2 = qdata2.ffill(limit=6)
    qdata = qdata.join(qdata2)
    
    # defined measures 
    qdata['payout'] = qdata['spdy'].div(qdata['spey'])
    
    return qdata
#########################################################  
def ewlinTrend(df):
    date_ordinal = pd.to_datetime(df.index).map(dt.datetime.toordinal)
    linTrend = np.polyval(np.polyfit(date_ordinal, df, 1), date_ordinal)
    return linTrend[-1]
#########################################################  
def ewlogTrend(df):
    date_ordinal = pd.to_datetime(df.index).map(dt.datetime.toordinal)
    linTrend = np.exp(np.polyval(np.polyfit(date_ordinal, np.log(df), 1), date_ordinal))
    return linTrend[-1]
#########################################################  
def logTrend(df):
    date_ordinal = pd.to_datetime(df.index).map(dt.datetime.toordinal)
    linTrend = np.exp(np.polyval(np.polyfit(date_ordinal, np.log(df), 1), date_ordinal))
    return linTrend[-1]
#########################################################  
def MargContrRisk(Cov,w):
    port_risk = np.std(w.T*Cov*w)
    mcr = w.T*Cov/port_risk
    return mcr
#########################################################  
def ContrRisk(Cov,w):
    mcr = MargContrRisk(Cov,w)
    cr = w.T.dot(mcr)
    return cr
#########################################################  
def PercentContrRisk(Cov,w):
    mcr = MargContrRisk(Cov,w)
    cr = w.T.dot(mcr)
    return cr
#########################################################  
def SectorByFactorCharts(file,fund):
    path = "Z:/Shared/Risk Management and Investment Technology/Equity Risk Modeling/"
    #file = 'ONGSINTL Sector by Factor 03-17-22.xlsx'
    sxf_df = pd.read_excel(path+file,skiprows=3,skipfooter=3)

    for sector in sxf_df['Filter Level 2'].unique():

        if str(sector) == 'nan':
            continue

        mask = (sxf_df['Filter Level 2']==sector)\
                                  &(sxf_df['Level']==0)&(sxf_df['Filter Level 3']=='STYLE')

        labels = sxf_df.loc[mask,' Title'].tolist()
        factor_exp = sxf_df.loc[mask,'Active']
        risk_contr = sxf_df.loc[mask,'Active (bp).1']

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(20,10))
        ax2 = ax.twinx() 

        if fund != 'GSS':
            rects1 = ax.bar(x - width/2, factor_exp, width, label='Active Factor Exposure')
            rects2 = ax2.bar(x + width/2, risk_contr, width, label='Active Risk Contribution (bps right)', color='red',)
        else:
            rects1 = ax.bar(x - width/2, factor_exp, width, label='Factor Exposure')
            rects2 = ax2.bar(x + width/2, risk_contr, width, label='Risk Contribution (bps right)', color='red',)

        ax.set_title(fund+': Style Factors within ' +sector,fontsize=25);

        ax.set_xticks(x);
        ax.set_xticklabels(labels, rotation='vertical');
        #ax.legend();
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

        ax.set_ylim(-factor_exp.abs().max()*1.1,factor_exp.abs().max()*1.1);
        ax2.set_ylim(-risk_contr.abs().max()*1.1,risk_contr.abs().max()*1.1);
        #ax.bar_label(rects1, padding=3)
        #ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        plt.show()

        fig.savefig(path+fund+'StyleFactors_' +sector+'.png',format="png")