'''
Signal evaluation utilities. Simple functions, primarily useful with pandas groupby and apply.
'''

#from IPython.core.debugger import set_trace      # Uncomment for debugging and set_trace() at the troublespot

import numpy as np
import pandas as pd
from scipy import stats
import plotly
import plotly.graph_objs as goankIC
import plotly.express as px
#import statsmodels.api as sm
#import statsmodels.formula.api as smf

AnnFactor = {'Daily':252, 'Weekly':52, 'Monthly':12, 'Annual':1}

Capbins = [0,100e6,5e9,15e9,100e9,100e12]
CapLabels = ["Micro Cap","Small Cap", "Mid Cap","Large Cap","Mega Cap"]

DateCutBins = pd.to_datetime(['19950101','20000324','20071009','20090309'])
DateCutLabels = ["Internet Bubble","Bubble to GFC","GFC","Post GFC"]
#########################################################
def KMB(x,precision=2):
    '''
    Converts large numbers to readable string with suffix
    K (thousands), M (millions), B(billions), or T(trillions)
    Arguments:
        x: number to be converted
        precision: number of decimal digits to retain
    Return Value: formatted string   
    '''
    s = f"{{:.{precision}f}}"
    if x < 1e6:
        d = 1e3
        suffix = " K"
    elif x < 1e9:
        d = 1e6
        suffix = " M"
    elif x < 1e12:
        d = 1e9
        suffix = " B"
    else:
        d = 1e12
        suffix = " T"
        
    return((s+suffix).format(x/d))
#########################################################
def DisplayBuckets(bins,labels):
    '''
    Displays buckets specified by bins
    Arguments:
        bins: boundaries of buckets
        labels: strings specifying labels for ranges. Length should be one less than number of bins.
    Return Value: data frame with label and upper and lower limits    
    '''    
    x = bins
    y = [(KMB(x[i],0),KMB(x[i+1],0)) for i in range(len(x)-1)]
    df = pd.DataFrame(dict(zip(labels,y)),index=["Lower Limit","Upper Limit"]).T
    return(df)
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
def quantiles(indf,signal,nq=10,flipSign=False):
    '''
    Assigns quantiles to each value in a signal.

    Arguments: 
        indf: input DataFrame
        signal: column of indf whose values are to be assigned to quantiles
        nq: number of quantiles. Defaults to 10, yielding deciles
        flipSign: bool. If True, 1 is the highest quantile, else it is the lowest.

    Return Value: input DataFrame with a column labeled "Quantile"
    '''    

    if (flipSign):
        x = -1*indf[signal]
    else: 
        x = indf[signal]
        
    indf["Quantile"] = pd.qcut(x,nq,labels=range(1,nq+1))
    return(indf)
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
def RankIC(indf,signal,flipSign=False,Date="Date",Return="Return"):
    '''
    Calculates the rank information coefficient (Spearman correlation)
    between a signal and subsequent period return
    
    Arguments:
        indf: input DataFrame
        signal: indf column that has signal values
        Date: Date column in the input frame
        Return: column with subsequent period returns
        
    Return Value: pandas Series with IC values, indexed by Date   
    '''
    if (flipSign):
        x = -1*indf[signal]
    else: 
        x = indf[signal]

    minRets = 6      # Minimum required returns for IC calculation
    indf = indf.loc[:,[Date,signal,Return]]
    CorrDF = indf.groupby(Date).corr(method="spearman",min_periods=minRets)
    ICVec =  CorrDF.loc[(slice(None),Return),signal].droplevel(1,axis=0)
    return(ICVec)
#########################################################
def QWRet(indf,weight="Equal",Return="Return",id="CUSIP"):
    '''
    Calculates the weighted return for a quantile portfolio
    Arguments:
        indf: input DataFrame
        weight: keyword specifying type of weighting scheme
        Return: column with subsequent period returns
        id: identifier column in input DataFrame (for retrieving MCAP if weight="MCAP"; NOT IMPELEMENTED YET)
        
    Return Value: pandas Series with return values, indexed by (Date, Quantile)   
    '''
    r = indf[Return]
    r = r[np.isfinite(r)]
    
    n = len(r)
    if (weight == "Equal"):
        weight = np.array([1/n for i in range(n)])
        
    # Code for market cap weighting goes here. id variable will be needed to retrieve MCAP
    
    if (sum(weight) != 1):
        weight = weight / sum(weight)         # Make sure weights add to 1
    
    wret = np.inner(weight,r)
    return(wret)
#########################################################
def QTurnOver(Qdf,Date="Date",id="cusip"):
    '''
    Calculates turnover in quantile constituents during each period
    Turnover = (Number added + Number removed)/(Initial count)
    Arguments:
        Qdf: input DataFrame with quantile constituents for each date
        Date: Date variable in DataFrame
        id: identifier column for constituents                
    Return Value: Dataframe with date,and turnover  
    '''
    dvec = np.sort(Qdf[Date].unique())
    TOvec = [np.NaN]
    for i,d in enumerate(dvec[1:],1):
        prevd = dvec[i-1]
        QSet = set(Qdf.loc[Qdf[Date]==d,id])
        QSetPrev = set(Qdf.loc[Qdf[Date]==prevd,id])
        NumStart = len(QSetPrev)
        NumAdded = len(QSet.difference(QSetPrev))
        NumRemoved = len(QSetPrev.difference(QSet))
        Turnover = (NumAdded + NumRemoved)/NumStart
        #print(QSet)
        TOvec.append(Turnover)
    return(pd.DataFrame({'date':dvec, 'Turnover': TOvec}))
#########################################################################
def QPorts(indf,signal, flipSign=False,Numq=10,Date="Date",Return="Return"):
    '''
    Computes quantile portfolio returns
    Arguments:
        indf: input DataFrame
        signal: name of column with signal values
        flipSign: boolean. If true,signal values are multiplied by -1
        Numq: number of quantiles
        Date: Date variable in DataFrame
        Return: column with subsequent period returns
                
    Return Value: List with quantile returns dataframe, quantile plot, long-short returns dataframe, LS plot   
    '''
    
    if (flipSign):
        x = -1*indf[signal]
    else: 
        x = indf[signal]

    indf = indf.dropna(subset=[signal])
    
    # Step 1. For each date, create quantiles based on signal values
    indf = indf.groupby(Date).apply(quantiles,signal,Numq,flipSign) 
    
    # Step 2. For each date and quantile, get weighted quantile portfolio returns

    QRets = indf.groupby([Date,"Quantile"]).apply(QWRet,Return=Return)
    # QRets.loc[(slice(None),1)]     # method to get all values of one level in multi-level index

    QRetsDF = pd.DataFrame(QRets,columns=["QRet"]).reset_index().sort_values(["Quantile",Date])
    
    # Step 3. Plot of quantile cumulative returns
    QCum = QRetsDF.set_index(Date).groupby("Quantile").apply(lambda x: CumRetVec(x["QRet"])).T
    if (isinstance(QCum,pd.Series)):
        QCum = QCum.reset_index().set_index(Date).pivot(columns="Quantile",values="QRet")

    QPlot = px.line(QCum,x=QCum.index,y=QCum.columns)
    QPlot.update_yaxes(title_text="Cumulative Return",tickformat='.1%')
    
    # Step 4. Long short return for each date
    LSRets = QRetsDF.loc[QRetsDF["Quantile"].isin([1,Numq]),:]
    LSRets = LSRets.pivot(index=Date,columns="Quantile",values="QRet")
    LSRets.columns = ["BottomQuantile","TopQuantile"]   
    LSRets["LSRet"] = LSRets["TopQuantile"]-LSRets["BottomQuantile"]
    
    # Step 5. Plot of cumulative LSRet
    LSCum = LSRets.apply(CumRetVec)
#   LSCum["LSRet"] = LSCum["TopQuantile"]-LSCum["BottomQuantile"]   # LS return without periodic rebalancing
    LSPlot = px.line(LSCum,x=LSCum.index,y=LSCum.columns)
    LSPlot.update_yaxes(title_text="Cumulative Return",tickformat='.1%')
    return([QRetsDF,QPlot,LSRets,LSPlot])   
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
def CalcBetaSigma(ydf,MRF,ydate="date",MRFdate="date",Return="RET",freq="Daily",T=252,EXPW=True,h=63):
    '''
    Calculates historical beta and sigma from daily or monthly returns.
    Arguments:
    ydf: security returns dataframe
    MRF: dataframe of market excess (column name MKTx) and risk-free returns (column name RF)
    ydate: date column name in ydf
    MRFdate: date column name in MRF
    Return: name of return column in ydf
    freq: frequency of returns data for annualization. Default is "Daily"
    T: number of lookback periods for regression. Default is 252. 
       NaN values returned if number of observations in ydf < T/2
    EXPW: bool, if True, regression is exponentially weighted
    h: half-life number of periods for exponential weighting. Default is 63
    
    Return value: pandas DataFrame with columns 'beta','beta_t','sigma', 'T'
                  where 'beta_t' is the t-stat for the beta value and 'T' is the
                  number of observations.
    '''
                                    
    #id = ydf["ticker"].unique()    # Useful for debugging
    
    ydf = ydf.loc[:,[ydate,Return]].drop_duplicates()
    
    if (ydf.shape[0] < T/2):               # No calculation if we have less than half the expected data
        return(pd.DataFrame({'beta':np.NaN,'beta_t':np.NaN,'sigma':np.NaN, 'T': ydf.shape[0]},index=[0]))
        
    if (ydf.shape[0] < T):
        T = ydf.shape[0]
    elif (ydf.shape[0] > T):
        ydf = ydf.sort_values(ydate)[-T:]
             
    ydf = ydf.merge(MRF,how='inner',left_on=ydate,right_on=MRFdate)
    ydf["RETx"] = ydf[Return]-ydf["RF"]
    ydf = ydf.loc[:,["RETx","MKTx"]]
    
    if (EXPW):
        assert h <= T,"h should be less than or equal to T"
        l = np.exp(np.log(0.5)/h)              # Exponential weighting constant lambda
        w = l**(T-np.arange(1,T+1))            # (lambda)^(T-t)
        model = smf.wls(formula = 'RETx ~ MKTx',weights=w,data=ydf)
    else:
        model = smf.ols(formula = 'RETx ~ MKTx',data=ydf)
        
    res = model.fit() 
    beta = res.params[1]
    beta_t = res.tvalues[1]
    sigma = np.sqrt(AnnFactor[freq]) * np.sqrt(res.ssr/(T-2))
    return(pd.DataFrame({'beta':beta,'beta_t':beta_t,'sigma':sigma, 'T':T},index=[0]))
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
    df['ret'] = df/df.loc[df.first_valid_index()]
    df['modMax'] = df.ret.cummax()
    df['modDD'] = 1-df.ret.div(df['modMax'])
    groups = df.groupby(df['modMax'])
    dd = groups[['modMax','modDD']].apply(lambda g: g[g['modDD'] == g['modDD'].max()])
    top10dd = dd[dd['modDD']>0].sort_values('modDD', ascending=False)
    return top10dd
#########################################################
def drawdown_group(df,index_list):
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
