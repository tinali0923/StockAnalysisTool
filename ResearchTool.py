
# coding: utf-8

import argparse

def terminal_app():
    
    global unit, ticker, start_date, end_date, ticker_list, weight_list, mode, frequency_for_rebalancing
    
    parser = argparse.ArgumentParser(description="Portfolio risk analysis tool")
    parser.add_argument('StartDate', help="the start date (yyyy-mm-dd)")
    parser.add_argument('EndDate', help="the end date (yyyy-mm-dd)")
    parser.add_argument('Tickers', help="The list of tickers to analyze, separated by commas. A list of two or more tickers will be formed into a portfolio.")
#    parser.add_argument('unit', help="The unit used for return calculation. Can be one of {d[aily], w[eekly], m[onthly]}")
    parser.add_argument('-u', '--unit', required=True, choices = ['d', 'w', 'm', 'daily', 'weekly', 'monthly'], help='The unit used for return calculation. Can be one of {d[aily], w[eekly], m[onthly]}')
    parser.add_argument('-r', '--rebalance', choices = ['d', 'w', 'm', 'daily', 'weekly', 'monthly'], help='The unit used for portfolio re-balancing. Can be one of {d[aily], w[eekly], m[onthly]}')
    parser.add_argument('-w', '--weights', help="The list of weights used on each ticker to construct the portfolio, separated by commas. Must sum up to less than 1. Default is equal weight.")
    args = parser.parse_args()
    print(args)
    def error_exit(s, code = 1):
        parser.print_usage()
        print(parser.prog + ': error: ' + s)
        exit(code)
    #import pdb; pdb.set_trace()
    unit = args.unit[0]
    start_date = args.StartDate
    end_date = args.EndDate
    if len(args.Tickers.split(',')) > 1:
        mode = 'y'
        ticker_list = args.Tickers.split(',')
        if not (args.rebalance):
            error_exit('--rebalance must be given for portfolio analysis.')
        frequency_for_rebalancing = args.rebalance[0]
        if (args.weights):
            if len(args.weights.split(',')) != len(ticker_list):
                error_exit('Incorrect number of weights given. Must give same number of weights as number of tickers')
            weight_list = [float(w) for w in args.weights.split(',')]
            if sum(weight_list) > 1:
                error_exit('Weights summed to greater than 1')
        else:
            weight_list = [1.0 / len(ticker_list)] * len(ticker_list)
    else:
        mode = 'n'
        ticker = args.Tickers
        if (args.weights): error_exit('No weights needed for single ticker.')
        if (args.rebalance): error_exit('No re-balance time unit needed for single ticker.')


if __name__ == '__main__':
    terminal_app()
    #exit()#Uncomment me


# In[1]:
import tushare as ts
import numpy as py
import pandas as pd
import math
import os
from datetime import datetime
from pandas.stats.api import ols
from pandas.tools.plotting import scatter_matrix
#from __future__ import print_function
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.integrate import quad
from sympy import *

#ts.set_token('a76443ee238b9549846083d5535bae10b278fa59daddba4bb6dbae6b9f2b6d1a')
mkt=ts.Market()

print(start_date)

# In[13]:

CACHE_FOLDER = "./cache/"

def CacheConstructor(function):
    def CachedFunction(*args, **kargs):
        cached = True
        file_name = CACHE_FOLDER + function.__module__ + '.' + function.__name__ + '().hdf5'
        with pd.HDFStore(file_name, format='table') as storage:
            #Build key and check for alpha_numerics:
            karg_list = []
            for k in sorted(kargs.keys()):
                karg_list.append(k)
                karg_list.append(kargs[k])
            key = 'k'
            for args_list in [args, karg_list]:
                for v in args_list:
                    if (not type(v) is str) or (not v.replace('-','').isalnum()):
                        raise NotImplementedError('None alphanumeric in the argument list.')
                    key += v.replace('-','_d') + '__'
                key = key[:-1] + 'k'
            if '/' + key not in storage.keys():
                    cached = False
        if not cached:
            data = function(*args, **kargs)
            with pd.HDFStore(file_name, format='table') as storage:
                storage.put(key, data)
        with pd.HDFStore(file_name, format='table') as storage:    
            return storage.get(key)
    return CachedFunction

cached_get_h=CacheConstructor(ts.get_h_data)
cached_get_hist=CacheConstructor(ts.get_hist_data)



# In[5]:

#Setting variables
'''
unit=''
ticker=''
start_date=''
end_date=''
mode=''
#start
#end

market_excess=0
SMB=0
HML=0
UMD=0
REV=0
return_mom=0
industry_mom=0

ticker_list=[]
weight_list=[]
frequency_for_rebalancing=''
'''

data=pd.DataFrame()


# In[6]:

def cal_return(ticker, start_date, end_date, unit, date_index_string_hyphen):
    data=ts.get_h_data(ticker,start_date,end_date,autype='hfq')
    try:    
        data=data.loc[data.index.isin(date_index_string_hyphen)]
        if unit == 'd':
            data_day=data
            data_day['daily_return']=data_day['close'].pct_change(-1)
            return data_day
        elif unit == 'w':
            data_week=data
            data_week['weekly_return']=data_week['close'].pct_change(-1)
            data_week['end date of week']=pd.to_datetime(data_week.index.to_series())
            data_week=data_week.set_index('end date of week')
            return data_week
        elif unit == 'm':
            data_month=data
            data_month['monthly_return']=data_month['close'].pct_change(-1)
            data_month['end date of month']=pd.to_datetime(data_month.index.to_series())
            data_month=data_month.set_index('end date of month')
            return data_month
    except:
        print('No data available for the indicated period.')


# In[8]:

#program()


# In[6]:

#program()


# In[10]:

#construct the main chart
if mode =='n':
    data_copy=cached_get_hist(ticker,start_date,end_date,ktype=unit)
    data_copy['datetime']=pd.to_datetime(data_copy.index.to_series())
    data_copy=data_copy.set_index('datetime')
    date_index_string_hyphen=data_copy.index.to_series().apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
   
    return_dependent_variable=cal_return(ticker, start_date, end_date, unit,date_index_string_hyphen)
    date_index=return_dependent_variable.index
    date_index_string=return_dependent_variable.index.to_series().apply(lambda x: x.strftime('%Y%m%d')).tolist()


if mode=='y':
    portfolio_close_price=pd.DataFrame()

    data_copy=cached_get_hist(ticker_list[0],start_date,end_date,ktype=unit)
    data_copy['datetime']=pd.to_datetime(data_copy.index.to_series())
    data_copy=data_copy.set_index('datetime')
    date_index_string_hyphen=data_copy.index.to_series().apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    
    return_dependent_variable_first=cal_return(ticker_list[0], start_date, end_date, unit,date_index_string_hyphen)
    date_index=return_dependent_variable_first.index
    date_index_string=return_dependent_variable_first.index.to_series().apply(lambda x: x.strftime('%Y%m%d')).tolist()


# In[11]:

#Setting Variables
Factor_Data=pd.DataFrame(index=date_index)
Factor_Data['Risk Free Rate']=py.nan
Factor_Data['Market Return']=py.nan
Factor_Data['Industry Return']=py.nan
Factor_Data['Concept Return']=py.nan
Factor_Data['SMB']=py.nan
Factor_Data['HML']=py.nan
Factor_Data['UMD']=py.nan


# In[12]:

########################  Market Return & Risk Free Rate ###########################
print('Calculating market return...')

market_return=cached_get_hist('hs300',start_date,end_date,ktype=unit)
market_return['date']=pd.to_datetime(market_return.index)
market_return=market_return.set_index('date')

for mr in range(len(date_index)):
    Factor_Data['Market Return'][date_index[mr]]=market_return['p_change'][date_index[mr]]

rf=0.035
Factor_Data['Risk Free Rate']=[rf]*len(date_index)


# In[10]:

####################  Industry  #######################
print('Calculating industry return...')

def add_zeros(tickers):
    for k in range(len(tickers)):
        tickers[k]=str(tickers[k])
        if len(tickers[k])!=6:
            tickers[k]='0'*(6-len(tickers[k]))+tickers[k]
    return tickers

#Getting industry information
industry_classification=ts.get_industry_classified()
industry_name=industry_classification.loc[industry_classification['code']==ticker]['c_name'].to_string()[-4:]
all_companies_same_industry=industry_classification.loc[industry_classification['c_name']==industry_name]


# In[13]:

####Industry return
for each_date in range(len(date_index_string)-2):
    all_companies_same_industry.loc[:,'return '+ date_index_string[each_date]]=py.nan
    all_companies_same_industry.loc[:,'market cap '+ date_index_string[each_date]]=py.nan
    all_companies_same_industry.loc[:,'percentage '+ date_index_string[each_date]]=py.nan
    all_companies_same_industry.loc[:,'weighted return '+ date_index_string[each_date]]=py.nan
    
number_industry=all_companies_same_industry.count()['code']
all_companies_same_industry=all_companies_same_industry.set_index(pd.Series(list(range(number_industry))))
list_of_company=all_companies_same_industry['code'].tolist()


# In[14]:

#Getting returns, market cap
for each_company in range(number_industry):
    try:
        company_return=cal_return(all_companies_same_industry['code'][each_company],start_date,end_date,unit,date_index_string_hyphen)
    except TypeError:
        continue
    for each_date in range(len(date_index_string)-2):
        try:
            all_companies_same_industry['return '+ date_index_string[each_date]][each_company]=company_return.loc[date_index[each_date]][-1]
        except KeyError:
            continue


# In[ ]:

for each_date in range(len(date_index)-2):
    try:
        industry_stock_cap_one_day=mkt.StockFactorsOneDay(tradeDate=date_index_string[each_date],field='ASSI,ticker')
    except :
        industry_stock_cap_one_day=mkt.StockFactorsOneDay(tradeDate=date_index_string[each_date],field='ASSI,ticker')
    industry_stock_cap_one_day['ticker']=pd.Series(add_zeros(industry_stock_cap_one_day['ticker'].tolist()))
    for each_company in range(number_industry):
        try:
            market_cap_one_company=industry_stock_cap_one_day.loc[industry_stock_cap_one_day['ticker']==list_of_company[each_company]]['ASSI'].__float__()
            all_companies_same_industry['market cap '+ date_index_string[each_date]][each_company]=market_cap_one_company
        except (ValueError,TypeError):
            continue
###!!!!!!!!! a lot of data of market cap is Nan, need to know the reason


# In[14]:

#calculating weighted return
for each_date in range(len(date_index)-2):
    all_companies_same_industry['percentage '+ date_index_string[each_date]]=all_companies_same_industry['market cap '+ date_index_string[each_date]]/all_companies_same_industry['market cap '+ date_index_string[each_date]].sum()
    all_companies_same_industry['weighted return '+ date_index_string[each_date]]=all_companies_same_industry['percentage '+ date_index_string[each_date]]*all_companies_same_industry['return '+ date_index_string[each_date]]

#Industry Return
for each_date in range(len(date_index)-2):
    Industry_Return=all_companies_same_industry['weighted return '+ date_index_string[each_date]].sum()
    Factor_Data['Industry Return'][each_date]=Industry_Return


# In[18]:

###################  Concept  ####################
print('Calculating concept return...')


#Getting concept information
concept_classification=ts.get_concept_classified()
concept_name=concept_classification.loc[concept_classification['code']==ticker]['c_name'].to_string()[-4:]
all_companies_same_concept=concept_classification.loc[concept_classification['c_name']==concept_name]

####concept return
for each_date in range(len(date_index_string)-2):
    all_companies_same_concept['return '+ date_index_string[each_date]]=py.nan
    all_companies_same_concept['market cap '+ date_index_string[each_date]]=py.nan
    all_companies_same_concept['percentage '+ date_index_string[each_date]]=py.nan
    all_companies_same_concept['weighted return '+ date_index_string[each_date]]=py.nan
    
number_concept=all_companies_same_concept.count()['code']
all_companies_same_concept=all_companies_same_concept.set_index(pd.Series(list(range(number_concept))))
list_of_company_concept=all_companies_same_concept['code'].tolist()

#Getting returns, market cap
for each_company in range(number_concept):
    try:
        company_return=cal_return(all_companies_same_concept['code'][each_company],start_date,end_date,unit,date_index_string_hyphen)
    except:
        continue
    for each_date in range(len(date_index_string)-2):
        try:
            all_companies_same_concept['return '+ date_index_string[each_date]][each_company]=company_return.loc[date_index[each_date]][-1]
        except:
            continue

for each_date in range(len(date_index)-2):
    try:
        concept_stock_cap_one_day=mkt.StockFactorsOneDay(tradeDate=date_index_string[each_date],field='ASSI,ticker')
    except :
        concept_stock_cap_one_day=mkt.StockFactorsOneDay(tradeDate=date_index_string[each_date],field='ASSI,ticker')
    concept_stock_cap_one_day['ticker']=pd.Series(add_zeros(concept_stock_cap_one_day['ticker'].tolist()))
    for each_company in range(number_concept):
        try:
            market_cap_one_company=concept_stock_cap_one_day.loc[concept_stock_cap_one_day['ticker']==list_of_company_concept[each_company]]['ASSI'].__float__()
            all_companies_same_concept['market cap '+ date_index_string[each_date]][each_company]=market_cap_one_company
        except (ValueError,TypeError):
            continue
###!!!!!!!!! a lot of data of market cap is Nan, need to know the reason

#calculating weighted return
for each_date in range(len(date_index)-2):
    all_companies_same_concept['percentage '+ date_index_string[each_date]]=all_companies_same_concept['market cap '+ date_index_string[each_date]]/all_companies_same_concept['market cap '+ date_index_string[each_date]].sum()
    all_companies_same_concept['weighted return '+ date_index_string[each_date]]=all_companies_same_concept['percentage '+ date_index_string[each_date]]*all_companies_same_concept['return '+ date_index_string[each_date]]

#Concept Return
for each_date in range(len(date_index)-2):
    Concept_Return=all_companies_same_concept['weighted return '+ date_index_string[each_date]].sum()
    Factor_Data['Concept Return'][each_date]=Concept_Return



# In[ ]:

##################  SMB & HML  ####################
print('Calculating SMB & HML return...')

x=0.3
for k in range(len(date_index_string)-2):
    try:
        all_stock_cap_PB_one_day=mkt.StockFactorsOneDay(tradeDate=date_index_string[k],field='ASSI,PB,ticker')
    except:
        all_stock_cap_PB_one_day=mkt.StockFactorsOneDay(tradeDate=date_index_string[k],field='ASSI,PB,ticker')


    all_stock_cap_PB_one_day=all_stock_cap_PB_one_day[all_stock_cap_PB_one_day['PB'].notnull()]
    all_stock_cap_PB_one_day=all_stock_cap_PB_one_day[all_stock_cap_PB_one_day['ASSI'].notnull()]

    number=all_stock_cap_PB_one_day.index.__len__()
    number_x=int(number*x)
    ###############  SMB  ################
    sorted_stock_cap=all_stock_cap_PB_one_day.sort_values(by='ASSI',ascending=True).set_index(pd.Series(list(range(number))))

    #Creating Sorting index and Sum
    sorted_stock_cap['sorting index']=number_x*['A']+(number-2*number_x)*[' ']+number_x*['B']

    #Getting tickers
    small_tickers_cap=sorted_stock_cap.loc[list(range(number_x))]['ticker'].tolist()
    big_tickers_cap=sorted_stock_cap.loc[list(range(number-number_x,number))]['ticker'].tolist()

    small_tickers_cap=add_zeros(small_tickers_cap)
    big_tickers_cap=add_zeros(big_tickers_cap)

    #Getting Returns
    sorted_stock_cap['return']=py.nan

    for y in range(0,number_x):
        try:
            sorted_stock_cap['return'][y]=cal_return(small_tickers_cap[y], start_date, end_date, unit,date_index_string_hyphen).loc[date_index[k]][-1]
        except:
            sorted_stock_cap['return'][y]=py.nan

    for z in range(number-number_x,number):
        try:
            sorted_stock_cap['return'][z]=cal_return(big_tickers_cap[z-(number-number_x)],start_date,end_date,unit,date_index_string_hyphen).loc[date_index[k]][-1]
        except:
            sorted_stock_cap['return'][z]=py.nan

    #Weighted Return
    grouped_cap=sorted_stock_cap.groupby("sorting index")

    Small_Index=grouped_cap.get_group('A')['return'].mean()
    Big_Index=grouped_cap.get_group('B')['return'].mean()

    SMB=Small_Index-Big_Index

    Factor_Data['SMB'][date_index[k]]=SMB
    
    
    ############  HML  ############

    
    #Creating Sorting index and Sum
    sorted_stock_PB=sorted_stock_cap.sort_values(by='PB',ascending=True).set_index(pd.Series(list(range(number))))
    sorted_stock_PB['sorting index']=int(number*0.5)*['A']+(number-2*int(number*0.5))*[' ']+int(number*0.5)*['B']

    #Getting tickers
    small_tickers_PB=sorted_stock_PB.loc[list(range(int(number*0.5)))]['ticker'].tolist()
    big_tickers_PB=sorted_stock_PB.loc[list(range(number-int(number*0.5),number))]['ticker'].tolist()

    small_tickers_PB=add_zeros(small_tickers_PB)
    big_tickers_PB=add_zeros(big_tickers_PB)

    #Getting Returns
    for y in range(0,int(number*0.5)):
        if math.isnan(sorted_stock_PB['return'][y]):
            try:
                sorted_stock_PB['return'][y]=cal_return(small_tickers_PB[y],start_date,end_date,unit,date_index_string_hyphen).loc[date_index[k]][-1]
            except:
                sorted_stock_PB['return'][y]=py.nan

    for z in range(number-int(number*0.5),number):
        if math.isnan(sorted_stock_PB['return'][y]):
            try:
                sorted_stock_PB['return'][z]=cal_return(big_tickers_PB[z-(number-number_x)],start_date,end_date,unit,date_index_string_hyphen).loc[date_index[k]][-1]
            except:
                sorted_stock_PB['return'][z]=py.nan

    #Calculating weighted return
    grouped_PB=sorted_stock_PB.groupby("sorting index")

    High_Index=grouped_PB.get_group('A')['return'].mean()
    Low_Index=grouped_PB.get_group('B')['return'].mean()
  
    HML=High_Index-Low_Index

    Factor_Data['HML'][date_index[k]]=HML


# In[100]:

############### UMD #################
print('Calculating UMD return...')

#Getting a chart of all stocks and their return for all dates
all_stock_basics=ts.get_stock_basics()
all_tickers=all_stock_basics.index.tolist()
Momentum_Data=pd.DataFrame(index=all_stock_basics.index)
Momentum_Data['market cap']=py.nan

for each_date in date_index_string[:-1]:
    Momentum_Data[each_date]=py.nan

for each in all_tickers:
    try:
        each_stock_record=cal_return(each,start_date,end_date,unit,date_index_string_hyphen)
    except:
        continue
    for a_number in range(len(date_index)-1):
        try:
            Momentum_Data.loc[each][date_index_string[a_number]]=each_stock_record.loc[date_index[a_number]][-1].__float__()
        except:
            continue

Momentum_Data_clean=Momentum_Data
for each_date in range(len(date_index_string)-1):
    Momentum_Data_clean=Momentum_Data_clean[Momentum_Data_clean[date_index_string[each_date]].notnull()]

    
def cal_UMD_one_period(k, Momentum_Data_clean, date_index, date_index_string):
    Momentum_Data_clean.sort_values(by=date_index_string[k],ascending=False)

    all_tickers_new=Momentum_Data_clean.index.tolist()

    try:
        all_stock_cap_one_day_momentum=mkt.StockFactorsOneDay(tradeDate=date_index_string[k],field='ASSI,ticker')
    except:
        all_stock_cap_one_day_momentum=mkt.StockFactorsOneDay(tradeDate=date_index_string[k],field='ASSI,ticker')

    for each in all_tickers_new:
        Momentum_Data_clean['market cap'][each]=all_stock_cap_one_day_momentum.loc[all_stock_cap_one_day_momentum['ticker']==int(each)]['ASSI'].__float__()

    Momentum_Data_clean=Momentum_Data_clean[Momentum_Data_clean['market cap'].notnull()]

    number_momentum=Momentum_Data_clean.count()[date_index_string[k]]
    number_momentum_x=int(0.3*number_momentum)

    #adding sorting index
    Momentum_Data_clean['sorting index']=number_momentum_x*['A']+(number_momentum-2*number_momentum_x)*[' ']+number_momentum_x*['B']
    grouped_momentum=Momentum_Data_clean.groupby('sorting index')
    big_cap=grouped_momentum.get_group('A').sum()['market cap']
    small_cap=grouped_momentum.get_group('B').sum()['market cap']
    Momentum_Data_clean['sum']=[big_cap]*number_momentum_x+(number_momentum-2*number_momentum_x)*[py.nan]+[small_cap]*number_momentum_x
    Momentum_Data_clean['percentage']=Momentum_Data_clean['market cap']/Momentum_Data_clean['sum']

    #Getting weighted return
    Momentum_Data_clean['weighted return']=Momentum_Data_clean[date_index_string[k]]*Momentum_Data_clean['percentage']

    #Getting Final Result
    grouped_momentum_again=Momentum_Data_clean.groupby('sorting index')
    top_30=grouped_momentum_again.get_group('A').sum()['weighted return']
    bottom_30=grouped_momentum_again.get_group('B').sum()['weighted return']
    Momentum_one_period=top_30-bottom_30
    
    return Momentum_one_period

    
for k in range(1,len(date_index)-1):
    
    Factor_Data['UMD'][date_index[k-1]]=cal_UMD_one_period(k, Momentum_Data_clean, date_index, date_index_string)


# In[ ]:

if mode == 'y':
    for every_ticker in ticker_list:
        every_stock_in_portfolio=cal_return(every_ticker, start_date, end_date, unit,date_index_string_hyphen)
        portfolio_close_price[every_ticker+' close']=every_stock_in_portfolio['close']

    #set index
    if unit=='w':
        portfolio_close_price['end date of week']=date_index_string_hyphen
        portfolio_close_price=portfolio_close_price.set_index('end date of week')
    if unit=='m':
        portfolio_close_price['end date of month']=date_index_string_hyphen
        portfolio_close_price=portfolio_close_price.set_index('end date of month')

    #get the date for rebalancing
    rebalancing_date=cached_get_hist(ticker_list[0],start_date,end_date,ktype=frequency_for_rebalancing).index[::-1]

    #initialise the weight chart
    weight_chart=pd.DataFrame()
    weight_chart['column']=['designated weight','weight after rebalancing','capital reallocation','total capital','new unit']
    weight_chart=weight_chart.set_index('column')

    total_weight=0
    for every_weight in weight_list:
        total_weight+=every_weight

    cash_weight=1-total_weight

    for number in range(len(ticker_list)):
        weight_chart[ticker_list[number]]=[weight_list[number],weight_list[number],0,10000/cash_weight*weight_list[number],10000/cash_weight*weight_list[number]/portfolio_close_price[ticker_list[number]+' close'][rebalancing_date[0]]]

    weight_chart['cash']=[cash_weight, cash_weight,0,10000,'N/A']

    #initialise captital & cash column in the main chart
    total_cap=0
    for every_ticker in ticker_list:
        portfolio_close_price[every_ticker+' asset capital']=weight_chart[every_ticker][3]
        total_cap+=weight_chart[every_ticker][3]
    portfolio_close_price['cash']=10000
    portfolio_close_price['total capital']=total_cap+10000

    #rebalancing algo functions

    #generate a list of delta weight
    def cal_delta_weight(ticker_list, weight_chart):
        delta_weight=[]
        for every_ticker in ticker_list:
            individual_weight=weight_chart[every_ticker]['weight after rebalancing']-weight_chart[every_ticker]['designated weight']
            delta_weight.append(individual_weight)
        individual_weight=weight_chart['cash']['weight after rebalancing']-weight_chart['cash']['designated weight']
        delta_weight.append(individual_weight)
        return delta_weight

    #find the maximum in delta weight list
    def find_max(delta_weight, symbol):
        if symbol==0:
            maximum=0
            prev=py.fabs(delta_weight[0])
            for num in range(1,len(delta_weight)):
                item=py.fabs(delta_weight[num])
                if item>prev:
                    maximum=num
                    prev=item
            return maximum
        if symbol==1:
            maximum=0
            prev=-1
            count=0
            while prev<0:
                prev=delta_weight[count]
                maximum=count
                count+=1
            for num in range(count,len(delta_weight)):
                item=delta_weight[num]
                if item>=0:
                    if item>prev:
                        maximum=num
                        prev=item
            return maximum
        if symbol==-1:
            maximum=0
            prev=1
            count=0
            while prev>0:
                prev=delta_weight[count]
                maximum=count
                count+=1
            for num in range(count, len(delta_weight)):
                item=delta_weight[num]
                if item<=0:
                    if item<prev:
                        maximum=num
                        prev=item
            return maximum

    #initilise the list
    ticker_cash_list=ticker_list+['cash']


    #rebalancing algo: for each rebalancing date    
    for the_date in rebalancing_date[1::]:

    #update new weight and new capital in weight chart
        new_total_capital=0
        for every_ticker in ticker_list:
            weight_chart[every_ticker][3]=weight_chart[every_ticker][4]*portfolio_close_price[every_ticker+' close'][the_date]
            new_total_capital+=weight_chart[every_ticker][3]
        new_total_capital+=weight_chart['cash'][3]

        new_total_weight=0
        for every_ticker in ticker_list:
            weight_chart[every_ticker][1]=weight_chart[every_ticker][3]/new_total_capital
            new_total_weight+=weight_chart[every_ticker][1]
        weight_chart['cash'][1]=1-new_total_weight

    #update total capital in the main chart
        portfolio_close_price['total capital'][the_date]=new_total_capital

    #the rebalancing algo, update every item in weight chart
        delta_weight=cal_delta_weight(ticker_list, weight_chart)
        index_of_max=find_max(delta_weight,0)
        while py.fabs(delta_weight[index_of_max])>0.001:
            if delta_weight[index_of_max]>0:
                index_of_oppo_max=find_max(delta_weight,-1)
            if delta_weight[index_of_max]<0:
                index_of_oppo_max=find_max(delta_weight,1)
            weight_chart[ticker_cash_list[index_of_max]][1]+=delta_weight[index_of_oppo_max]
            weight_chart[ticker_cash_list[index_of_oppo_max]][1]=weight_chart[ticker_cash_list[index_of_oppo_max]][0]
            cap_to_reallocate=portfolio_close_price['total capital'][the_date]*delta_weight[index_of_oppo_max]
            weight_chart[ticker_cash_list[index_of_max]][2]+=cap_to_reallocate
            weight_chart[ticker_cash_list[index_of_oppo_max]][2]-=cap_to_reallocate
            weight_chart[ticker_cash_list[index_of_max]][3]+=cap_to_reallocate
            weight_chart[ticker_cash_list[index_of_oppo_max]][3]-=cap_to_reallocate
            if index_of_max!=len(delta_weight)-1:
                weight_chart[ticker_cash_list[index_of_max]][4]+=cap_to_reallocate/portfolio_close_price[ticker_cash_list[index_of_max]+' close'][the_date]
            if index_of_oppo_max!=len(delta_weight)-1:
                weight_chart[ticker_cash_list[index_of_oppo_max]][4]-=cap_to_reallocate/portfolio_close_price[ticker_cash_list[index_of_oppo_max]+' close'][the_date]
            delta_weight[index_of_max]+=delta_weight[index_of_oppo_max]
            delta_weight[index_of_oppo_max]=0
            #print(weight_chart)
            for every_item in ticker_cash_list:
                weight_chart[every_item][2]=0
            index_of_max=find_max(delta_weight,0)

    #update the main chart
        for every_ticker in ticker_list:
            portfolio_close_price[every_ticker+' asset capital'][the_date]=weight_chart[every_ticker][3]
        portfolio_close_price['cash'][the_date]=weight_chart['cash'][3]



    #transform the portfolio main chart into return_dependent_variable
    portfolio_close_price['portfolio return']=portfolio_close_price['total capital'].pct_change(-1)

    return_dependent_variable=pd.DataFrame(index=portfolio_close_price.index)
    return_dependent_variable['return']=py.nan
    return_dependent_variable['return']=portfolio_close_price['portfolio return']


# In[ ]:
print('Dependent Variable Return Chart')
if mode=='n':
    print(return_dependent_variable)
if mode=='y':
    print(portfolio_close_price)

print('Factor Data Chart')
print(Factor_Data)

# In[12]:

######################  Linear Regression  ######################
#Combining data
Factor_Data['Market Excess Return']=Factor_Data['Market Return']-Factor_Data['Risk Free Rate']
Factor_Data['Industry Excess Return']=Factor_Data['Industry Return']-Factor_Data['Risk Free Rate']
Factor_Data['Concept Excess Return']=Factor_Data['Concept Return']-Factor_Data['Risk Free Rate']
Complete_Data=Factor_Data.copy()[0:-2]
#Complete_Data['stock return']=py.nan
#Complete_Data['stock return'][:]=return_dependent_variable.iloc[0:-2, -1]
#Complete_Data['dependent variable']=Complete_Data['stock return']-Complete_Data['Risk Free Rate']

# Method 2, Robust analysis
Complete_Data=Complete_Data[['Market Excess Return','Industry Excess Return','Concept Excess Return','SMB','HML','UMD']]
return_dependent_variable2=return_dependent_variable.copy()
return_dependent_variable2.iloc[:,-1]=return_dependent_variable2.iloc[:,-1]-rf
Dependent_Variable_array=return_dependent_variable2.iloc[:-2,-1].as_matrix()


# In[86]:

print('Robust linear regression with no lag:')
Complete_Data_array=Complete_Data.as_matrix()
Complete_Data_array=sm.add_constant(Complete_Data_array)

return_dependent_variable2=return_dependent_variable.copy()
return_dependent_variable2.iloc[:,-1]=return_dependent_variable2.iloc[:,-1]-rf
Dependent_Variable_array=return_dependent_variable2.iloc[:-2,-1].as_matrix()

regression_model = sm.RLM(Dependent_Variable_array, Complete_Data_array, M=sm.robust.norms.HuberT())

regression_result=regression_model.fit()
print(regression_result.summary())


# In[37]:

######################   Correlation Matrix  #######################
Correlation_Matrix=Complete_Data[['Market Excess Return','Industry Excess Return','Concept Excess Return','SMB','HML','UMD']]
Correlation_Matrix=Correlation_Matrix.corr()
print('Correlation Matrix, robust linear regression')
print(Correlation_Matrix)


# In[2]:

###################  Covariance Scatter Plot  #######################
Complete_Data_Plot=Complete_Data.copy()
Complete_Data_Plot.columns=['MER', 'IER','CER', 'SMB', 'HML', 'UMD']

Covariance_Plot=scatter_matrix(Complete_Data_Plot, alpha=1, figsize=(10, 10), diagonal='kde')

print('Covariance Scatter Plot, robust linear regression')
matplotlib.pyplot.show()


# In[234]:

################# Risk Contribution Chart  #########################
Covariance_Matrix=Complete_Data[['Market Excess Return','Industry Excess Return','Concept Excess Return','SMB','HML','UMD']]
Covariance_Matrix=Covariance_Matrix.cov()
se=regression_result.bse
coef=regression_result.params

#Calculate porfolio risk
portfolio_risk=0
for item in range(1,len(se)):
    each=(se[item]*coef[item])**2
    portfolio_risk=portfolio_risk+each


for row in range(len(Covariance_Matrix.columns)):
    for column in range(row+1,len(Covariance_Matrix.iloc[row])):
        each=2*Covariance_Matrix.iloc[row,column]*coef[row+1]*coef[column+1]
        portfolio_risk=portfolio_risk+each
portfolio_risk=portfolio_risk**0.5

# Calculate individual risk contribution
def cal_risk_contribution(k,se,coef,Covariance_Matrix,portfolio_risk):
    def integrand(x,k,se,coef,Covariance_Matrix,portfolio_risk):
        risk_x=0
        li=list(range(1,len(se)))
        li.remove(k+1)

        for item in li:
            each=(se[item]*coef[item])**2
            risk_x=risk_x+each

        li2=list(range(len(Covariance_Matrix.columns)))
        li2.remove(k)

        for row in li2:
            for column in range(row+1,len(Covariance_Matrix.iloc[row])):
                each=2*Covariance_Matrix.iloc[row,column]*coef[row+1]*coef[column+1]
                risk_x=risk_x+each
        risk_x

        y=Symbol('y')

        summation=0
        for column in range(k+1,len(Covariance_Matrix.iloc[row])):
            each=2*Covariance_Matrix.iloc[row,column]*y*coef[column+1]
            summation=summation+each

        risk=(y**2*se[k+1]**2+summation+risk_x)**0.5
        return risk.diff(y).replace(y,x)
    I = quad(integrand, 0, coef[k+1], args=(k,se,coef,Covariance_Matrix,portfolio_risk))
    contribution=I[0]/portfolio_risk
    return contribution

##Risk Contribution Chart in number
risk_contribution_chart=pd.DataFrame(py.nan,index=['risk contribution'],columns=Covariance_Matrix.columns)
for k in range(len(Covariance_Matrix.columns)):
    risk_contribution_chart.iloc[0,k]=cal_risk_contribution(k,se,coef,Covariance_Matrix,portfolio_risk)

print('Risk Contribution Chart, robust linear regression')

print(risk_contribution_chart)


# In[ ]:

##Risk Contribution Chart in graphics
risk_contribution_chart.plot.bar(); plt.axhline(0, color='k')
matplotlib.pyplot.show()


# In[ ]:

def find_max(a,b,c,d,e,f):
    li=[a,b,c,e,d,f]
    maximum=a
    for x in range(1,6):
        if li[x]>maximum:
            maximum=li[x]
    return maximum


# In[ ]:

lag_number=len(Complete_Data.index)-8
column_list=Complete_Data.columns
count=0
Complete_Data_2=Complete_Data.copy()

max_1=0
max_2=0
max_3=0
max_4=0
max_5=0
max_6=0
max_rsquared=0

for a in range(lag_number):
    for b in range(lag_number):
        for c in range(lag_number):
            for d in range(lag_number):
                for e in range(lag_number):
                    for f in range(lag_number):
                        Complete_Data_2=Complete_Data.copy()
                        Dependent_Variable_array_2=Dependent_Variable_array.copy()
                        Complete_Data_2[column_list[0]]=Complete_Data_2[column_list[0]].shift(-a)
                        Complete_Data_2[column_list[1]]=Complete_Data_2[column_list[1]].shift(-b)
                        Complete_Data_2[column_list[2]]=Complete_Data_2[column_list[2]].shift(-c)
                        Complete_Data_2[column_list[3]]=Complete_Data_2[column_list[3]].shift(-d)
                        Complete_Data_2[column_list[4]]=Complete_Data_2[column_list[4]].shift(-e)
                        Complete_Data_2[column_list[5]]=Complete_Data_2[column_list[5]].shift(-f)
                        
                        
                        max_lag=find_max(a,b,c,d,e,f)
                        
                        if max_lag>0:
                            Complete_Data_2=Complete_Data_2[:-max_lag]
                            Dependent_Variable_array_2=Dependent_Variable_array_2[:-max_lag]
                        
                        Complete_Data_array=Complete_Data_2.as_matrix()

                        Complete_Data_array=sm.add_constant(Complete_Data_array)
                        regression_model = sm.OLS(Dependent_Variable_array_2, Complete_Data_array, M=sm.robust.norms.HuberT())
                        regression_result=regression_model.fit()

                        if regression_result.rsquared>max_rsquared:
                            max_1=a
                            max_2=b
                            max_3=c
                            max_4=d
                            max_5=e
                            max_6=f
                            
                            max_rsquared=regression_result.rsquared

print('The lag combination that maximises r-square is', max_1, ',' , max_2 , ',' , max_3 , ',', max_4 , ',' ,max_5 , ',' , max_6 ,'.')
Complete_Data_2=Complete_Data.copy()
Dependent_Variable_array_2=Dependent_Variable_array.copy()
Complete_Data_2[column_list[0]]=Complete_Data_2[column_list[0]].shift(-max_1)
Complete_Data_2[column_list[1]]=Complete_Data_2[column_list[1]].shift(-max_2)
Complete_Data_2[column_list[2]]=Complete_Data_2[column_list[2]].shift(-max_3)
Complete_Data_2[column_list[3]]=Complete_Data_2[column_list[3]].shift(-max_4)
Complete_Data_2[column_list[4]]=Complete_Data_2[column_list[4]].shift(-max_5)
Complete_Data_2[column_list[5]]=Complete_Data_2[column_list[5]].shift(-max_6)


max_lag=find_max(max_1,max_2,max_3,max_4,max_5,max_6)

if max_lag>0:
    Complete_Data_2=Complete_Data_2[:-max_lag]
    Dependent_Variable_array_2=Dependent_Variable_array_2[:-max_lag]

Complete_Data_array=Complete_Data_2.as_matrix()

Complete_Data_array=sm.add_constant(Complete_Data_array)
regression_model = sm.OLS(Dependent_Variable_array_2, Complete_Data_array, M=sm.robust.norms.HuberT())
regression_result=regression_model.fit()
print(regression_result.summary())


# In[ ]:

######################   Correlation Matrix  #######################
Correlation_Matrix=Complete_Data_2[['Market Excess Return','Industry Excess Return','Concept Excess Return','SMB','HML','UMD']]
Correlation_Matrix=Correlation_Matrix.corr()

print('Correlation Matrix, with maximised r-squred value')
print(Correlation_Matrix)


# In[ ]:

###################  Covariance Scatter Plot  #######################
Complete_Data_Plot=Complete_Data_2.copy()
Complete_Data_Plot.columns=['MER', 'IER','CER', 'SMB', 'HML', 'UMD']

Covariance_Plot=scatter_matrix(Complete_Data_Plot, alpha=1, figsize=(10, 10), diagonal='kde')

print('Covariance Scatter Plot, with maximised r-squred value')
matplotlib.pyplot.show()


# In[ ]:

################# Risk Contribution Chart  #########################
Covariance_Matrix=Complete_Data_2[['Market Excess Return','Industry Excess Return','Concept Excess Return','SMB','HML','UMD']]
Covariance_Matrix=Covariance_Matrix.cov()
se=regression_result.bse
coef=regression_result.params

#Calculate porfolio risk
portfolio_risk=0
for item in range(1,len(se)):
    each=(se[item]*coef[item])**2
    portfolio_risk=portfolio_risk+each


for row in range(len(Covariance_Matrix.columns)):
    for column in range(row+1,len(Covariance_Matrix.iloc[row])):
        each=2*Covariance_Matrix.iloc[row,column]*coef[row+1]*coef[column+1]
        portfolio_risk=portfolio_risk+each
portfolio_risk=portfolio_risk**0.5

# Calculate individual risk contribution
def cal_risk_contribution(k,se,coef,Covariance_Matrix,portfolio_risk):
    def integrand(x,k,se,coef,Covariance_Matrix,portfolio_risk):
        risk_x=0
        li=list(range(1,len(se)))
        li.remove(k+1)

        for item in li:
            each=(se[item]*coef[item])**2
            risk_x=risk_x+each

        li2=list(range(len(Covariance_Matrix.columns)))
        li2.remove(k)

        for row in li2:
            for column in range(row+1,len(Covariance_Matrix.iloc[row])):
                each=2*Covariance_Matrix.iloc[row,column]*coef[row+1]*coef[column+1]
                risk_x=risk_x+each
        risk_x

        y=Symbol('y')

        summation=0
        for column in range(k+1,len(Covariance_Matrix.iloc[row])):
            each=2*Covariance_Matrix.iloc[row,column]*y*coef[column+1]
            summation=summation+each

        risk=(y**2*se[k+1]**2+summation+risk_x)**0.5
        return risk.diff(y).replace(y,x)
    I = quad(integrand, 0, coef[k+1], args=(k,se,coef,Covariance_Matrix,portfolio_risk))
    contribution=I[0]/portfolio_risk
    return contribution

##Risk Contribution Chart in number
risk_contribution_chart=pd.DataFrame(py.nan,index=['risk contribution'],columns=Covariance_Matrix.columns)
for k in range(len(Covariance_Matrix.columns)):
    risk_contribution_chart.iloc[0,k]=cal_risk_contribution(k,se,coef,Covariance_Matrix,portfolio_risk)

print('Risk Contribution Chart, with maximised r-squred value')
print(risk_contribution_chart)


# In[ ]:

##Risk Contribution Chart in graphics
risk_contribution_chart.plot.bar(); plt.axhline(0, color='k')
matplotlib.pyplot.show()

