# LH_VS_价值策略


# 是否开启性能分析
#enable_profile()

'''
================================================================================
导入Python库文件
================================================================================
'''
# 基础库
import pandas as pd
import numpy as np
import datetime as dt
# IO
from six import StringIO
import pickle
# 导入自己的函数库
import LH_common_10_StockFilter as lhsf


'''
================================================================================
总体回测前
================================================================================
'''
#总体回测前要做的事情,g变量设定后无法在交易中通过更新代码的方式进行修改
def initialize(context):
    set_variables()                          # 设置中间变量
    set_backtest()                           # 设置回测条件
    set_benchmark('000300.XSHG')             # 设置回测基准
    set_trade_freq()                         # 设置交易频率，按天、周、月

#设置中间变量，仅定义需要在回测间保存的变量
def set_variables():
    # 计算持仓比例
    g.risk_lag = 180                         # 定义计算风险的分析期间
    g.confidence_level = 2.58                # 定义置信率，即：正态分布下的发生概率，1.96, 95%; 2.06, 96%; 2.18, 97%; 2.34, 98%; 2.58, 99%; 5, 99.9999%
    g.portfolio_risk_lag = 60
    g.portfolio_confidence_level = 1.96      # 1.96, 95%
    g.portfolio_risk_rate = 0.03             # 定义能够成熟的最大损失

    g.portfolio_risk_lag_w = 30
    g.portfolio_confidence_level_w = 2.18    # 1.96, 95%
    g.portfolio_risk_rate_w = 0.05           # 定义能够成熟的最大损失

    # 初始化df
    df_initial = pd.DataFrame()
    write_file('data/LH/stock_hold_history.csv', df_initial.to_csv(index=True), append=False)

    #
    g.stock_to_buy_limit = 10                # 定义买入股票的排名阈值
    g.intra_industry_count = 2               # 定义组合中同一个行业的股票个数

#设置回测条件
def set_backtest():
    set_option('use_real_price',True)        # 用真实价格交易
    log.set_level('order','error')           # 设置报错等级


def set_trade_freq():
    # 指定每月第一个交易日, 在开盘第一分钟执行
    #run_monthly(monthly, 1, 'open')

    # 指定每周倒数第一个交易日, 在开盘前执行, 此函数中不能下单
    #run_weekly(weekly, -1, 'before_open')
    run_weekly(lh_main,-1,'open')

    # 指定每天收盘后执行, 此函数中不能下单
    #run_daily(daily, 'after_close')
    #run_daily(lh_main,'open')

    # 指定在每天的10:00运行, 必须选择分钟回测, 否则不会执行
    #run_daily(daily, '10:00')

    # 指定在每天的14:00运行, 必须选择分钟回测, 否则不会执行
    #run_daily(daily, '14:00')

'''
================================================================================
每天开盘前
================================================================================
'''
#每天开盘前要做的事情
def before_trading_start(context):

    # 设置手续费
    set_slip_fee(context)

# 根据不同的时间段设置滑点与手续费
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 根据不同的时间段设置手续费
    dt=context.current_dt
    if dt>datetime.datetime(2013,1, 1):
        set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    elif dt>datetime.datetime(2011,1, 1):
        set_commission(PerTrade(buy_cost=0.001, sell_cost=0.002, min_cost=5))
    elif dt>datetime.datetime(2009,1, 1):
        set_commission(PerTrade(buy_cost=0.002, sell_cost=0.003, min_cost=5))
    else:
        set_commission(PerTrade(buy_cost=0.003, sell_cost=0.004, min_cost=5))


'''
================================================================================
交易时
================================================================================
'''
def lh_main(context):

    # 引用 lib
    g.lh = lh_lib()

    # 第一步：行业筛选
    selected_industry_list = g.lh.get_selected_industry_list(context)

    # 第二步：股票筛选
    stock_to_hold_list = g.lh.get_selected_stock_list(context, selected_industry_list)

    # 第三步：构建持仓DF
    df_stock_to_hold = g.lh.build_hold_stock_df(context,stock_to_hold_list)

    # 第四步：风险控制+仓位计算
    df_stock_to_hold = g.lh.calculate_stock_positions(context, df_stock_to_hold)

    # 第五步：执行操作
    g.lh.trade_execution(context,df_stock_to_hold)


################################################################################
################################################################################
class lh_lib():
    # 初始化
    def __init__(self):
        pass

    # 根据旧的行业筛选和新的板块筛选，来更新当前的行业筛选
    def good_perf_industry_list(self, context):
        # 读取行业周次指数
        body = read_file('data/LH/df_ind_all_close_week.csv')
        df_ind_all_close_week = pd.read_csv(StringIO(body), index_col=0)
        df_ind_all_close_week.index = pd.to_datetime(df_ind_all_close_week.index)

        # 计算对数同比序列，周期为12周
        pct_period = 12
        df_ind_all_close_week_pct = np.log(df_ind_all_close_week) - np.log(df_ind_all_close_week.shift(pct_period))
        df_ind_all_close_week_pct = df_ind_all_close_week_pct[pct_period:]

        # 取得距离当日最近的板块排名，数据类型为pd series
        current_date = context.current_dt
        #current_industry_rank = df_ind_all_close_week_pct_rank[df_ind_all_close_week_pct_rank.index<current_date].iloc[-1]
        df_ind_all_close_week_pct = df_ind_all_close_week_pct[df_ind_all_close_week_pct.index<current_date].iloc[-1]
        # 降序排序
        df_ind_all_close_week_pct.sort(axis=0, ascending=False)

        # 取大于12周指数变化率大于0的行业（效果一般，暂不使用）
        #growth_ind_counts = df_ind_all_close_week_pct[df_ind_all_close_week_pct>0].shape[0]

        # 取得排名靠前的行业，一共包括28个行业，2080原则筛选掉最差的20%
        ind_perf_rank_threshold = 22

        if df_ind_all_close_week_pct.shape[0]>ind_perf_rank_threshold:
            selected_industry_list = df_ind_all_close_week_pct[:ind_perf_rank_threshold].index.values.tolist()
        else:
            selected_industry_list = df_ind_all_close_week_pct.index.values.tolist()

        # 类型转换
        selected_industry_list_update = []
        for i in selected_industry_list:
            selected_industry_list_update.append(int(i))
        return selected_industry_list_update

    ############################################################################

    # 第一步：行业筛选
    def get_selected_industry_list(self, context):
        # 剔除业绩差的行业
        selected_industry_list = self.good_perf_industry_list(context)
        return selected_industry_list

    ############################################################################
    ############################################################################

    # 根据行业取得相关的股票代码
    def get_initial_stock_list(self, context, selected_industry_list):
        # 读取股票代码与行业代码的mapping表
        body = read_file('data/LH/stock_industry_info_shenwan1.csv')
        df_ind_stock_mapping = pd.read_csv(StringIO(body), index_col=0)
        # 取得股票列表
        initial_stock_list = df_ind_stock_mapping[df_ind_stock_mapping['industry_code'].isin(selected_industry_list)].index.values

        return initial_stock_list

    # 删除停牌和ST的股票
    def delete_pause_st_stock(self,context,initial_stocks):
        # 判断初始股票池的股票是否停牌和ST，返回list
        paused_info = []
        st_info = []
        current_data = get_current_data()
        for i in initial_stocks:
            paused_info.append(current_data[i].paused)
            st_info.append(current_data[i].is_st)
        df_paused_info = pd.DataFrame({'paused_info':paused_info},index = initial_stocks)
        df_st_info = pd.DataFrame({'st_info':st_info},index = initial_stocks)
        #
        unpaused_stocks =list(df_paused_info.index[df_paused_info.paused_info == False])
        unst_stocks = list(df_st_info.index[df_st_info.st_info == False])
        unsuspened_stocks = list(set(unpaused_stocks) & set(unst_stocks))
        return unsuspened_stocks

    # 删除新股和次新股
    def delete_new_stock(self,context,stock_list,deltaday):
        deltaDate = context.current_dt.date() - dt.timedelta(deltaday)
        stock_list_new = []
        for stock in stock_list:
            if get_security_info(stock).start_date < deltaDate:
                stock_list_new.append(stock)
        return stock_list_new

    # 组成包括价值因子和成长因子的df
    def build_stock_pool(self, context, initial_stock_list):
        # 查询数据
        q_data = query(valuation.code, \
                       valuation.market_cap, \
                       valuation.pe_ratio, \
                       valuation.pb_ratio, \
                       indicator.roe, \
                       indicator.roa, \
                       indicator.inc_revenue_year_on_year, \
                       indicator.inc_net_profit_year_on_year \
                         ).filter(valuation.code.in_(initial_stock_list))
        df_data = get_fundamentals(q_data,context.current_dt.date())
        df_data = df_data.dropna()
        # 重新给columns命名
        df_data.columns = ['code','mkt_cap','pe','pb','roe','roa','inc_rev_yoy','inc_npft_yoy']
        # 插入peg和per
        peg_values = df_data['pe'].values / df_data['inc_npft_yoy'].values
        df_data.insert(df_data.shape[1],'peg',peg_values)
        peg_values = df_data['pe'].values / df_data['inc_rev_yoy'].values
        df_data.insert(df_data.shape[1],'per',peg_values)
        # 将df的index更新为股票代码，并删掉原股票代码列
        df_data.index = df_data['code'].values
        df_data = df_data.drop('code',axis=1)

        return df_data

    # 限制每个行业中的股票数量
    def diversify_by_industry(self,context,df_sort_PEG):
        # 初始化
        industry_list = []
        stock_name_list = []
        ind_name_list = []
        bk_name_list = []
        # 读取数据文件
        body = read_file('data/LH/stock_industry_info_shenwan1.csv')
        # 读取数据的第一列为index
        pd_data = pd.read_csv(StringIO(body), index_col=0)
        # 复制index
        peg_index = df_sort_PEG.index
        # 开始循环处理
        for stock_code in peg_index:
            # 防止行业信息不全
            if stock_code in pd_data.index:
                # 获得该股票的行业代码
                industry_code = pd_data['industry_code'][stock_code]
                #
                stock_name = get_security_info(stock_code).display_name
                industry_name = pd_data['industry_name'][stock_code]
                bankuai_name = pd_data['bankuai_name'][stock_code]
                # 进行判断
                if industry_list.count(industry_code) < g.intra_industry_count:
                    # 将该行业code加入到列表
                    industry_list.append(industry_code)
                    #
                    stock_name_list.append(stock_name)
                    ind_name_list.append(industry_name)
                    bk_name_list.append(bankuai_name)
                else:
                    # 删除该行
                    df_sort_PEG = df_sort_PEG.drop(stock_code)
        #
        df_sort_PEG.insert(0,'bankuai_name',bk_name_list)
        df_sort_PEG.insert(0,'industry_name',ind_name_list)
        df_sort_PEG.insert(0,'stock_name',stock_name_list)
        return df_sort_PEG


    ############################################################################

    # 第二步：股票筛选
    def get_selected_stock_list(self, context, selected_industry_list):
        # 取得筛选后的行业所对应的股票列表
        initial_stock_list = self.get_initial_stock_list(context, selected_industry_list)
        # 删除停牌和ST的股票
        initial_stock_list = self.delete_pause_st_stock(context,initial_stock_list)
        # 去除新股和次新股，原来参数为600
        initial_stock_list = self.delete_new_stock(context,initial_stock_list,300)

        # 组成包括各因子的df
        df_stock_pool = self.build_stock_pool(context, initial_stock_list)

        # 初步筛选
        df_stock_pool_selected = df_stock_pool[ (df_stock_pool.pe>0)&
                                                (df_stock_pool.roe>0)&
                                                (df_stock_pool.roa>0)&
                                                (df_stock_pool.inc_rev_yoy>0)&
                                                (df_stock_pool.inc_npft_yoy>0)
                                                ]

        # 限制因子的范围
        df_stock_pool_selected = df_stock_pool_selected[(df_stock_pool_selected.peg<0.5)
                                                        &(df_stock_pool_selected.inc_rev_yoy<150)
                                                        &(df_stock_pool_selected.inc_npft_yoy<150)
                                                        &(df_stock_pool_selected.mkt_cap<1500)
                                                        ]
        # 注意： 做了行业筛选后，在大盘牛市期间会限制策略的增长，因为牛市期间越是前期被低谷的股票，越容易被拉升增长

        # 排序
        df_stock_pool_selected = df_stock_pool_selected.sort(columns='peg', axis=0, ascending=True)
        #df_stock_pool_selected = df_stock_pool_selected.sort(columns='mkt_cap', axis=0, ascending=True)

        # 增加行业限制，并追加股票名称、行业名称、板块名称
        df_stock_pool_selected = self.diversify_by_industry(context,df_stock_pool_selected)

        # 限制个数
        if df_stock_pool_selected.shape[0] > g.stock_to_buy_limit:
            df_stock_pool_selected = df_stock_pool_selected[:g.stock_to_buy_limit]

        print df_stock_pool_selected

        # 输出清单
        stock_to_hold_list = df_stock_pool_selected.index.values

        return stock_to_hold_list

    ############################################################################
    ############################################################################

    def define_stock_to_hold_df(self):
        # 定义持仓分析DF
        df_stock_to_hold = pd.DataFrame(columns=['Date'
                                                ,'StockCode'
                                                ,'StockName'
                                                ,'Hold' #True or False
                                                ,'TradeMethod' #BUY,SELL,NA,TBD
                                                ,'OriginalAmount'
                                                ,'OriginalValue'
                                                ,'Risk'
                                                ,'RiskPosition'
                                                ,'RiskPosPercent'
                                                ,'AdjRiskPosPercent'
                                                ,'TargetValue'
                                                ,'FinalHoldAmount'
                                                ])
        return df_stock_to_hold

    def add_stock_to_hold_df(self,df_stock_to_hold,current_date,stock_code,stock_name,tmp_Hold,
                             tmp_TradeMethod,
                             tmp_OriginalAmount,tmp_OriginalValue,
                             tmp_Risk,tmp_RiskPosition,tmp_RiskPosPercent,tmp_AdjRiskPosPercent,
                             tmp_TargetValue,tmp_FinalHoldAmount):
        df_stock_to_hold_tmp = pd.DataFrame([[current_date
                                             ,stock_code
                                             ,stock_name
                                             ,tmp_Hold
                                             ,tmp_TradeMethod
                                             ,tmp_OriginalAmount
                                             ,tmp_OriginalValue
                                             ,tmp_Risk
                                             ,tmp_RiskPosition
                                             ,tmp_RiskPosPercent
                                             ,tmp_AdjRiskPosPercent
                                             ,tmp_TargetValue
                                             ,tmp_FinalHoldAmount]],
                                             columns=df_stock_to_hold.columns)
        df_stock_to_hold = pd.concat([df_stock_to_hold,df_stock_to_hold_tmp])
        df_stock_to_hold.index = np.arange(0,df_stock_to_hold.shape[0],1)

        return df_stock_to_hold

    def add_current_position_to_hold_df(self,context,stock_to_hold_list,df_stock_to_hold):

        # 取得当前的持仓信息
        current_positions = context.portfolio.long_positions
        current_date = context.current_dt.date()

        for stock_code in current_positions.keys():
            # 取得当前的仓位
            tmp_StockName = get_security_info(stock_code).display_name
            tmp_OriginalAmount = current_positions[stock_code].total_amount
            tmp_OriginalValue = current_positions[stock_code].value
            # 判断是否在ToHoldList中
            # 在 则Hold为True，TradeMethod为TBD
            # 不在 则Hold为False，且TradeMode为Sell
            if stock_code in stock_to_hold_list:
                # 如果在待买入清单中，则加入到df
                tmp_Hold = True
                tmp_TradeMethod = 'NA'
            else:
                # 判断是否停牌或者涨跌停
                if (lhsf.get_stock_pause_info(stock_code)==True)or(lhsf.get_stock_zhangdieting_info(stock_code)==True):
                    tmp_Hold = True
                    tmp_TradeMethod = 'NA'
                else:
                    tmp_Hold = False
                    tmp_TradeMethod = 'SELL'
            # 将股票信息加入到
            df_stock_to_hold = self.add_stock_to_hold_df(df_stock_to_hold
                                                        ,current_date
                                                        ,stock_code
                                                        ,tmp_StockName
                                                        ,tmp_Hold
                                                        ,tmp_TradeMethod
                                                        ,tmp_OriginalAmount
                                                        ,tmp_OriginalValue
                                                        ,0.0
                                                        ,0.0
                                                        ,0.0
                                                        ,0.0
                                                        ,0.0
                                                        ,0.0
                                                        )

        return df_stock_to_hold

    def add_new_stock_to_hold_df(self,context,stock_to_hold_list,df_stock_to_hold,stock_to_buy_limit):
        current_date = context.current_dt.date()
        available_position_count = stock_to_buy_limit - df_stock_to_hold[df_stock_to_hold['Hold']==True].shape[0]

        tmp_count = 0
        tmp_index = 0

        while (tmp_count < available_position_count)and(tmp_index < len(stock_to_hold_list)):
            stock_code = stock_to_hold_list[tmp_index]
            if stock_code not in df_stock_to_hold['StockCode'].values:

                tmp_Hold = True
                tmp_TradeMethod = 'BUY'
                tmp_StockName = get_security_info(stock_code).display_name
                df_stock_to_hold = self.add_stock_to_hold_df(df_stock_to_hold,current_date,stock_code,tmp_StockName,tmp_Hold,tmp_TradeMethod
                                                            ,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
                tmp_count = tmp_count + 1
                tmp_index = tmp_index + 1
            else:
                tmp_index = tmp_index + 1

        return df_stock_to_hold

    ############################################################################

    # 第三步：构建持仓DF
    def build_hold_stock_df(self,context,stock_to_hold_list):
        # 初始化待输出的
        df_stock_to_hold = self.define_stock_to_hold_df()
        # 将当前的持仓信息写入df
        df_stock_to_hold = self.add_current_position_to_hold_df(context,stock_to_hold_list,df_stock_to_hold)
        #
        df_stock_to_hold = self.add_new_stock_to_hold_df(context,stock_to_hold_list,df_stock_to_hold,g.stock_to_buy_limit)

        return df_stock_to_hold

    ############################################################################
    ############################################################################

    # ES风险计算逻辑
    def cal_stock_risk_ES(self,context,stock_code,risk_lag,confidence_level):
        # 取得历史收盘数据
        #hStocks = history(risk_lag, '1d', 'close', stock_code)
        # 去除停牌期间的数据，避免ES计算的失真
        hStocks = history(risk_lag, '1d', 'close', stock_code, skip_paused=True)
        # 计算日收益率／日变动率
        #dailyReturns = hStocks.resample('D',how='last').pct_change().fillna(value=0, method=None, axis=0).values
        dailyReturns = hStocks.pct_change().fillna(value=0, method=None, axis=0).values
        # 按照升序排序
        dailyReturns_sort =  sorted(dailyReturns)

        # 根据置信水平，计算Alpha。表示基于正态分布函数下的事件发生概率
        if confidence_level   == 1.96:
            alpha = (1 - 0.95)
        elif confidence_level == 2.06:
            alpha = (1 - 0.96)
        elif confidence_level == 2.18:
            alpha = (1 - 0.97)
        elif confidence_level == 2.34:
            alpha = (1 - 0.98)
        elif confidence_level == 2.58:
            alpha = (1 - 0.99)
        elif confidence_level == 5:
            alpha = (1 - 0.99999)
        else:
            alpha = (1 - 0.95)
        # 根据risk_lag(采样期间)和alpha，计算ES的期间
        ES_sample = risk_lag * alpha
        # 初始化
        count = 0
        sum_value = 0
        # 计算ES的和
        for i in range(len(dailyReturns_sort)):
            if i < ES_sample:
                sum_value = sum_value+dailyReturns_sort[i][0]
                count = count+1
        # 计算ES
        if count == 0:
            ES = 0
        else:
            # 增加ceil函数，修正当ES_sample为1.X时，对ES的放大作用
            ES = -(sum_value / ceil(ES_sample))
        if isnan(ES):
            ES = 0
        return ES

    # 更新持仓df中的风险信息，包括：Risk，Position，PositionPercent（含停牌股票），AdjPositionPercent（不含停牌股票）
    def update_stock_risk_info(self,context,df_stock_to_hold):
        # 遍历df，计算ES的Risk和Position
        for i in df_stock_to_hold.index:
            # 对需要持仓的股票进行遍历
            if df_stock_to_hold['Hold'][i]==True:
                #df_stock_to_hold['Risk'][i]=self.cal_stock_risk_ES(context,df_stock_to_hold['StockCode'][i],g.risk_lag,g.confidence_level)
                df_stock_to_hold.loc[i,'Risk']=round(self.cal_stock_risk_ES(context,df_stock_to_hold['StockCode'][i],g.risk_lag,g.confidence_level),4)
                #df_stock_to_hold['RiskPosition'][i] = 1.0 / df_stock_to_hold['Risk'][i]
                df_stock_to_hold.loc[i,'RiskPosition'] = round((1.0 / df_stock_to_hold['Risk'][i]),4)

        # 根据ES值计算持仓比例
        total_position = sum(df_stock_to_hold['RiskPosition'].values)
        adj_total_position = sum(df_stock_to_hold[df_stock_to_hold['TradeMethod']<>'NA']['RiskPosition'].values)
        # 再次遍历df，计算ES的percent和adj percent
        for i in df_stock_to_hold.index:
            if df_stock_to_hold['Hold'][i]==True:
                # 计算ES的position的百分比
                position_percent_tmp = df_stock_to_hold['RiskPosition'][i]/total_position
                if isnan(position_percent_tmp):
                    position_percent_tmp = 0
                # 计算调整后的position的百分比
                adj_position_percent_tmp = df_stock_to_hold['RiskPosition'][i]/adj_total_position
                if isnan(adj_position_percent_tmp):
                    adj_position_percent_tmp = 0

                # 如果该股票已停牌或涨跌停
                if df_stock_to_hold['TradeMethod'][i]=='NA':
                    #df_stock_to_hold['RiskPosPercent'][i] = round(position_percent_tmp,4)
                    df_stock_to_hold.loc[i,'RiskPosPercent'] = round(position_percent_tmp,4)
                else:
                    #df_stock_to_hold['RiskPosPercent'][i] = round(position_percent_tmp,4)
                    df_stock_to_hold.loc[i,'RiskPosPercent'] = round(position_percent_tmp,4)
                    #df_stock_to_hold['AdjRiskPosPercent'][i] = round(adj_position_percent_tmp,4)
                    df_stock_to_hold.loc[i,'AdjRiskPosPercent'] = round(adj_position_percent_tmp,4)

        return df_stock_to_hold

    # 计算个股的Return，周期是参数，可配置为1d，5d等
    def get_stock_return(self,stock_code,freq,lag):
        hStocks = history(lag, freq, 'close', stock_code)
        #dailyReturns = hStocks.resample('D',how='last').pct_change().fillna(value=0, method=None, axis=0).values
        dailyReturns = hStocks.pct_change().fillna(value=0, method=None, axis=0).values
        return dailyReturns

    # 计算股票组合的Return，周期是参数，可配置为1d，5d等
    def get_portfolio_return(self,context,df_stock_to_hold,freq,portfolio_risk_lag):
        # 初始化
        portfolio_Returns = []
        # 遍历df
        for i_index in df_stock_to_hold.index:
            if df_stock_to_hold['Hold'][i_index]==True:
                if df_stock_to_hold['RiskPosPercent'][i_index]<>0.0:
                    stock_Returns = self.get_stock_return(df_stock_to_hold['StockCode'][i_index],freq,portfolio_risk_lag)
                    tmp_list = []
                    for i in range(len(stock_Returns)):
                        tmp_list.append(stock_Returns[i]*df_stock_to_hold['RiskPosPercent'][i_index])
                    # 组合的累加
                    if portfolio_Returns:
                        tmp_list_2 = []
                        for i in range(len(portfolio_Returns)):
                            tmp_list_2.append(portfolio_Returns[i]+tmp_list[i])
                        portfolio_Returns = tmp_list_2
                    else:
                        portfolio_Returns = tmp_list

        return portfolio_Returns

    # 计算股票组合的ES风险系数，周期是参数，可配置为1d，5d等
    def cal_portfolio_risk_ES(self,context,df_stock_to_hold,freq,portfolio_risk_lag,portfolio_confidence_level):
        # 计算投资组合的Daily return
        portfolio_Returns = self.get_portfolio_return(context,df_stock_to_hold,freq,portfolio_risk_lag)
        portfolio_Returns_sort = sorted(portfolio_Returns)

        # 根据置信水平，计算Alpha。表示基于正态分布函数下的事件发生概率
        if portfolio_confidence_level   == 1.96:
            alpha = (1 - 0.95)
        elif portfolio_confidence_level == 2.06:
            alpha = (1 - 0.96)
        elif portfolio_confidence_level == 2.18:
            alpha = (1 - 0.97)
        elif portfolio_confidence_level == 2.34:
            alpha = (1 - 0.98)
        elif portfolio_confidence_level == 2.58:
            alpha = (1 - 0.99)
        elif portfolio_confidence_level == 5:
            alpha = (1 - 0.99999)
        else:
            alpha = (1 - 0.95)
        # 根据risk_lag(采样期间)和alpha，计算ES的期间
        ES_sample = portfolio_risk_lag * alpha
        # 初始化
        count = 0
        sum_value = 0
        # 计算ES的和
        for i in range(len(portfolio_Returns_sort)):
            if i < ES_sample:
                sum_value = sum_value+portfolio_Returns_sort[i][0]
                count = count+1
        # 计算ES
        if count == 0:
            ES = 0
        else:
            # 增加ceil函数，修正当ES_sample为1.X时，对ES的放大作用
            ES = -(sum_value / ceil(ES_sample))
        if isnan(ES):
            ES = 0

        return ES

    # 计算股票组合的VaR风险系数，周期是参数，可配置为1d，5d等
    def cal_portfolio_risk_VaR(self,context,df_stock_to_hold,freq,portfolio_risk_lag,portfolio_confidence_level):
        # 计算投资组合的Daily return
        portfolio_Returns = self.get_portfolio_return(context,df_stock_to_hold,freq,portfolio_risk_lag)
        VaR_ratio = 1.0 * portfolio_confidence_level * np.std(portfolio_Returns)
        return VaR_ratio

    # 根据股票组合的整体风险，计算整理仓位比例，周期是参数，可配置为1d，5d等
    def cal_portfolio_alloc_ratio(self,context,df_stock_to_hold,freq):
        # 计算股票组合的ES风险
        portfolio_ES_ratio = self.cal_portfolio_risk_ES(context,df_stock_to_hold,freq,g.portfolio_risk_lag,g.portfolio_confidence_level)
        # 计算股票组合的VaR风险
        portfolio_VaR_ratio = self.cal_portfolio_risk_VaR(context,df_stock_to_hold,freq,g.portfolio_risk_lag,g.portfolio_confidence_level)

        # 计算风险金
        risk_money = context.portfolio.total_value * g.portfolio_risk_rate
        maxrisk_money = 1.2 * risk_money
        # 计算可承受风险下，投资组合的金额
        portfolio_risk_value_ES = 0
        if portfolio_ES_ratio:
            portfolio_risk_value_ES = maxrisk_money / portfolio_ES_ratio

        portfolio_risk_value_VaR = 0
        if portfolio_VaR_ratio:
            portfolio_risk_value_VaR = risk_money / portfolio_VaR_ratio

        portfolio_risk_value = 0
        if portfolio_risk_value_ES == 0:
            portfolio_risk_value = portfolio_risk_value_VaR
        elif portfolio_risk_value_VaR == 0:
            portfolio_risk_value = portfolio_risk_value_ES
        else:
            portfolio_risk_value = min(portfolio_risk_value_ES,portfolio_risk_value_VaR)

            #log
            #if portfolio_risk_value_ES<portfolio_risk_value_VaR:
            #    print 'ES:',portfolio_ES_ratio
            #else:
            #    print 'VaR:',portfolio_VaR_ratio
        #
        total_value = context.portfolio.total_value
        if portfolio_risk_value>total_value:
            portfolio_alloc_ratio = 1.0
        else:
            portfolio_alloc_ratio = round((portfolio_risk_value/total_value),2)


        return portfolio_alloc_ratio

    # 根据各支股票的持仓比例，计算调仓信息
    def update_to_hold_df(self,context,df_stock_to_hold,portfolio_alloc_ratio):

        # 取得可分配的总金额
        total_capital = context.portfolio.total_value * portfolio_alloc_ratio
        # 取得不能交易部分股票的金额
        unavailable_capital = sum(df_stock_to_hold[df_stock_to_hold['TradeMethod']=='NA']['OriginalValue'].values)
        # 计算可分配的金额
        available_capital = total_capital-unavailable_capital
        # 遍历df，计算TargetValue值
        for i in df_stock_to_hold.index:
            if df_stock_to_hold['Hold'][i]==True:
                if df_stock_to_hold['TradeMethod'][i]<>'NA':
                    #df_stock_to_hold['TargetValue'][i] = available_capital * df_stock_to_hold['TargetValue'][i] * df_stock_to_hold['TrendWeight'][i]
                    df_stock_to_hold.loc[i,'TargetValue'] = round(available_capital * df_stock_to_hold['AdjRiskPosPercent'][i])
                    if df_stock_to_hold['TargetValue'][i] > df_stock_to_hold['OriginalValue'][i]:
                        #df_stock_to_hold['TradeMethod'][i] = 'BUY'
                        df_stock_to_hold.loc[i,'TradeMethod'] = 'BUY'
                    else:
                        #df_stock_to_hold['TradeMethod'][i] = 'SELL'
                        df_stock_to_hold.loc[i,'TradeMethod'] = 'SELL'

        return df_stock_to_hold

    ############################################################################

    # 第四步：风险控制
    def calculate_stock_positions(self,context, df_stock_to_hold):
        # 更新持仓df中的风险信息，包括：Risk，Position，PositionPercent（含停牌股票），AdjPositionPercent（不含停牌股票）
        df_stock_to_hold = self.update_stock_risk_info(context,df_stock_to_hold)
        # 根据各只股票的持仓比例，计算整体风险和整体仓位百分比
        portfolio_alloc_ratio = self.cal_portfolio_alloc_ratio(context,df_stock_to_hold,'1d')
        print 'Portfolio Alloction Ratio',portfolio_alloc_ratio

        # 根据整体仓位比例和各支股票的持仓比例，更新各股的TradeMethod和TargetValue
        df_stock_to_hold = self.update_to_hold_df(context,df_stock_to_hold,portfolio_alloc_ratio)

        return df_stock_to_hold

    ############################################################################
    ############################################################################

    def stock_trade_operation(self,context,TradeMethod,df_stock_to_hold):
        for i in df_stock_to_hold.index:
            if df_stock_to_hold['TradeMethod'][i]==TradeMethod:
                change_percent = (df_stock_to_hold['TargetValue'][i]-df_stock_to_hold['OriginalValue'][i])/df_stock_to_hold['OriginalValue'][i]
                # 如果调整值超过10%则交易：
                if abs(change_percent)>0.1:
                    order_target_value(df_stock_to_hold['StockCode'][i], df_stock_to_hold['TargetValue'][i])

    def update_n_save_hold_df(self,context,df_stock_to_hold):
        # 取得当前的持仓信息
        current_positions = context.portfolio.long_positions
        # 遍历df
        for i in df_stock_to_hold.index:
            if df_stock_to_hold['StockCode'][i] in current_positions.keys():
                #df_stock_to_hold['FinalHoldAmount'][i] = current_positions[df_stock_to_hold['StockCode'][i]].total_amount
                df_stock_to_hold.loc[i,'FinalHoldAmount'] = current_positions[df_stock_to_hold['StockCode'][i]].total_amount
        # 追加行业和板块信息
        body = read_file('data/LH/stock_industry_info_shenwan1.csv')
        df_ind_stock_mapping = pd.read_csv(StringIO(body), index_col=0)
        # 取得股票列表
        ind_name_list = df_ind_stock_mapping.loc[df_stock_to_hold['StockCode'].values]['industry_name'].values
        bk_name_list = df_ind_stock_mapping.loc[df_stock_to_hold['StockCode'].values]['bankuai_name'].values
        # 追加
        df_stock_to_hold.insert(3,'bankuai',bk_name_list)
        df_stock_to_hold.insert(3,'industry',ind_name_list)

        # 输出df
        write_file('data/LH/stock_hold_history.csv', df_stock_to_hold.to_csv(index=True), append=True)

    ############################################################################

    # 第五步：实际操作
    def trade_execution(self,context,df_stock_to_hold):
        # 执行交易
        self.stock_trade_operation(context,'SELL',df_stock_to_hold)
        self.stock_trade_operation(context,'BUY',df_stock_to_hold)
        # 更新df
        self.update_n_save_hold_df(context,df_stock_to_hold)

        #print df_stock_to_hold

    ############################################################################
    ############################################################################

################################################################################
################################################################################


'''
================================================================================
每天收盘后
================================================================================
'''
# 每天收盘后做的事情
# 进行长运算（本策略中不需要）
def after_trading_end(context):

    return
