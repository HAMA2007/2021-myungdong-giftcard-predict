import itertools

import argparse as ap
import os
from os import getenv
from types import SimpleNamespace as SimpleNs

import sqlalchemy

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

import plotly.graph_objects as pgo

# --------------------------------------------------- #
# DB FUNCTIONS
# --------------------------------------------------- #
# CREATE DB ENGINE INTERNAL AND RETURN
def create_db_engine_with_pool(url):
    return sqlalchemy.create_engine(
        url,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,  # 30 seconds
        #pool_recycle=600,  # 30 minutes
        pool_pre_ping=True,
    )

# CREATE DB ENGINE FRONT AND RETURN
def create_mysql_db_engine(host, port, username, password, database):
    return create_db_engine_with_pool(sqlalchemy.engine.url.URL.create(
        drivername='mysql+pymysql',
        username=username,
        password=password,
        database=database,
        host=host,
        port=port
    ))

# -------------------------------------------------- #
# FUNCTIONS
# -------------------------------------------------- #
def _load_datas(db_eng):
    '''Load data from DB and return Dataframe.
    '''
    # LOAD PAST DATA(WOOCHEON)
    with db_eng.connect() as dbconn:
        sql = f'''SELECT * FROM woocheon ORDER BY `datetime` DESC'''
        woocheon_raw = dbconn.execute(sql, []).fetchall()
    df_woocheon_raw = pd.DataFrame(woocheon_raw, columns=['datetime', 'ltt50', 'ltt10', 'ssg50', 'ssg10', 'hdi50', 'hdi10', 'ggs10', 'hps10'])
    df_woocheon_raw.set_index(keys='datetime', inplace=True)

    # LOAD PAST DATA(WOOHYUN)
    with db_eng.connect() as dbconn:
        sql = f'''SELECT * FROM woohyun ORDER BY `datetime` DESC'''
        woohyun_raw = dbconn.execute(sql, []).fetchall()
    df_woohyun_raw = pd.DataFrame(woohyun_raw, columns=['datetime', 'ltt50', 'ltt10', 'ssg50', 'ssg10', 'hdi50', 'hdi10', 'ggs10', 'hps10'])
    df_woohyun_raw.set_index(keys='datetime', inplace=True)

    return SimpleNs(woocheon=df_woocheon_raw, woohyun=df_woohyun_raw)

def _clean_datas(raw_datas):
    '''Clean Datas.
    '''
    clean_woocheon_datas = raw_datas.woocheon.resample('5T').bfill()
    clean_woohyun_datas = raw_datas.woohyun.resample('5T').bfill()
    return SimpleNs(woocheon=clean_woocheon_datas, woohyun=clean_woohyun_datas)

def _cut_datas(clean_datas):
    '''Cut Data based on Day.
    '''
    woocheon_temp = clean_datas.woocheon.reset_index()
    woocheon_data_start = (woocheon_temp.iloc[0].loc['datetime']).replace(hour=0, minute=0, second=0, microsecond=0) + pd.Timedelta(days=1)
    woocheon_data_end = (woocheon_temp.iloc[-1].loc['datetime']).replace(hour=0, minute=0, second=0, microsecond=0) - (woocheon_temp.iloc[-1].loc['datetime'] - woocheon_temp.iloc[-2].loc['datetime'])
    cutted_woocheon_datas = clean_datas.woocheon[woocheon_data_start:woocheon_data_end]

    woohyun_temp = clean_datas.woohyun.reset_index()
    woohyun_data_start = (woohyun_temp.iloc[0].loc['datetime']).replace(hour=0, minute=0, second=0, microsecond=0) + pd.Timedelta(days=1)
    woohyun_data_end = (woohyun_temp.iloc[-1].loc['datetime']).replace(hour=0, minute=0, second=0, microsecond=0) - (woohyun_temp.iloc[-1].loc['datetime'] - woohyun_temp.iloc[-2].loc['datetime'])
    cutted_woohyun_datas = clean_datas.woohyun[woohyun_data_start:woohyun_data_end]

    return SimpleNs(woocheon=cutted_woocheon_datas, woohyun=cutted_woohyun_datas)

def _predict_datas(data, days:int):
    '''Predict with fbprophet.
    '''
    class suppress_stdout_stderr(object):
        '''
        source - https://github.com/facebook/prophet/issues/223#issuecomment-326455744
        fbprophet에서 fitting 중 발생되는 INFO Log를 제거하기 위한 목적의 코드

        A context manager for doing a "deep suppression" of stdout and stderr in
        Python, i.e. will suppress all print, even if the print originates in a
        compiled C/Fortran sub-function.
        This will not suppress raised exceptions, since exceptions are printed
        to stderr just before a script exits, and after the context manager has
        exited (at least, I think that is why it lets exceptions through).

        '''
        def __init__(self):
            # Open a pair of null files
            self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
            # Save the actual stdout (1) and stderr (2) file descriptors.
            self.save_fds = [os.dup(1), os.dup(2)]

        def __enter__(self):
            # Assign the null pointers to stdout and stderr.
            os.dup2(self.null_fds[0], 1)
            os.dup2(self.null_fds[1], 2)

        def __exit__(self, *_):
            # Re-assign the real stdout/stderr back to (1) and (2)
            os.dup2(self.save_fds[0], 1)
            os.dup2(self.save_fds[1], 2)
            # Close the null files
            for fd in self.null_fds + self.save_fds:
                os.close(fd)
    
    # 1. fitting 할 데이터를 만드는 곳 (과거 데이터만 존재하여야 함)
    df = pd.DataFrame()
    df["ds"] = data.index
    df["y"] = data.values

    # 2. model을 만드는 곳
    # changepoint_prior_scale -> default가 0.05인데, 숫자를 줄임으로서 유연성 감소
    model = Prophet(
        changepoint_prior_scale=0.5,
        interval_width=0.1)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=4)

    # 3. fitting을 진행
    with suppress_stdout_stderr():
        model.fit(df)

    # cross validation(모델의 하이퍼파라미터가 적합한지 확인용도)
    # df_cv2 = cross_validation(model, initial='150 days', period='90 days', horizon='60 days')
    # print(df_cv2.head())

    # 오차율을 MAPE로 반환해줌(plot도 해줌)
    # df_p = performance_metrics(df_cv2)
    # print(df_p['mape'])
    # fig = plot_cross_validation_metric(df_cv2, metric='mape')
    # fig.savefig('mape.png')


    # 4. 예측할 데이터의 기간(날짜로 조정)
    df_future = model.make_future_dataframe(periods=12*24*days, freq='5T', include_history=False)    

    # 6. 예측하고 싶은 날짜의 데이터를 삽입
    forecast = model.predict(df_future)

    # 7. 원하는 출력된 값
    df_forecast = pd.DataFrame(data=forecast['yhat'].values, index=forecast['ds'])

    return df_forecast

def _round_datas(data):
    index = data.index.tolist()
    datas = list(itertools.chain(*data.values.tolist()))
    for data_index in range(len(datas)):
        if datas[data_index] % 0.05 >= 0.025:
            datas[data_index] = 0.05 * ((datas[data_index] // 0.05) + 1)
        else:
            datas[data_index] = 0.05 * (datas[data_index] // 0.05)
    
    return pd.DataFrame(datas, index=index)

def _compare_diff(data, column1:str, column2:str):
    data1 = data[column1].to_list()
    data2 = data[column2].to_list()
    diff_count = 0
    for i in range(len(data1)):
        if data1[i] != data2[i]:
            diff_count += 1
    print((diff_count/len(data1))*100)

def _plot_datas(data, name, filename):
    '''Plot with plotly.
    '''
    fig = pgo.Figure()
    fig.add_scatter(x=data.index.tolist(), y=list(itertools.chain(*data.values.tolist())), name=f'<b>{name}</b>')
    fig.update_layout(legend=dict(orientation='h', y=1, yanchor='bottom', font=dict(size=17)),
                    yaxis_title='<b>Won</b>')
    fig.update_yaxes(titlefont=dict(size=17))
    fig.update_layout(title=f'<b>{name}</b>')
    fig.write_image(f'./{filename}.png', width=1000, height=500)
    # fig.show()

# -------------------------------------------------- #
# MAIN
# -------------------------------------------------- #
def _run():
    # LOAD ENV
    from dotenv import load_dotenv
    load_dotenv(dotenv_path='./term_project.env')

    # CREATE DB ENGINE
    db_eng = create_mysql_db_engine(
                host=getenv('DB_HOST', None),
                port=getenv('DB_PORT', None),
                username=getenv('DB_USER', None),
                password=getenv('DB_PASS', None),
                database=getenv('DB_NAME', None)
    )

    # 필요 데이터 로드
    raw_datas = _load_datas(db_eng)

    # 데이터 클리닝
    cleaned_datas = _clean_datas(raw_datas)

    # 1일 단위로 데이터 자르기
    cutted_data = _cut_datas(cleaned_datas)

    ##########################################################
    # TEST CODE FOR HYPERPARAMETER TUNING(일단위)
    ##########################################################
    # 평균값으로 일단위 리샘플링
    # ltt50_day = cutted_data.woocheon["ltt50"].to_frame().resample('D').mean()
    # woocheon_ltt50_pdt = _predict_datas(ltt50_day["ltt50"], 60)
    # _plot_datas(woocheon_ltt50_pdt, '우천상품권 롯데백화점 50만원권(예측)', 'woocheon_ltt50')
    # woocheon_ltt50_pdt = _round_datas(woocheon_ltt50_pdt)
    # _plot_datas(woocheon_ltt50_pdt, '우천상품권 롯데백화점 50만원권(Final)', 'woocheon_ltt50_1d_nzd')

    # param_grid = {  
    #     'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    #     'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    # }

    # # Generate all combinations of parameters
    # all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    # rmses = []  # Store the RMSEs for each params here

    # df = pd.DataFrame()
    # df["ds"] = ltt50_day["ltt50"].index
    # df["y"] = ltt50_day["ltt50"].values

    # # Use cross validation to evaluate all parameters
    # for params in all_params:
    #     model = Prophet(**params).fit(df)  # Fit model with given params
    #     df_cv = cross_validation(model, initial='150 days', period='90 days', horizon='60 days', parallel="processes")
    #     df_p = performance_metrics(df_cv, rolling_window=1)
    #     rmses.append(df_p['rmse'].values[0])

    # # Find the best parameters
    # tuning_results = pd.DataFrame(all_params)
    # tuning_results['rmse'] = rmses
    # print(tuning_results)

    #########################################################

    # 10만원권과 50만원권 차이 비교
    _compare_diff(cutted_data.woocheon, 'ltt50', 'ltt10')
    _compare_diff(cutted_data.woocheon, 'ssg50', 'ssg10')
    _compare_diff(cutted_data.woocheon, 'hdi50', 'hdi10')

    # 데이터 CSV추출
    raw_datas.woocheon.to_csv('woocheon_original.csv', sep=',', na_rep='NaN')
    raw_datas.woohyun.to_csv('woohyun_original.csv', sep=',', na_rep='NaN')
    cutted_data.woocheon.to_csv('woocheon_clean.csv', sep=',', na_rep='NaN')
    cutted_data.woohyun.to_csv('woohyun_clean.csv', sep=',', na_rep='NaN')

    # 예측
    # woocheon_ltt50_pdt = _predict_datas(cutted_data.woocheon["ltt50"], 60)
    woocheon_ltt10_pdt = _predict_datas(cutted_data.woocheon["ltt10"], 60)
    # woocheon_ssg50_pdt = _predict_datas(cutted_data.woocheon["ssg50"], 60)
    woocheon_ssg10_pdt = _predict_datas(cutted_data.woocheon["ssg10"], 60)
    # woocheon_hdi50_pdt = _predict_datas(cutted_data.woocheon["hdi50"], 60)
    woocheon_hdi10_pdt = _predict_datas(cutted_data.woocheon["hdi10"], 60)
    # woocheon_ggs10_pdt = _predict_datas(cutted_data.woocheon["ggs10"], 60)
    # woocheon_hps10_pdt = _predict_datas(cutted_data.woocheon["hps10"], 60)

    # 5분단위 예측 결과
    _plot_datas(woocheon_ltt10_pdt, '우천상품권 롯데백화점 10만원권(예측Min)', 'woocheon_ltt10_5m')
    _plot_datas(woocheon_ssg10_pdt, '우천상품권 신세계백화점 10만원권(예측Min)', 'woocheon_ssg10_5m')
    _plot_datas(woocheon_hdi10_pdt, '우천상품권 현대백화점 10만원권(예측Min)', 'woocheon_hdi10_5m')

    # 평균값으로 일단위 리샘플링
    woocheon_ltt10_pdt = woocheon_ltt10_pdt.resample('D').mean()
    woocheon_ssg10_pdt = woocheon_ssg10_pdt.resample('D').mean()
    woocheon_hdi10_pdt = woocheon_hdi10_pdt.resample('D').mean()

    # 1일단위 예측 결과
    _plot_datas(woocheon_ltt10_pdt, '우천상품권 롯데백화점 10만원권(예측Day)', 'woocheon_ltt10_1d')
    _plot_datas(woocheon_ssg10_pdt, '우천상품권 신세계백화점 10만원권(예측Day)', 'woocheon_ssg10_1d')
    _plot_datas(woocheon_hdi10_pdt, '우천상품권 현대백화점 10만원권(예측Day)', 'woocheon_hdi10_1d')

    # 0.05 단위로 반올림한다
    woocheon_ltt10_pdt = _round_datas(woocheon_ltt10_pdt)
    woocheon_ssg10_pdt = _round_datas(woocheon_ssg10_pdt)
    woocheon_hdi10_pdt = _round_datas(woocheon_hdi10_pdt)
    
    # 반올림 결과를 출력한다
    _plot_datas(woocheon_ltt10_pdt, '우천상품권 롯데백화점 10만원권(Final)', 'woocheon_ltt10_1d_nzd')
    _plot_datas(woocheon_ssg10_pdt, '우천상품권 신세계백화점 10만원권(Final)', 'woocheon_ssg10_1d_nzd')
    _plot_datas(woocheon_hdi10_pdt, '우천상품권 현대백화점 10만원권(Final)', 'woocheon_hdi10_1d_nzd')


# -------------------------------------------------- #
# ENTRY POINT
# -------------------------------------------------- #
if __name__ == '__main__':
    _run()