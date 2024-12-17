# asset_allocation.py
import os
import zipfile
import requests
from datetime import datetime
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant

CACHE_DIR = 'data_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_filename(url):
    base_name = url.split('/')[-1]
    return os.path.join(CACHE_DIR, base_name.replace('.zip','_cached.csv'))

def download_and_extract_zip(url, extract_to='temp_folder'):
    cache_file = get_cache_filename(url)
    if os.path.exists(cache_file):
        # Use cached file
        return cache_file
    
    zip_file = 'temp.zip'
    os.makedirs(extract_to, exist_ok=True)
    with open(zip_file, 'wb') as f:
        f.write(requests.get(url).content)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_file)
    extracted_files = os.listdir(extract_to)
    csv_files = [f for f in extracted_files if f.lower().endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the extracted folder.")
    csv_path = os.path.join(extract_to, csv_files[0])
    # Cache the extracted CSV
    os.replace(csv_path, cache_file)
    # Cleanup temp folder
    cleanup_temp_files(extract_to)
    return cache_file

def cleanup_temp_files(folder='temp_folder'):
    if os.path.exists(folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
        os.rmdir(folder)

def process_csv_file(csv_file_path):
    csv_data = pd.read_csv(csv_file_path, skiprows=9, low_memory=False)
    csv_data = csv_data.apply(pd.to_numeric, errors='coerce')
    industry_names = csv_data.columns[1:]
    rows_nan = csv_data[csv_data.iloc[:, 0].isna()].index
    from_row = 0
    until_row = rows_nan[0]
    data = csv_data.iloc[from_row:until_row, :].to_numpy()
    ret = data[:, 1:] / 100
    caldt = pd.to_datetime(data[:, 0].astype(int).astype(str), format='%Y%m%d')
    ret[ret <= -0.99] = np.nan
    return pd.DataFrame(data=ret, index=caldt, columns=industry_names), caldt

def market_french_reconciled(caldt, url):
    csv_file_path = download_and_extract_zip(url)
    csv_data = pd.read_csv(csv_file_path, skiprows=3, header=None, low_memory=False)
    csv_data = csv_data.apply(pd.to_numeric, errors='coerce')
    csv_data = csv_data.dropna(subset=[0])
    caldt_mkt = pd.to_datetime(csv_data.iloc[:, 0].astype(int).astype(str), format='%Y%m%d')
    ret_mkt = (csv_data.iloc[:, 1] + csv_data.iloc[:, 4]) / 100
    y = np.full(len(caldt), np.nan)
    idx = np.where(np.in1d(caldt_mkt, caldt))[0]
    y[np.in1d(caldt, caldt_mkt)] = ret_mkt.iloc[idx]
    return pd.Series(data=y, index=caldt)

def tbill_french_reconciled(caldt, url):
    csv_file_path = download_and_extract_zip(url)
    csv_data = pd.read_csv(csv_file_path, skiprows=3, header=None, low_memory=False)
    csv_data = csv_data.apply(pd.to_numeric, errors='coerce')
    csv_data = csv_data.dropna(subset=[0])
    caldt_tbill = pd.to_datetime(csv_data.iloc[:, 0].astype(int).astype(str), format='%Y%m%d')
    ret_tbill = csv_data.iloc[:, 4] / 100
    y = np.full(len(caldt), np.nan)
    idx = np.where(np.in1d(caldt_tbill, caldt))[0]
    y[np.in1d(caldt, caldt_tbill)] = ret_tbill.iloc[idx]
    return pd.Series(data=y, index=caldt)

def prepare_data():
    INDUSTRY_PORTFOLIOS_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/48_Industry_Portfolios_daily_CSV.zip'
    FRENCH_DATA_FACTORS_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
    csv_file_path = download_and_extract_zip(INDUSTRY_PORTFOLIOS_URL)
    data, caldt = process_csv_file(csv_file_path)
    data['mkt_ret'] = market_french_reconciled(caldt, FRENCH_DATA_FACTORS_URL)
    data['tbill_ret'] = tbill_french_reconciled(caldt, FRENCH_DATA_FACTORS_URL)
    new_columns = [f'ret_{i+1}' for i in range(len(data.columns)-2)] + ['mkt_ret', 'tbill_ret']
    data.columns = new_columns
    data['caldt'] = caldt
    data = data[['caldt'] + new_columns]
    return data

def process_indicators(data, UP_DAY, DOWN_DAY, ADR_VOL_ADJ, KELT_MULT):
    data['caldt'] = pd.to_datetime(data['caldt'])
    data.set_index('caldt', inplace=True)
    num_portfolios = data.shape[1] - 2
    price = (1 + data.iloc[:, :num_portfolios].fillna(0)).cumprod()
    def rolling_vol(df, window):
        return df.rolling(window=window).std(ddof=0)
    def rolling_ema(df, window):
        return df.ewm(span=window, adjust=False).mean()
    def rolling_max(df, window):
        return df.rolling(window=window).max()
    def rolling_min(df, window):
        return df.rolling(window=window).min()
    def rolling_mean(df, window):
        return df.rolling(window=window, min_periods=window-1).mean()
    vol = rolling_vol(data.iloc[:, :num_portfolios], UP_DAY)
    ema_down = rolling_ema(price, DOWN_DAY)
    ema_up = rolling_ema(price, UP_DAY)
    donc_up = rolling_max(price, UP_DAY)
    donc_down = rolling_min(price, DOWN_DAY)
    price_change = price.diff(periods=1).abs()
    kelt_up = ema_up + KELT_MULT * rolling_mean(price_change, UP_DAY)
    kelt_down = ema_down - KELT_MULT * rolling_mean(price_change, DOWN_DAY)
    long_band = pd.DataFrame(np.minimum(donc_up.values, kelt_up.values), index=donc_up.index, columns=donc_up.columns)
    short_band = pd.DataFrame(np.maximum(donc_down.values, kelt_down.values), index=donc_down.index, columns=donc_down.columns)
    long_band_shifted = long_band.shift(1)
    short_band_shifted = short_band.shift(1)
    long_signal = (price >= long_band_shifted) & (long_band_shifted > short_band_shifted)
    indicator_dfs = {f'ret_{i+1}': data.iloc[:, i] for i in range(num_portfolios)}
    indicator_dfs.update({f'price_{i+1}': price.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'vol_{i+1}': vol.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'ema_down_{i+1}': ema_down.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'ema_up_{i+1}': ema_up.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'donc_up_{i+1}': donc_up.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'donc_down_{i+1}': donc_down.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'kelt_up_{i+1}': kelt_up.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'kelt_down_{i+1}': kelt_down.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'long_band_{i+1}': long_band.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'short_band_{i+1}': short_band.iloc[:, i] for i in range(num_portfolios)})
    indicator_dfs.update({f'long_signal_{i+1}': long_signal.iloc[:, i] for i in range(num_portfolios)})
    indicators_df = pd.concat(indicator_dfs.values(), axis=1)
    indicators_df.columns = indicator_dfs.keys()
    indicators_df['mkt_ret'] = data['mkt_ret']
    indicators_df['tbill_ret'] = data['tbill_ret']
    return indicators_df, num_portfolios

def backtest_strategy(indicators_df, num_portfolios, AUM_0, invest_cash, target_vol, max_leverage, max_not_trade):
    T = len(indicators_df['price_1'])
    exposure = np.zeros((T, num_portfolios))
    ind_weight = np.zeros((T, num_portfolios))
    trail_stop_long = np.full((T, num_portfolios), np.nan)
    rets = indicators_df[[f'ret_{j+1}' for j in range(num_portfolios)]].values
    long_signals = indicators_df[[f'long_signal_{j+1}' for j in range(num_portfolios)]].values
    long_bands = indicators_df[[f'long_band_{j+1}' for j in range(num_portfolios)]].values
    short_bands = indicators_df[[f'short_band_{j+1}' for j in range(num_portfolios)]].values
    prices = indicators_df[[f'price_{j+1}' for j in range(num_portfolios)]].values
    vols = indicators_df[[f'vol_{j+1}' for j in range(num_portfolios)]].values
    for t in range(1, T):
        valid_entries = ~np.isnan(rets[t]) & ~np.isnan(long_bands[t])
        prev_exposure = exposure[t - 1]
        current_exposure = exposure[t]
        current_trail_stop = trail_stop_long[t]
        current_long_signals = long_signals[t]
        current_short_bands = short_bands[t]
        current_prices = prices[t]
        current_vols = vols[t]
        new_long_condition = (prev_exposure <= 0) & (current_long_signals == 1)
        confirm_long_condition = (prev_exposure == 1) & (current_prices > np.maximum(trail_stop_long[t - 1], current_short_bands))
        exit_long_condition = (prev_exposure == 1) & (current_prices <= np.maximum(trail_stop_long[t - 1], current_short_bands))
        new_longs = valid_entries & new_long_condition
        current_exposure[new_longs] = 1
        current_trail_stop[new_longs] = current_short_bands[new_longs]
        confirm_longs = valid_entries & confirm_long_condition
        current_exposure[confirm_longs] = 1
        current_trail_stop[confirm_longs] = np.maximum(trail_stop_long[t - 1, confirm_longs], current_short_bands[confirm_longs])
        exit_longs = valid_entries & exit_long_condition
        current_exposure[exit_longs] = 0
        ind_weight[t, exit_longs] = 0
        active_longs = current_exposure == 1
        lev_vol = np.divide(target_vol, current_vols, out=np.zeros_like(current_vols), where=current_vols != 0)
        ind_weight[t, active_longs] = lev_vol[active_longs]
        trail_stop_long[t] = current_trail_stop
        exposure[t] = current_exposure
    new_columns = {}
    for j in range(num_portfolios):
        new_columns[f'exposure_{j+1}'] = exposure[:, j]
        new_columns[f'ind_weight_{j+1}'] = ind_weight[:, j]
        new_columns[f'trail_stop_long_{j+1}'] = trail_stop_long[:, j]
    new_columns_df = pd.DataFrame(new_columns, index=indicators_df.index)
    indicators_df = pd.concat([indicators_df, new_columns_df], axis=1)
    return indicators_df

def aggregate_portfolio_level_analysis(indicators_df, num_portfolios, AUM_0, invest_cash, target_vol, max_leverage, max_not_trade):
    port = pd.DataFrame(index=indicators_df.index)
    port['caldt'] = indicators_df.index
    port['available'] = indicators_df.filter(like='ret_').notna().sum(axis=1)
    ind_weight_df = indicators_df.filter(like='ind_weight_')
    port_weights = ind_weight_df.div(port['available'], axis=0)
    port_weights = port_weights.clip(upper=max_not_trade)
    port['sum_exposure'] = port_weights.sum(axis=1)
    idx_above_max_lev = port[port['sum_exposure'] > max_leverage].index
    port_weights.loc[idx_above_max_lev] = port_weights.loc[idx_above_max_lev].div(
        port['sum_exposure'][idx_above_max_lev], axis=0
    ).mul(max_leverage)
    port['sum_exposure'] = port_weights.sum(axis=1)
    for i in range(num_portfolios):
        port[f'weight_{i+1}'] = port_weights.iloc[:, i]
    ret_long_components = [
        port[f'weight_{i+1}'].shift(1).fillna(0) * indicators_df[f'ret_{i+1}'].fillna(0) for i in range(num_portfolios)
    ]
    port['ret_long'] = sum(ret_long_components)
    port['ret_tbill'] = (1 - port[[f'weight_{i+1}' for i in range(num_portfolios)]].shift(1).sum(axis=1)) * indicators_df['tbill_ret']
    if invest_cash == "YES":
        port['ret_long'] += port['ret_tbill']
    port['AUM'] = AUM_0 * (1 + port['ret_long']).cumprod()
    port['AUM_SPX'] = AUM_0 * (1 + indicators_df['mkt_ret']).cumprod()
    return port

def compute_hit_ratio(returns, timeframe='daily'):
    if timeframe not in ['daily', 'monthly', 'yearly']:
        raise ValueError('Invalid timeframe.')
    if timeframe != 'daily':
        resample_rule = {'monthly': 'MS', 'yearly': 'YS'}[timeframe]
        returns = returns.resample(resample_rule).apply(lambda x: np.prod(1 + x) - 1)
    return round((returns > 0).mean() * 100, 0)

def compute_skewness(returns, timeframe='daily'):
    if timeframe not in ['daily', 'monthly', 'yearly']:
        raise ValueError('Invalid timeframe.')
    if timeframe != 'daily':
        resample_rule = {'monthly': 'MS', 'yearly': 'YS'}[timeframe]
        returns = returns.resample(resample_rule).apply(lambda x: np.prod(1 + x) - 1)
    return round(returns.skew(), 2)

def sortino_ratio(returns):
    downside_risk = returns[returns < 0].std()
    return returns.mean() / downside_risk

def max_drawdown(aum):
    cumulative = np.log1p(aum).cumsum()
    max_cumulative = cumulative.cummax()
    drawdown = np.expm1(cumulative - max_cumulative)
    return drawdown.min()

def table_of_stats(aum, mkt_ret, tbill_ret):
    ret = aum['AUM'].pct_change(fill_method=None).fillna(0)
    stats = {}
    stats['irr'] = round((np.prod(1 + ret) ** (252 / len(ret)) - 1) * 100, 1)
    stats['vol'] = round(ret.std() * np.sqrt(252) * 100, 1)
    stats['sr'] = round(ret.mean() / ret.std() * np.sqrt(252), 2)
    stats['sortino'] = round(sortino_ratio(ret) * np.sqrt(252), 2)
    stats['ir_d'] = compute_hit_ratio(ret, 'daily')
    stats['ir_m'] = compute_hit_ratio(ret, 'monthly')
    stats['ir_y'] = compute_hit_ratio(ret, 'yearly')
    stats['skew_d'] = round(ret.skew(), 2)
    stats['skew_m'] = compute_skewness(ret, 'monthly')
    stats['skew_y'] = compute_skewness(ret, 'yearly')
    stats['mdd'] = round(max_drawdown(aum['AUM']) * 100, 0)
    X = (mkt_ret - tbill_ret).values
    Y = (ret - tbill_ret).values
    valid_idx = ~np.isnan(Y)
    X = add_constant(X[valid_idx])
    Y = Y[valid_idx]
    model = OLS(Y, X).fit()
    intercept, coef = model.params
    stats['alpha'] = round(intercept * 100 * 252, 2)
    stats['beta'] = round(coef, 2)
    stats['worst_ret'] = round(ret.min() * 100, 2)
    stats['worst_day'] = aum['caldt'][ret.idxmin()].strftime('%Y-%m-%d')
    stats['best_ret'] = round(ret.max() * 100, 2)
    stats['best_day'] = aum['caldt'][ret.idxmax()].strftime('%Y-%m-%d')
    return stats

def run_asset_allocation(start_date=None, end_date=None,
                         UP_DAY=20, DOWN_DAY=40, ADR_VOL_ADJ=1.4, KELT_MULT=2.8,
                         AUM_0=1, invest_cash="YES", target_vol=0.015,
                         max_leverage=2, max_not_trade=0.20):
    raw_data = prepare_data()
    if start_date is not None:
        raw_data = raw_data[raw_data['caldt']>=pd.to_datetime(start_date)]
    if end_date is not None:
        raw_data = raw_data[raw_data['caldt']<=pd.to_datetime(end_date)]
    raw_data = raw_data.dropna(subset=['mkt_ret','tbill_ret'])
    indicators_df, num_portfolios = process_indicators(raw_data, UP_DAY, DOWN_DAY, ADR_VOL_ADJ, KELT_MULT)
    indicators_df = backtest_strategy(indicators_df, num_portfolios, AUM_0, invest_cash, target_vol, max_leverage, max_not_trade)
    port = aggregate_portfolio_level_analysis(indicators_df, num_portfolios, AUM_0, invest_cash, target_vol, max_leverage, max_not_trade)
    metrics = table_of_stats(port[['caldt','AUM']], indicators_df['mkt_ret'], indicators_df['tbill_ret'])
    return metrics, port 

if __name__ == "__main__":
    # Example of running:
    # pytho