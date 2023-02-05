import pandas as pd
import numpy as np
from talib import abstract as taf
from datetime import datetime as dt, timedelta
import os
import glob
import requests
import empyrical
from sklearn.model_selection import ParameterGrid


def downsample_OHLC(_df, _interval_min="2min"):
    if "volume" in _df.columns:
        ohlc = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    else:
        ohlc = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }
    _df = _df.resample(_interval_min, offset=_interval_min).apply(ohlc)
    _df = _df.dropna()
    return _df


def candle_cal(_df, _series=False):

    _out = taf.CDLHAMMER(_df) + taf.CDLSHOOTINGSTAR(_df)
    """
    taf.CDLHARAMI(_df) + taf.CDLHAMMER(_df) + taf.CDLENGULFING(_df) + \
    taf.CDLSHOOTINGSTAR(_df) + taf.CDLMARUBOZU(_df)
    """
    _out[_out > 0] = 1
    _out[_out < 0] = -1
    if not _series:
        _out = _out.iloc[-1]
        _out = int(_out)
    return _out


def RSI(_df, _timeperiod=14, _low=40, _up=60, _series=False):
    _out = taf.RSI(_df, timeperiod=_timeperiod)

    _out[_out >= _up] = 1
    _out[_out <= _low] = -1
    _out[(_out < _up) & (_out > _low)] = 0
    if not _series:
        _out = _out.iloc[-1]
        _out = int(_out)
    return _out


def create_heikenashi(_df, overwrite=False):
    _df_temp = _df.copy()
    _df_temp["close_h"] = (
        _df_temp["open"] + _df_temp["high"] + _df_temp["low"] + _df_temp["close"]
    ) / 4
    _df_temp["open_h"] = _df_temp["open"]
    #     _df_temp['open_h'] = 0
    for _i in range(len(_df_temp)):
        if _i == 0:
            _df_temp["open_h"][_i] = _df_temp["open"][_i]

        else:
            _df_temp["open_h"][_i] = (
                _df_temp["open_h"][_i - 1] + _df_temp["close_h"][_i - 1]
            ) / 2

    _df_temp["high_h"] = _df_temp[["open_h", "close_h", "high"]].max(axis=1)
    _df_temp["low_h"] = _df_temp[["open_h", "close_h", "low"]].min(axis=1)
    _df_temp = _df_temp.round(2)
    if overwrite:
        _df_temp = _df_temp.drop(["high", "open", "low", "close"], axis=1)
        _df_temp = _df_temp.rename(
            {"high_h": "high", "open_h": "open", "low_h": "low", "close_h": "close"},
            axis=1,
        )
    return _df_temp


def OHLC_to_tick(_df, _interval_min=2):
    _df_temp = _df.copy()
    _df_tick = pd.DataFrame()
    _df_open = _df[["open"]]
    _df_close = _df[["close"]].shift(1)
    _df_close.index = _df_close.index - pd.Timedelta(minutes=1)
    _df_tick = pd.concat(
        [
            _df_open.rename({"open": "tick"}, axis=1),
            _df_close.rename({"close": "tick"}, axis=1),
        ],
        axis=0,
    )
    _df_tick = _df_tick.sort_index()
    return _df_tick


def pivot_location(_open, _close, _ser, _get_level=False):
    _out = False
    _open, _close = min(_open, _close), max(_open, _close)
    if (
        (_open < _ser.min() and _close > _ser.max())
        or (_open > _ser.max() and _close > _ser.max())
        or (_open < _ser.min() and _close < _ser.min())
    ):
        out = False

    elif _open <= _ser.min():
        if _ser[_ser <= _close].max() == _ser.min():
            _out = True
            _level = _ser.min()

    elif _close >= _ser.max():
        if _ser[_ser >= _open].min() == _ser.max():
            _out = True
            _level = _ser.max()

    elif (_open >= _ser.min() and _close >= _ser.min()) and (
        _open <= _ser.max() and _close <= _ser.max()
    ):
        if _ser[_ser >= _open].min() == _ser[_ser <= _close].max():
            _out = True
            _level = _ser[_ser >= _open].min()

    if _get_level and _out:
        return _out, _level

    else:
        return _out


def stoch_RSI(
    _df, _rsi_period=14, _stoch_period=14, _smooth_k=3, _smooth_d=3, _close="close"
):
    _df = _df.copy()
    _df["RSI"] = taf.RSI(_df[_close], timeperiod=_rsi_period)

    _df["Stoch"] = (_df["RSI"] - _df["RSI"].rolling(_stoch_period).min()).div(
        _df["RSI"].rolling(_stoch_period).max()
        - _df["RSI"].rolling(_stoch_period).min()
    )
    _df["Stoch_k"] = _df["Stoch"].rolling(_smooth_k).mean() * 100
    _df["Stoch_d"] = _df["Stoch_k"].rolling(_smooth_d).mean()
    return _df["Stoch_k"].values, _df["Stoch_d"].values


def tr(data):
    data["previous_close"] = data["close"].shift(1)
    data["high-low"] = abs(data["high"] - data["low"])
    data["high-pc"] = abs(data["high"] - data["previous_close"])
    data["low-pc"] = abs(data["low"] - data["previous_close"])

    tr = data[["high-low", "high-pc", "low-pc"]].max(axis=1)

    return tr


def atr(data, period):
    data["tr"] = tr(data)
    atr = data["tr"].rolling(period).mean()

    return atr


def supertrend(_df, _lookback, _multiplier):

    # ATR
    _high = _df.high
    _low = _df.low
    _close = _df.close
    tr1 = pd.DataFrame(_high - _low)
    tr2 = pd.DataFrame(abs(_high - _close.shift(1)))
    tr3 = pd.DataFrame(abs(_low - _close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join="inner").max(axis=1)
    atr = tr.ewm(_lookback).mean()

    # H/L AVG AND BASIC UPPER & LOWER BAND

    hl_avg = (_high + _low) / 2
    upper_band = (hl_avg + _multiplier * atr).dropna()
    _lower_band = (hl_avg - _multiplier * atr).dropna()

    # FINAL UPPER BAND
    final_bands = pd.DataFrame(columns=["upper", "_lower"])
    final_bands.iloc[:, 0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:, 1] = final_bands.iloc[:, 0]
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i - 1, 0]) | (
                _close[i - 1] > final_bands.iloc[i - 1, 0]
            ):
                final_bands.iloc[i, 0] = upper_band[i]
            else:
                final_bands.iloc[i, 0] = final_bands.iloc[i - 1, 0]

    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (_lower_band[i] > final_bands.iloc[i - 1, 1]) | (
                _close[i - 1] < final_bands.iloc[i - 1, 1]
            ):
                final_bands.iloc[i, 1] = _lower_band[i]
            else:
                final_bands.iloc[i, 1] = final_bands.iloc[i - 1, 1]

    _supertrend = pd.DataFrame(columns=[f"_supertrend_{_lookback}"])
    _supertrend.iloc[:, 0] = [x for x in final_bands["upper"] - final_bands["upper"]]

    for i in range(len(_supertrend)):
        if i == 0:
            _supertrend.iloc[i, 0] = 0
        elif (
            _supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0]
            and _close[i] < final_bands.iloc[i, 0]
        ):
            _supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif (
            _supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0]
            and _close[i] > final_bands.iloc[i, 0]
        ):
            _supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif (
            _supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1]
            and _close[i] > final_bands.iloc[i, 1]
        ):
            _supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif (
            _supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1]
            and _close[i] < final_bands.iloc[i, 1]
        ):
            _supertrend.iloc[i, 0] = final_bands.iloc[i, 0]

    _supertrend = _supertrend.set_index(upper_band.index)
    _supertrend = _supertrend.dropna()[1:]

    # ST UPTREND/DOWNTREND

    upt = []
    dt = []
    _close = _close.iloc[len(_close) - len(_supertrend) :]

    for i in range(len(_supertrend)):
        if _close[i] > _supertrend.iloc[i, 0]:
            upt.append(_supertrend.iloc[i, 0])
            dt.append(np.nan)
        elif _close[i] < _supertrend.iloc[i, 0]:
            upt.append(np.nan)
            dt.append(_supertrend.iloc[i, 0])
        else:
            upt.append(np.nan)
            dt.append(np.nan)

    st, upt, dt = pd.Series(_supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
    upt.index, dt.index = _supertrend.index, _supertrend.index
    _df["supertrend"] = st
    _df["upt"] = upt
    _df["signal"] = True
    _df["signal"][_df.upt.isnull()] = False
    return _df["supertrend"], _df["signal"]


def atr_tss(_df, _lookback, _multiplier):

    # ATR
    _high = _df.high
    _low = _df.low
    _close = _df.close
    tr1 = pd.DataFrame(_high - _low)
    tr2 = pd.DataFrame(abs(_high - _close.shift(1)))
    tr3 = pd.DataFrame(abs(_low - _close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join="inner").max(axis=1)
    atr = tr.ewm(_lookback).mean()

    # H/L AVG AND BASIC UPPER & LOWER BAND

    # hl_avg = (_high + _low) / 2
    hl_avg = _close
    upper_band = (hl_avg + _multiplier * atr).dropna()
    _lower_band = (hl_avg - _multiplier * atr).dropna()

    # FINAL UPPER BAND
    final_bands = pd.DataFrame(columns=["upper", "_lower"])
    final_bands.iloc[:, 0] = [x for x in upper_band - upper_band]
    final_bands.iloc[:, 1] = final_bands.iloc[:, 0]
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i - 1, 0]) | (
                _close[i - 1] > final_bands.iloc[i - 1, 0]
            ):
                final_bands.iloc[i, 0] = upper_band[i]
            else:
                final_bands.iloc[i, 0] = final_bands.iloc[i - 1, 0]

    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (_lower_band[i] > final_bands.iloc[i - 1, 1]) | (
                _close[i - 1] < final_bands.iloc[i - 1, 1]
            ):
                final_bands.iloc[i, 1] = _lower_band[i]
            else:
                final_bands.iloc[i, 1] = final_bands.iloc[i - 1, 1]

    _supertrend = pd.DataFrame(columns=[f"_supertrend_{_lookback}"])
    _supertrend.iloc[:, 0] = [x for x in final_bands["upper"] - final_bands["upper"]]

    for i in range(len(_supertrend)):
        if i == 0:
            _supertrend.iloc[i, 0] = 0
        elif (
            _supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0]
            and _close[i] < final_bands.iloc[i, 0]
        ):
            _supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif (
            _supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 0]
            and _close[i] > final_bands.iloc[i, 0]
        ):
            _supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif (
            _supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1]
            and _close[i] > final_bands.iloc[i, 1]
        ):
            _supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif (
            _supertrend.iloc[i - 1, 0] == final_bands.iloc[i - 1, 1]
            and _close[i] < final_bands.iloc[i, 1]
        ):
            _supertrend.iloc[i, 0] = final_bands.iloc[i, 0]

    _supertrend = _supertrend.set_index(upper_band.index)
    _supertrend = _supertrend.dropna()[1:]

    # ST UPTREND/DOWNTREND

    upt = []
    dt = []
    _close = _close.iloc[len(_close) - len(_supertrend) :]

    for i in range(len(_supertrend)):
        if _close[i] > _supertrend.iloc[i, 0]:
            upt.append(_supertrend.iloc[i, 0])
            dt.append(np.nan)
        elif _close[i] < _supertrend.iloc[i, 0]:
            upt.append(np.nan)
            dt.append(_supertrend.iloc[i, 0])
        else:
            upt.append(np.nan)
            dt.append(np.nan)

    st, upt, dt = pd.Series(_supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
    upt.index, dt.index = _supertrend.index, _supertrend.index
    _df["atr_tss"] = st
    _df["upt"] = upt
    _df["signal"] = True
    _df["signal"][_df.upt.isnull()] = False
    return _df["atr_tss"], _df["signal"]


def get_ltp(_token, _kite):
    _ltps = _kite.ltp(str(_token))
    _ltp = _ltps[str(_token)]["last_price"]
    return _ltp


def write_signal(_str, _filename="auto"):
    if _filename == "auto":
        # _filename = 'signal_file.{}'.format(dt.now().minute*60 + dt.now().second)
        _filename = "signal_file.fractal_21MA.log"
    # print(dt.now().minute*60 + dt.now().second, _filename)
    with open(_filename, "w") as f:
        f.write(_str)


def send_msg(_msg, _api_urls):
    # global api_url
    for _api_url in _api_urls:
        text_msg = '{}"{}"'.format(_api_url, _msg)
        requests.get(text_msg)


def filter_option(
    _kite, _segment="CE", _vol_lim=1000, _price_lim_l=150, _price_lim_u=500
):
    _ltp = fm.get_ltp(260105)
    global option_info, his_start_time, his_end_time
    _option_info = option_info.copy()
    _option_info = _option_info[_option_info.instrument_type == _segment]

    _option_info["diff"] = (_option_info["strike"] - _ltp).abs()
    _option_info = _option_info[_option_info["diff"] < 500]
    _option_info = _option_info.sort_values(by="diff")

    _final_list = []
    _out = "NoTrade"
    for i_opt, _option in _option_info.iterrows():
        # print(_option.instrument_token, his_start_time, his_end_time, interval)
        if _segment == "CE":
            if _option.strike > _ltp:

                _df_his = _kite.historical_data(
                    _option.instrument_token,
                    from_date=his_start_time,
                    to_date=his_end_time,
                    interval=interval,
                )
                if len(_df_his) > 0:
                    _df_his = _df_his[-1]
                    if (
                        _df_his["volume"] > _vol_lim
                        and _df_his["close"] > _price_lim_l
                        and _df_his["close"] < _price_lim_u
                    ):
                        _out = _option.name
                        break

        if _segment == "PE":
            if _option.strike < _ltp:

                _df_his = _kite.historical_data(
                    _option.instrument_token,
                    from_date=his_start_time,
                    to_date=his_end_time,
                    interval=interval,
                )
                if len(_df_his) > 0:
                    _df_his = _df_his[-1]
                    if (
                        _df_his["volume"] > _vol_lim
                        and _df_his["close"] > _price_lim_l
                        and _df_his["close"] < _price_lim_u
                    ):
                        _out = _option.name
                        break
    return _out


def ITM(_price, _div=100, _segment="CE"):
    if _segment == "CE":
        _price = _price // _div * _div + _div
    if _segment == "PE":
        _price = _price // _div * _div
    return _price


def log_write(_log_file, _input_string):
    _input_string = "\n" + _input_string
    with open(_log_file, "a") as f:
        f.write(_input_string)


def calc_slope(x):
    _slope = np.polyfit(range(len(x)), x, 1)[0]
    _slope = np.arctan(_slope) / np.pi * 180
    return _slope


def ma_slope(_df, _slope_on="close", _slope_period=15):

    _df = _df.copy()

    _dfslope = _df[_slope_on].rolling(window=_slope_period).apply(calc_slope)
    return _dfslope


def trading_day(_date):
    # LIST HOLIDAYS FROM NSE https://www1.nseindia.com/products/content/equities/equities/mrkt_timing_holidays.htm
    if _date == None:
        _date = dt.today()
    holidays = [
        "2022-01-26",
        "2022-03-01",
        "2022-03-18",
        "2022-04-14",
        "2022-04-15",
        "2022-05-03",
        "2022-08-09",
        "2022-08-15",
        "2022-08-31",
        "2022-10-05",
        "2022-10-24",
        "2022-10-26",
        "2022-11-08",
    ]

    holidays = [dt.strptime(x, "%Y-%m-%d") for x in holidays]

    if _date.weekday() in [6, 7] or _date in holidays:

        return False

    return True


def pivot_calc(_df, _open, _close):
    P = _df["high"] + _df["low"] + _df["close"]
    R1 = (0.382 * (_df["high"] - _df["low"])) + P
    R2 = (0.618 * (_df["high"] - _df["low"])) + P
    R3 = (1 * (_df["high"] - _df["low"])) + P

    S1 = -(0.382 * (_df["high"] - _df["low"])) + P
    S2 = -(0.618 * (_df["high"] - _df["low"])) + P
    S3 = -(1 * (_df["high"] - _df["low"])) + P
    pivot_location


def zdh_to_ao_convert(_symbol, _instruments, _instrument_ao, _exchange="NSE"):
    _instruments = _instruments[_instruments.exchange == _exchange]
    _insz = _instruments.loc[_symbol]
    _instrument_ao[_instrument_ao.token == str(_insz.exchange_token)]
    _insz_ao = _instrument_ao[_instrument_ao.token == str(_insz.exchange_token)]
    _ao_symbol = _insz_ao.symbol.iloc[0]
    _ao_token = _insz.exchange_token
    return _ao_token, _ao_symbol


def fractal_calc_4(_df):
    _df["frach_con_1"] = np.sign(_df["high"].shift(0) - _df["high"].shift(1))
    _df["frach_con_2"] = np.sign(_df["high"].shift(1) - _df["high"].shift(2))
    _df["frach_con_3"] = np.sign(_df["high"].shift(3) - _df["high"].shift(2))
    _df["frach_con_4"] = np.sign(_df["high"].shift(4) - _df["high"].shift(3))
    _df["fractalh"] = (
        _df["frach_con_1"]
        + _df["frach_con_2"]
        + _df["frach_con_3"]
        + _df["frach_con_4"]
    )

    _df["fracl_con_1"] = np.sign(_df["low"].shift(0) - _df["low"].shift(1))
    _df["fracl_con_2"] = np.sign(_df["low"].shift(1) - _df["low"].shift(2))
    _df["fracl_con_3"] = np.sign(_df["low"].shift(3) - _df["low"].shift(2))
    _df["fracl_con_4"] = np.sign(_df["low"].shift(4) - _df["low"].shift(3))
    _df["fractall"] = (
        _df["fracl_con_1"]
        + _df["fracl_con_2"]
        + _df["fracl_con_3"]
        + _df["fracl_con_4"]
    )

    _df["fractalh"][_df["fractalh"] != -4] = float("nan")
    _df["fractalh"][_df["fractalh"] == -4] = 1

    _df["fractalh"] = _df["fractalh"] * _df["high"].shift(2)
    _df["fractalh"] = _df["fractalh"].fillna(method="ffill")

    _df["fractall"][_df["fractall"] != 4] = float("nan")
    _df["fractall"][_df["fractall"] == 4] = 1

    _df["fractall"] = _df["fractall"] * _df["low"].shift(2)
    _df["fractall"] = _df["fractall"].fillna(method="ffill")

    return _df["fractalh"], _df["fractall"]


def fractal_calc_2(_df):
    _df["frach_con_1"] = np.sign(_df["high"].shift(0) - _df["high"].shift(1))
    _df["frach_con_2"] = np.sign(_df["high"].shift(2) - _df["high"].shift(1))
    _df["fractalh"] = _df["frach_con_1"] + _df["frach_con_2"]

    _df["fracl_con_1"] = np.sign(_df["low"].shift(0) - _df["low"].shift(1))
    _df["fracl_con_2"] = np.sign(_df["low"].shift(2) - _df["low"].shift(1))
    _df["fractall"] = _df["fracl_con_1"] + _df["fracl_con_2"]

    _df["fractalh"][_df["fractalh"] != -2] = float("nan")
    _df["fractalh"][_df["fractalh"] == -2] = 1

    _df["fractalh"] = _df["fractalh"] * _df["high"].shift(1)
    _df["fractalh"] = _df["fractalh"].fillna(method="ffill")

    _df["fractall"][_df["fractall"] != 2] = float("nan")
    _df["fractall"][_df["fractall"] == 2] = 1

    _df["fractall"] = _df["fractall"] * _df["low"].shift(1)
    _df["fractall"] = _df["fractall"].fillna(method="ffill")

    return _df["fractalh"], _df["fractall"]


def maximum_drawdown(returns):

    cum_returns = (1 + returns).cumprod()
    drawdown = 1 - cum_returns.div(cum_returns.cummax())
    return max(drawdown)


def average_drawdown(returns):

    cum_returns = (1 + returns).cumprod()
    drawdown = 1 - cum_returns.div(cum_returns.cummax())
    return np.mean(drawdown)


def get_streak_info(profit_loss_list):
    profit_streak_list = []
    loss_streak_list = []
    loss_count = 0
    profit_count = 0
    for i in profit_loss_list:
        if i == 0:
            if profit_count != 0:
                profit_streak_list.append(profit_count)
            profit_count = 0
            loss_flag = True
            profit_flag = False
            if loss_flag:
                loss_count += 1
        elif i == 1:
            if loss_count != 0:
                loss_streak_list.append(loss_count)
            loss_count = 0
            loss_flag = False
            profit_flag = True
            if profit_flag:
                profit_count += 1
    if len(profit_streak_list) >= 2 or len(loss_streak_list) >= 2:
        maximum_profit_streak = np.max(profit_streak_list)
        average_profit_streak = np.mean(profit_streak_list)
        maximum_loss_streak = np.max(loss_streak_list)
        average_loss_streak = np.mean(loss_streak_list)
    else:
        maximum_profit_streak = 0
        average_profit_streak = 0
        maximum_loss_streak = 0
        average_loss_streak = 0
    return (
        maximum_profit_streak,
        average_profit_streak,
        maximum_loss_streak,
        average_loss_streak,
    )


def create_algo_analytics(data, lot_size=1, initial_investment=1000000, multiplier=1):
    backtesting_metric = {}
    if len(data) > 2:
        data = data.dropna()
        if "profit" not in data.columns:
            data["profit"] = data.apply(
                lambda x: x["exit_price"] - x["entry_price"]
                if x["view"] == "long"
                else -1 * (x["exit_price"] - x["entry_price"]),
                axis=1,
            )
        if "pft_percent" not in data.columns:
            data["pft_percent"] = data.profit / data.entry_price * 100
        holding_period = pd.to_datetime(data["exit_time"]) - pd.to_datetime(
            data["entry_time"]
        )
        capital_growth = initial_investment * (
            1 + (data["profit"] / data["entry_price"]).cumsum()
        )

        backtesting_metric["initial_investment"] = initial_investment
        backtesting_metric["capital_growth"] = capital_growth.iloc[-1]
        data["realized_return"] = (
            100 * data["profit"] / (lot_size * data["entry_price"])
        )
        backtesting_metric["total_return(%)"] = (
            100 * (capital_growth.iloc[-1] - initial_investment) / initial_investment
        )
        backtesting_metric["profitable_trades(%)"] = (
            100 * len(data[data["realized_return"] > 0]) / len(data)
        )
        backtesting_metric["winning_trades"] = len(data[data["realized_return"] > 0])
        backtesting_metric["losing_trades"] = len(data[data["realized_return"] < 0])
        backtesting_metric["sharpe_ratio"] = (
            empyrical.sharpe_ratio(data["realized_return"]) * multiplier
        )
        backtesting_metric["sortino_ratio"] = empyrical.sortino_ratio(
            data["realized_return"]
        )
        backtesting_metric["maximum_drawdown(%)"] = (
            100 * maximum_drawdown(data["realized_return"] / 100) * multiplier
        )
        backtesting_metric["maximum_drawdown"] = (
            data["entry_price"].mean() * backtesting_metric["maximum_drawdown(%)"] / 100
        )
        backtesting_metric["average_drawdown(%)"] = (
            100 * average_drawdown(data["realized_return"] / 100) * multiplier
        )
        backtesting_metric["average_drawdown"] = (
            data["entry_price"].mean() * backtesting_metric["average_drawdown(%)"] / 100
        )

        backtesting_metric["value_at_risk(%)"] = 100 * empyrical.value_at_risk(
            data["realized_return"] / 100, cutoff=0.05
        )
        backtesting_metric[
            "conditional_value_at_risk(%)"
        ] = 100 * empyrical.conditional_value_at_risk(
            data["realized_return"] / 100, cutoff=0.05
        )
        backtesting_metric["average_trade"] = data["profit"].mean()
        backtesting_metric["average_winning_trade"] = data[data["profit"] > 0][
            "profit"
        ].mean()
        backtesting_metric["average_winning_trade_percent"] = (
            data[data["profit"] > 0]["pft_percent"].mean() * 100
        )

        backtesting_metric["average_losing_trade"] = data[data["profit"] < 0][
            "profit"
        ].mean()
        backtesting_metric["average_losing_trade_percent"] = (
            data[data["profit"] < 0]["pft_percent"].mean() * 100
        )

        backtesting_metric["expectancy"] = (
            backtesting_metric["profitable_trades(%)"]
            / 100
            * backtesting_metric["average_winning_trade"]
            - (1 - backtesting_metric["profitable_trades(%)"] / 100)
            * backtesting_metric["average_losing_trade"]
        )

        backtesting_metric["expectancy_percent"] = (
            backtesting_metric["profitable_trades(%)"]
            / 100
            * backtesting_metric["average_winning_trade_percent"]
            - (1 - backtesting_metric["profitable_trades(%)"] / 100)
            * backtesting_metric["average_losing_trade_percent"]
        )

        backtesting_metric["average_risk_reward"] = abs(
            backtesting_metric["average_winning_trade"]
            / backtesting_metric["average_losing_trade"]
        )
        backtesting_metric["profit_factor"] = (
            data[data["profit"] > 0]["profit"].sum()
            / -data[data["profit"] < 0]["profit"].sum()
        )

        backtesting_metric["largest_winning_trade"] = max(
            data[data["profit"] > 0]["profit"]
        )

        backtesting_metric["largest_winning_trade_parcent"] = max(
            data[data["pft_percent"] > 0]["pft_percent"]
        )

        backtesting_metric["largest_losing_trade_parcent"] = min(
            data[data["pft_percent"] < 0]["pft_percent"]
        )
        backtesting_metric["max_carrying_reward"] = max(
            data[data["realized_return"] > 0]["realized_return"]
        )
        backtesting_metric["max_carrying_risk"] = min(
            data[data["realized_return"] < 0]["realized_return"]
        )

        profit_encoding = streak = [1 if i > 0 else 0 for i in data["profit"]]
        (
            maximum_profit_streak,
            average_profit_streak,
            maximum_loss_streak,
            average_loss_streak,
        ) = get_streak_info(profit_encoding)
        backtesting_metric["maximum_profit_streak"] = maximum_profit_streak
        backtesting_metric["average_profit_streak"] = average_profit_streak
        backtesting_metric["maximum_loss_streak"] = maximum_loss_streak
        backtesting_metric["average_loss_streak"] = average_loss_streak
        backtesting_metric["max_holding_period"] = np.max(holding_period)
        backtesting_metric["min_holding_period"] = np.min(holding_period)
    return backtesting_metric


def stochastic(_df, _timeperiod=14, _smoothk=7, _smoothd=3):
    _df["14-high"] = _df["high"].rolling(_timeperiod).max()
    _df["14-low"] = _df["low"].rolling(_timeperiod).min()
    _df["slowK"] = (
        (_df["close"] - _df["14-low"]) * 100 / (_df["14-high"] - _df["14-low"])
    )
    _df["slowK"] = _df["slowK"].rolling(_smoothk).mean()
    _df["slowD"] = _df["slowK"].rolling(_smoothd).mean()
    return _df[["slowK", "slowD"]]


class Backtest:
    def __init__(self):
        self.df = None
        self.func = None
        self.default_param = None
        self.param_dic = None

    def backtest(self, df, func, X=None):
        self.func = func
        self.df = df
        if X != None:
            self.X = X
            self.def_txn = func(df, **X)
        else:
            self.def_txn = func(df)
        return self.def_txn

    def get_def_report(self):
        _output = create_algo_analytics(self.def_txn)
        return _output

    def bruteforce(self, param_dic):
        _final_out = []
        param_grid = ParameterGrid(param_dic)
        for dict_ in param_grid:
            _out = self.func(self.df, **dict_)
            _output = pd.Series()
            if len(_out) > 0:
                _output = create_algo_analytics(_out)
                _output = pd.Series(_output)
            _params = pd.Series(dict_)
            _out = _params.append(_output)
            _out = pd.Series(_out).to_frame().T
            _final_out.append(_out)
        _final_out = pd.concat(_final_out).reset_index()
        return _final_out
