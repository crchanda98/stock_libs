from datetime import date, timedelta, datetime as dt
from tracemalloc import start
from urllib.parse import quote_plus as urlquote
from kiteconnect import KiteConnect
from pangres import upsert
import pandas as pd
from sqlalchemy import create_engine, text, null
import re
import yaml
import sys
import traceback
import misc_libs as mlibs
import time
from nsepy import get_history

history_day = 1
if len(sys.argv) > 1:
    history_day = int(sys.argv[1])

db_cred = yaml.safe_load(open("../../data_lake/master_config.yml"))
slack_channel = db_cred["slack_channel"]
mlibs.send_slack_monitor_alert(
    _url=slack_channel,
    _title="Data Download code",
    _message="Data download code started",
)
option_list = []
index_list = []
segments = ["option", "future", "index", "equity", "daily_data"]

aa = bb = dd = pd.DataFrame()
try:
    engine = create_engine(
        f"postgresql://{db_cred['db_config']['user_name']}:%s@{db_cred['db_config']['user_ip']}:{db_cred['db_config']['user_port']}/postgres"
        % urlquote(db_cred["db_config"]["user_passwd"])
    )

    df_cred = pd.read_sql(
        f"select * from stock_db.client_secret where client_id='C00004'", con=engine
    )

    _api_key = df_cred.loc[0, "data4"]
    _access_token = df_cred.loc[0, "data6"]
    kite = KiteConnect(api_key=_api_key)
    kite.set_access_token(_access_token)

    time_now = dt.now()

    print("Code started", time_now)

    df_his = kite.historical_data(
        260105, from_date=time_now - timedelta(days=5), to_date=time_now, interval="day"
    )
    y_day = df_his[-2]["date"]
    y_day = time_now - timedelta(days=history_day)
    y_day = y_day.replace(hour=9).replace(minute=00)

    instruments = pd.read_csv("../../data_lake/base_dir/instruments")
    df_final = pd.DataFrame()
    if "index" in segments:
        aa = instruments[instruments.name.isin(["NIFTY BANK", "NIFTY 50", "INDIA VIX"])]
        index_list.append(aa["tradingsymbol"].unique().tolist())
        for i, idx in aa.iterrows():
            time.sleep(1)
            try:
                df = kite.historical_data(
                    idx.instrument_token,
                    from_date=y_day,
                    to_date=time_now,
                    interval="minute",
                )
                df = pd.DataFrame(df)
                if len(df) > 0:
                    print(idx.tradingsymbol)
                    df = df.rename({"date": "record_time"}, axis=1)
                    df["record_time"] = df["record_time"].dt.tz_localize(None)
                    df["timeframe"] = "1m"
                    df["volume"] = 0
                    df["trading_symbol"] = idx.tradingsymbol.replace(" ", "_")
                    df["expiry_date"] = null()
                    df = df.drop_duplicates(
                        subset=["trading_symbol", "record_time"], keep="last"
                    )
                    df_final = pd.concat([df, df_final], axis=0)
                    df = df.set_index(["trading_symbol", "record_time"])

                    upsert(
                        con=engine,
                        df=df,
                        table_name="index_minute",
                        if_row_exists="update",
                        schema="stock_db",
                    )
            except Exception as e:
                e = traceback.format_exc()
                print(e)
                mlibs.send_slack_error_alert(
                    _url=slack_channel,
                    _title="Data Download code",
                    _message=e,
                )

    if "option" in segments:
        aa = instruments[instruments.name.isin(["NIFTY BANK", "NIFTY 50"])]
        index_list.append(aa["tradingsymbol"].unique().tolist())
        mapping = {"NIFTY BANK": "BANKNIFTY", "NIFTY 50": "NIFTY"}
        for i, idx in aa.iterrows():
            try:
                df = kite.historical_data(
                    idx.instrument_token,
                    from_date=time_now - timedelta(days=4),
                    to_date=time_now,
                    interval="day",
                )

                df = pd.DataFrame(df).iloc[-1]
                high, low = df.high, df.low
                high = high * 1.10
                low *= 0.9
                opt_list = instruments[
                    (instruments.name == mapping[idx.tradingsymbol])
                    & (instruments.strike < high)
                    & (instruments.strike > low)
                    & (instruments.exchange == "NFO")
                ]
                opt_list["expiry"] = pd.to_datetime(opt_list["expiry"])
                opt_list = opt_list[(opt_list["expiry"] - time_now).dt.days < 15]
                opt_list = opt_list[
                    opt_list.tradingsymbol.apply(lambda x: len(x) == 17)
                ]
                for i, iops in opt_list.iterrows():
                    time.sleep(1)
                    try:
                        df = kite.historical_data(
                            iops.instrument_token,
                            from_date=y_day,
                            to_date=time_now,
                            interval="minute",
                        )
                        df = pd.DataFrame(df)
                        if len(df) > 0:
                            print(iops.tradingsymbol)
                            df = df.rename({"date": "record_time"}, axis=1)
                            df["record_time"] = df["record_time"].dt.tz_localize(None)
                            df["timeframe"] = "1m"
                            df["trading_symbol"] = iops.tradingsymbol.replace(" ", "_")
                            df["expiry_date"] = iops.expiry
                            df = df.drop_duplicates(
                                subset=["trading_symbol", "record_time"], keep="last"
                            )
                            df_final = pd.concat([df, df_final], axis=0)
                            df = df.set_index(
                                ["trading_symbol", "record_time", "expiry_date"]
                            )
                            upsert(
                                con=engine,
                                df=df,
                                table_name="option_minute",
                                if_row_exists="update",
                                schema="stock_db",
                            )
                            a = df.copy()
                    except Exception as e:
                        e = traceback.format_exc()
                        print(e)
                        mlibs.send_slack_error_alert(
                            _url=slack_channel,
                            _title="Data Download code",
                            _message=e,
                        )
            except Exception as e:
                e = traceback.format_exc()

    if "future" in segments:

        future_info = instruments[
            instruments.tradingsymbol.str.contains(r"^BANKNIFTY|^NIFTY")
            & (instruments.instrument_type == "FUT")
        ]
        future_info["expiry"] = pd.to_datetime(future_info["expiry"], format="%Y-%m-%d")
        future_info = future_info[future_info.expiry.dt.month == time_now.month]
        for i, idx in future_info.iterrows():
            try:
                df = kite.historical_data(
                    idx.instrument_token,
                    from_date=y_day,
                    to_date=time_now,
                    interval="minute",
                )
                df = pd.DataFrame(df)
                if len(df) > 0:
                    print(idx.tradingsymbol)
                    df = df.rename({"date": "record_time"}, axis=1)
                    df["record_time"] = df["record_time"].dt.tz_localize(None)
                    df["timeframe"] = "1m"
                    df["trading_symbol"] = idx.tradingsymbol.replace(" ", "_")
                    df = df.drop_duplicates(
                        subset=["trading_symbol", "record_time"], keep="last"
                    )
                    df_final = pd.concat([df, df_final], axis=0)
                    df = df.set_index(["trading_symbol", "record_time"])

                    upsert(
                        con=engine,
                        df=df,
                        table_name="future_minute",
                        if_row_exists="update",
                        schema="stock_db",
                    )
            except Exception as e:
                e = traceback.format_exc()
                print(e)
                mlibs.send_slack_error_alert(
                    _url=slack_channel,
                    _title="Data Download code",
                    _message=e,
                )
    if "equity" in segments:
        nifty500 = pd.read_csv("../../data_lake/base_dir/ind_nifty500list.csv")[
            "Symbol"
        ].tolist()

        dd = instruments[
            instruments.tradingsymbol.isin(nifty500)
            & (instruments.instrument_type == "EQ")
            & (instruments.segment == "NSE")
        ]

        for i, idx in dd.iterrows():
            time.sleep(1)
            try:
                print(idx.tradingsymbol)
                df = kite.historical_data(
                    idx.instrument_token,
                    from_date=y_day,
                    to_date=time_now,
                    interval="minute",
                )
                df = pd.DataFrame(df)

                df = df.rename({"date": "record_time"}, axis=1)
                df["record_time"] = df["record_time"].dt.tz_localize(None)
                df["timeframe"] = "1m"
                # df["volume"] = 0
                df["trading_symbol"] = idx.tradingsymbol.replace(" ", "_")
                df["expiry_date"] = null()
                df = df.drop_duplicates(
                    subset=["trading_symbol", "record_time"], keep="last"
                )
                df_final = pd.concat([df, df_final], axis=0)
                df = df.set_index(["trading_symbol", "record_time"])
                upsert(
                    con=engine,
                    df=df,
                    table_name="kaggle_archive",
                    if_row_exists="update",
                    schema="stock_db",
                )
            except Exception as e:
                e = traceback.format_exc()
                print(e)
                mlibs.send_slack_error_alert(
                    _url=slack_channel,
                    _title="Data Download code",
                    _message=e,
                )
    if "daily_data" in segments:
        instruments = pd.read_csv(
            "../../data_lake/base_dir/ind_nifty500list.csv", index_col="Symbol"
        )
        instruments = instruments.index.tolist()
        instruments += ["NIFTY BANK", "NIFTY 50"]
        end_date = date.today()
        start_date = end_date - timedelta(days=5)
        start_date = date(2022, 7, 20)
        for name in instruments:
            try:
                print(name)
                if "NIFTY" in name:
                    df = get_history(
                        symbol=name, start=start_date, end=end_date, index=True
                    ).reset_index()
                    df["symbol"] = name
                else:
                    df = get_history(
                        symbol=name, start=start_date, end=end_date
                    ).reset_index()
                df = df.reset_index()
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.rename(lambda x: x.lower(), axis=1)
                df = df.rename(
                    {"date": "record_time", "symbol": "trading_symbol"}, axis=1
                )
                df["timeframe"] = "1d"
                df["expiry_date"] = null()
                df = df[
                    [
                        "trading_symbol",
                        "record_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "expiry_date",
                        "timeframe",
                    ]
                ]
                df = df.set_index(["trading_symbol", "record_time"])
                upsert(
                    con=engine,
                    df=df,
                    table_name="daily_data",
                    schema="stock_db",
                    if_row_exists="update",
                )

            except Exception as e:
                mlibs.send_slack_error_alert(
                    _url=slack_channel,
                    _title="Data Download code",
                    _message=e,
                )

    mlibs.send_slack_monitor_alert(
        _url=slack_channel,
        _title="Data Download code",
        _message="Data download code copleted",
    )
except Exception as e:
    e = traceback.format_exc()
    print(e)
    mlibs.send_slack_error_alert(
        _url=slack_channel,
        _title="Data Download code",
        _message=e,
    )
