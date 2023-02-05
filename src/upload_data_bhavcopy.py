import pandas as pd
from datetime import datetime as dt, timedelta, date
from sqlalchemy import create_engine, text, null
from glob import glob
import numpy as np
import warnings
from pangres import upsert
import re

# from mysql import connector


"""
This code performs data quality check on raw observation records. It performs only for latest data uisng following method:
Get last 2 days data from raw table by timestamp
Get last 2 hours data from from dq table
Mask raw data by max and min of dq data
Take intersection in obs id
If any data is remaining, perform DQ else stop
Perform DQ on raw data.
Upload only the part which is uncommon by obs id
"""

warnings.filterwarnings("ignore")

# engine = connector.connect(
#     host="localhost", user="root", password="arijitdb", database="stock_db"
# )
# engine = create_engine(
#     f"postgresql://{dq_cred['artemis']['user_name']}:{dq_cred['artemis']['user_passwd']}@{dq_cred['artemis']['user_ip']}:{dq_cred['artemis']['user_port']}/mausam"
# )

engine = create_engine(f"postgresql://arijit:arijitdb@localhost:5432/stock_db")

filelist = glob("../../data_lake/bhavcopy/*.zip")[0:1]


for i, name in enumerate(filelist):
    print(f"{i} of {len(filelist)} done")
    try:
        df = pd.read_csv(name, compression="zip")
        df["EXPIRY_DT"] = pd.to_datetime(
            df["EXPIRY_DT"], format="%d-%b-%Y", infer_datetime_format=True
        )
        df["TIMESTAMP"] = pd.to_datetime(
            df["TIMESTAMP"], format="%d-%b-%Y", infer_datetime_format=True
        )
        df = df.rename(
            {
                "EXPIRY_DT": "expiry_date",
                "TIMESTAMP": "record_time",
                "CHG_IN_OI": "chg_oi",
                "CONTRACTS": "contract",
                "SYMBOL": "trading_symbol",
                "VAL_INLAKH": "value_in_lakh",
                "OPEN_INT": "oi",
                "STRIKE_PR": "strike",
                "OPTION_TYP": "option_type",
            },
            axis=1,
        )
        df = df.rename(lambda x: x.lower(), axis=1)
        df = df[
            [
                "trading_symbol",
                "record_time",
                "open",
                "high",
                "low",
                "close",
                "value_in_lakh",
                "contract",
                "oi",
                "chg_oi",
                "strike",
                "expiry_date",
                "option_type",
            ]
        ]
        df = df.set_index(
            ["trading_symbol", "record_time", "strike", "expiry_date", "option_type"]
        )
        # # df.to_sql("option_backup", engine, if_exists="replace", index=False)

        upsert(
            engine=engine,
            df=df,
            table_name="daily_bhav",
            if_row_exists="update",
        )

    except Exception as e:
        print(name, "not done")
        print(e)
