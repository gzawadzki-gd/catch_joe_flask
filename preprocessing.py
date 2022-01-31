import pandas as pd
import numpy as np

from pytz import timezone


fav_sites = [
    "blogadda.com",
    "jeuxonline.info",
    "lenta.ru",
    "mail.google.com",
    "slack.com",
    "toptal.com",
    "ucoz.net",
    "vk.com",
    "youtube.com",
]
timezones = {
    "Chicago": -6,
    "Singapore": 8,
    "Paris": 1,
    "Berlin": 1,
    "San Francisco": -8,
    "Shanghai": 9,
    "Auckland": 13,
    "Kuala Lumpur": 8,
    "Moscow": 3,
    "Rio de Janeiro": -3,
    "Toronto": -5,
    "Madrid": 1,
    "Miami": -5,
    "Delhi": 5.5,
    "Amsterdam": 1,
    "Tokyo": 9,
    "London": 0,
    "Rome": 1,
    "Sydney": 11,
    "New York": -5,
    "Vancouver": -8,
}


def unwrap_location(df):
    df[["country", "city"]] = df["location"].str.split("/", expand=True)
    return df


def fix_locale(df):
    df["locale"] = df["locale"].str.replace("_", "-").replace(" ", "")
    return df


def unwrap_date(df):
    df = to_date_hour(df)
    df = fix_date_timezone(df)

    df[["dayofweek", "day", "month", "year"]] = df["date_hour"].apply(
        lambda x: pd.Series([x.dayofweek, x.day, x.month, x.year]).astype(str)
    )
    return df


def to_date_hour(df):
    df["date_hour"] = pd.to_datetime(df.date.astype(str) + " " + df.time.astype(str))
    return df


def fix_date_timezone(df):
    df["date_hour"] = df.apply(
        lambda x: (x["date_hour"] + pd.DateOffset(hours=timezones[x.city]))
        if x.city in timezones
        else x["date_hour"],
        axis=1,
    )
    return df


def time_to_decimal(df):
    df["hour"] = df["time"].apply(
        lambda x: pd.to_datetime(x).hour + pd.to_datetime(x).minute / 60
    )
    return df


def fixed_time_to_decimal(df):
    df["hour_fixed"] = df["date_hour"].apply(
        lambda x: pd.to_datetime(x).hour + pd.to_datetime(x).minute / 60
    )
    return df


def clean_sites(x):
    if x in fav_sites:
        return x
    elif x == "0":
        return "empty"
    else:
        return "other"


def get_num_sites(history):
    return history.notnull().sum(axis=1)


def get_session_len(history):
    return history.apply(lambda x: x.str["length"]).sum(axis=1)


def get_max_len_site(history):
    len_sites = history.apply(lambda x: x.str["length"]).fillna(0)
    return len_sites.max(axis=1)


def unwrap_sites(df):
    history = df["sites"].apply(pd.Series)
    df["sites_num"] = get_num_sites(history)
    df["session_len"] = get_session_len(history)
    df["max_len_site"] = get_max_len_site(history)
    for i in history.columns:
        df["site_" + str(i)] = (
            history.fillna("").apply(lambda x: x.str["site"])[i].apply(clean_sites)
        )
    return df


def add_is_joe(df):
    df["is_joe"] = df["user_id"].apply(lambda x: 1 if x == 0 else 0)
    return df


def preprocess(df):
    useless_cols = ["sites", "location", "date", "date_hour", "time"]
    df = unwrap_location(df)
    df = fix_locale(df)
    df = unwrap_date(df)

    

    df = time_to_decimal(df)
    df = fixed_time_to_decimal(df)
    df = unwrap_sites(df)
    # df = add_is_joe(df)
    df = df.drop(useless_cols, axis=1)

    return df
