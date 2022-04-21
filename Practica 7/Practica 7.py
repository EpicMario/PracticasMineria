import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numbers


def drop_column(df, column_names) -> pd.DataFrame:
    df = df.drop(column_names, axis = 1)
    return df

def transform_percent(str_percent:str) -> float:
    return float(str(str_percent).strip('%'))/100

def get_df() -> pd.DataFrame:
    df = pd.read_csv(".\Practica 1\Valve_Player_Data.csv")

    df = drop_column(df, ["Month_Year"])
    df["Date"] = pd.to_datetime(df["Date"], format = '%Y-%m-%d')
    df['Percent_Gain'] = df['Percent_Gain'].apply(transform_percent)

    return df

def transform_variable(df: pd.DataFrame, x: str) -> pd.Series:
    if isinstance(df[x][df.index[0]], numbers.Number):
        return df[x]  # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])


def linear_regression(df: pd.DataFrame, x: str, y: str) -> dict[str, float]:
    fixed_x = transform_variable(df, x)
    model = sm.OLS(list(df[y]), sm.add_constant(fixed_x), alpha=0.05).fit()
    bands = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]
    coef = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0][
        "coef"
    ]
    tables_0 = model.summary().tables[0].as_html()
    r_2_t = pd.read_html(tables_0, header=None, index_col=None)[0]
    return {
        "m": coef.values[1],
        "b": coef.values[0],
        "r2": r_2_t.values[0][3],
        "r2_adj": r_2_t.values[1][3],
        "low_band": bands["[0.025"][0],
        "hi_band": bands["0.975]"][0],
    }


def plt_lr(
    df: pd.DataFrame,
    x: str,
    y: str,
    m: float,
    b: float,
    r2: float,
    r2_adj: float,
    low_band: float,
    hi_band: float,
    colors: tuple[str, str],
):
    fixed_x = transform_variable(df, x)
    plt.plot(df[x], [m * x + b for _, x in fixed_x.items()], color=colors[0])
    plt.fill_between(
        df[x],
        [m * x + low_band for _, x in fixed_x.items()],
        [m * x + hi_band for _, x in fixed_x.items()],
        alpha=0.2,
        color=colors[1],
    )

df = get_df()

games = df["Game_Name"] == "Counter Strike: Global Offensive"
dt_date = df["Date"].dt.date

df = df[games].groupby(dt_date).sum()
df.reset_index(inplace=True)
df = df[["Date", "Peak_Players"]]
df_tail = df.tail(50)

x = "Date"
y = "Peak_Players"

df_tail.plot(x=x, y=y, kind="scatter")
lr = linear_regression(df_tail, x, y)
plt_lr(df=df_tail, x=x, y=y, colors=("red", "orange"), **lr)

lr = linear_regression(df_tail.tail(5), x, y)
plt_lr(df=df_tail.tail(5), x=x, y=y, colors=("red", "orange"), **lr)
df_thursday = df_tail[pd.to_datetime(df_tail[x]).dt.dayofweek == 4]
print(df_thursday)

lr = linear_regression(df_thursday, x, y)
plt_lr(df=df_thursday, x=x, y=y, colors=("blue", "blue"), **lr)

plt.xticks(rotation=90)
plt.savefig(f"Practica 7/plots/lr_{y}_{x}_m.png")
plt.close()

df_2019_2020 = df.loc[
    (pd.to_datetime(df[x]) >= "2019-09-08")
    & (pd.to_datetime(df[x]) < "2020-09-08")
]

dfs = [
    ("50D", df_tail),
    ("10D", df_tail.tail(10)),
    ("5D", df_tail.tail(5)),
    (
        "jueves",
        df_tail[pd.to_datetime(df_tail[x]).dt.dayofweek == 4],
    ),
    ("50D-1Y", df_2019_2020),
    ("10D-Y", df_2019_2020.tail(10)),
    ("5D-Y", df_2019_2020.tail(5)),
    (
        "jueves-Y",
        df_2019_2020[pd.to_datetime(df_tail[x]).dt.dayofweek == 4],
    ),
]
lrs = [(title, linear_regression(_df, x=x, y=y), len(_df)) for title, _df in dfs[:-1:]]
lrs_p = [
    (title, lr_dict["m"] * size + lr_dict["b"], lr_dict)
    for title, lr_dict, size in lrs
]
print(lrs_p)