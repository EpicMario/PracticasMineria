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
    if isinstance(df[x][0], numbers.Number):
        return df[x]  # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])


def linear_regression(df: pd.DataFrame, x, y) -> None:
    fixed_x = transform_variable(df, x)
    model = sm.OLS(df[y], sm.add_constant(fixed_x)).fit()
    print(model.summary())
    html = model.summary().tables[1].as_html()
    coef = pd.read_html(html, header=0, index_col=0)[0]["coef"]
    df.plot(x=x, y=y, kind="scatter")
    plt.plot(df[x], [pd.DataFrame.mean(df[y]) for _ in fixed_x.items()], color="green")
    plt.plot(
        df[x],
        [coef.values[1] * x + coef.values[0] for _, x in fixed_x.items()],
        color="red",
    )
    plt.xticks(rotation=90)
    plt.savefig(f"practica 6/plots/lr_{y}_{x}.png")
    plt.close()

df = get_df()

df_peaks = df["Peak_Players"] >= 70320
grp_year = df[df_peaks].groupby(df["Date"].dt.year)
grp_year = grp_year.mean()

grp_year.reset_index(inplace=True)
grp_year.drop("Date", inplace=True, axis=1)

linear_regression(grp_year, "Peak_Players", "Avg_players")
linear_regression(grp_year, "Percent_Gain", "Gain")
linear_regression(grp_year, "Gain", "Avg_players")
linear_regression(grp_year, "Avg_players", "Percent_Gain")
linear_regression(grp_year, "Peak_Players", "Gain")
linear_regression(grp_year, "Peak_Players", "Percent_Gain")