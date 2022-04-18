import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

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

df = get_df()

dt_year = df["Date"].dt.year

grp_game_Name_and_year = df.groupby(["Game_Name", dt_year])
total_gain_views = grp_game_Name_and_year[["Peak_Players", "Gain"]].sum()
total_gain_views.reset_index(inplace=True)
total_gain_views.drop("Date", inplace=True, axis=1)

game_Name_x_views = total_gain_views[["Game_Name", "Peak_Players"]]

model = ols("Peak_Players ~ Game_Name", data=game_Name_x_views).fit()
df_anova = sm.stats.anova_lm(model, typ=2)

if df_anova["PR(>F)"][0] < 0.005:
    print("Hay diferencias")
    print(df_anova)
else:
    print("No hay diferencias")