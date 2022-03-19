import pandas as pd
import matplotlib.pyplot as plt

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
filter_year = df["Date"].dt.year
filter_name = df["Game_Name"]
group_filter_year = df.groupby(filter_year)

years = pd.unique(filter_year)
top_game_names = df.loc[group_filter_year["Peak_Players"].idxmax()]["Game_Name"][::-1]

for year, game_name in zip(years, top_game_names):
    new_df = df[(filter_year == year) & (filter_name == game_name)][["Date", "Peak_Players"]]
    new_df = new_df.apply(lambda date: date.dt.month if date.name == "Date" else date)
    new_df.plot(x = "Date", 
                y = "Peak_Players", 
                kind = "line", 
                xlabel = year, 
                title = game_name)
    plt.gca().ticklabel_format(axis='y', style='plain')
    plt.savefig(f"practica 4/peak_players_of_peak_games/peak_players_{year}.png")
    plt.close()

group_filter_name = df.groupby("Game_Name")
new_df = df[(filter_name == game_name) & (filter_year == 2020)]
new_df = group_filter_name["Peak_Players"].mean().round(2)
new_df = new_df.head(10)
new_df.plot(x = "Peak_Players", 
            kind = "pie", 
            ylabel = "",
            title = "Top 10 mean of players of 2020")
plt.gca().ticklabel_format(axis='y', style='plain')
plt.savefig("practica 4/top_10_mean_players_of_2020.png")