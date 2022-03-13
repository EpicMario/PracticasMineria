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
group_filter_year = df.groupby(filter_year)

print("El juego con el mayor pico de jugadores por año: ")
print(df.loc[group_filter_year["Peak_Players"].idxmax()][["Date", "Game_Name", "Peak_Players"]])

print("El juego mayor perdida de jugadores por año: ")
print(df.loc[group_filter_year["Gain"].idxmin()][["Date", "Game_Name", "Gain"]])

print("El promedio de picos de jugadores por año")
print(group_filter_year["Peak_Players"].mean().round(2))

years = pd.unique(filter_year)
game_names = df.loc[group_filter_year["Peak_Players"].idxmax()]["Game_Name"][::-1]

for year, game_name in zip(years, game_names):
    new_df = df[(filter_year == year) & (df["Game_Name"] == game_name)][["Date", "Peak_Players"]]
    new_df = new_df.apply(lambda date: date.dt.month if date.name == "Date" else date)
    new_df.plot(x = "Date", 
                y = "Peak_Players", 
                kind = "bar", 
                xlabel = year, 
                title = game_name)
    plt.gca().ticklabel_format(axis='y', style='plain')
    plt.savefig(f"practica 3/plots/peak_players_{year}.png")


