import pandas as pd

def drop_column(df, column_names) -> pd.DataFrame:
    df = df.drop(column_names, axis = 1)
    return df

def transform_percent(str_percent:str) -> float:
    return float(str(str_percent).strip('%'))/100

df = pd.read_csv(".\Practica 1\Valve_Player_Data.csv")

df = drop_column(df, ["Month_Year"])

print(df.dtypes)

df["Date"] = pd.to_datetime(df["Date"], format = '%Y-%m-%d')
df['Percent_Gain'] = df['Percent_Gain'].apply(transform_percent)

print()
print(df.dtypes)

print(df)