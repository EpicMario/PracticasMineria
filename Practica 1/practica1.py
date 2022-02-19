import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

if os.path.exists("Valve_Player_Data.csv"):
    print("El dataset ya se genero")
else:
    print("Generando dataset")
    api.dataset_download_file('jackogozaly/steam-player-data','Valve_Player_Data.csv')