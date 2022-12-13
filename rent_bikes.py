#Importación de Librerias
from fastapi import FastAPI
from fastapi import status
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from joblib import load
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from joblib import dump, load
from fastapi.middleware.cors import CORSMiddleware


#abrir el archivo csv
bike_data = pd.read_csv('daily-bike-share.csv')
#caracteristicas
X = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values
y = bike_data['rentals'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#predicción
model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)

print(X_test[0])
print(predictions[0])

dump(model, 'model.joblib') 

model_load = load('model.joblib') 
predictions = model_load.predict(X_test)
print(predictions[:2])


#caracteristicas
class Features(BaseModel):
    season: int
    mnth: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int
    temp: float
    atemp: float
    hum: float
    windspeed: float


#modificar
    class Config:
        schema_extra = {
            "example": {
                "season": 3.,
                "mnth": 7.,
                "holiday": 0.,
                "weekday": 6.,
                "workingday": 0,
                "weathersit": 1,
                "temp": 0.686667,
                "atemp": 0.638263,
                "hum": 0.585,
                "windspeed": 0.208342
            }
        }



class Label(BaseModel):
    rentals: float


#mensaje
class Message(BaseModel):
    message: float

description = """# RENTALS API

## Rentals prediction

1. **season**:un valor codificado numéricamente que indica la temporada 
2. **mnth**: El mes calendario en que se realizó la observación 
3. **holiday**: Un valor binario que indica si la observación se realizó o no en un día festivo)
4. **weekday**: El día de la semana en que se realizó la observación (0:Domingo ... 6:Sábado)
5. **workingday**: Un valor binario que indica si el día es o no un día laborable (no un fin de semana ni un día festivo)
6. **weathersit**: Un valor categórico que indica la situación meteorológica (1:claro, 2:niebla/nubes, 3:lluvia ligera/nieve, 4:lluvia intensa/granizo/nieve/niebla)
7. **temp**: La temperatura en Celsius (normalizada)
8. **atemp**: La temperatura aparente ("sensación") en grados Celsius (normalizada)
9. **hum**: El nivel de humedad (normalizado)
10. **windspeed**: La velocidad del viento (normalizada)

"""

app = FastAPI()


#terminales
origins = [
    "http://127.0.0.1:8000/",
    "https://8000-katiaolem-apibikes-e5yvw2ars1r.ws-us77.gitpod.io/"
    "*",        
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/rentals",
    response_model=Label,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Rentals prediction",
    description="Rentals prediction",
    tags=["Rentals"]
)
async def get_rentals(features:Features):
    try:
        model = load('model.joblib')
        data = [
            features.season,
            features.mnth,
            features.holiday,
            features.weekday,
            features.workingday,
            features.weathersit,
            features.temp,
            features.atemp,
            features.hum,
            features.windspeed
        ]
        predictions = model.predict([data])
        response = {"rentals": predictions[0]}
        return response
    except Exception as e:
        response = JSONResponse(
                    status_code=400,
                    content={"message":f"{e.args}"},
                )
        return response

