from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import pickle
import re


with open('model.pkl', 'rb') as f:
    best_model = pickle.load(f)

columns_after_get_dummies = \
    {'name_Force', 'name_Skoda', 'name_Lexus', 'name_Jeep',
    'owner_Test Drive Car', 'max_torque_rpm', 'name_Datsun',
     'name_Land', 'name_Maruti', 'name_Volvo', 'name_Jaguar',
     'seller_type_Trustmark Dealer', 'name_Isuzu', 'name_Hyundai',
     'torque', 'seller_type_Individual', 'owner_Third Owner',
     'max_power', 'name_Chevrolet', 'name_Mitsubishi', 'fuel_Diesel',
     'name_Ashok', 'mileage', 'name_Fiat', 'name_Kia', 'owner_Second Owner',
     'fuel_Petrol', 'name_Audi', 'year', 'name_MG', 'name_Peugeot',
     'name_Honda', 'owner_Fourth & Above Owner', 'name_Ford',
     'name_Volkswagen', 'name_Mercedes-Benz', 'name_Tata', 'name_Nissan',
     'name_Opel', 'seats', 'fuel_LPG', 'name_Daewoo', 'name_Mahindra',
     'name_BMW', 'engine', 'transmission_Manual', 'km_driven',
     'name_Renault', 'name_Toyota'}

feature_history = \
    [
     (2, 'name_Force', 'torque'), (2, 'name_Force', 'max_power'), (2, 'name_Force', 'year'),
     (2, 'name_Force', 'name_BMW'), (2, 'name_Force', 'engine'), (2, 'name_Force', 'transmission_Manual'),
     (2, 'name_Skoda', 'torque'), (2, 'name_Skoda', 'max_power'), (2, 'name_Skoda', 'year'), (2, 'name_Skoda', 'name_BMW'),
     (2, 'name_Skoda', 'engine'), (2, 'name_Skoda', 'transmission_Manual'), (2, 'name_Lexus', 'torque'), (2, 'name_Lexus', 'max_power'),
     (2, 'name_Lexus', 'year'), (2, 'name_Lexus', 'name_BMW'), (2, 'name_Lexus', 'engine'), (2, 'name_Lexus', 'transmission_Manual'),
     (2, 'name_Jeep', 'torque'), (2, 'name_Jeep', 'max_power'), (2, 'name_Jeep', 'year'), (0, 'name_Jeep', 'name_BMW'),
     (2, 'name_Jeep', 'name_BMW'), (2, 'name_Jeep', 'engine'), (2, 'name_Jeep', 'transmission_Manual'),
     (0, 'owner_Test Drive Car', 'name_Volvo'), (2, 'owner_Test Drive Car', 'torque'), (2, 'owner_Test Drive Car', 'max_power'),
     (2, 'owner_Test Drive Car', 'year'), (0, 'owner_Test Drive Car', 'name_Mercedes-Benz'), (0, 'owner_Test Drive Car', 'name_BMW'),
     (2, 'owner_Test Drive Car', 'name_BMW'), (2, 'owner_Test Drive Car', 'engine'), (2, 'owner_Test Drive Car', 'transmission_Manual'),
     (0, 'max_torque_rpm', 'max_power'), (0, 'max_torque_rpm', 'name_BMW'), (2, 'name_Datsun', 'torque'), (2, 'name_Datsun', 'max_power'),
     (2, 'name_Datsun', 'year'), (2, 'name_Datsun', 'name_BMW'), (2, 'name_Datsun', 'engine'), (2, 'name_Datsun', 'transmission_Manual'),
     (2, 'name_Land', 'torque'), (2, 'name_Land', 'max_power'), (2, 'name_Land', 'year'), (2, 'name_Land', 'name_BMW'),
     (2, 'name_Land', 'engine'), (2, 'name_Land', 'transmission_Manual'), (2, 'name_Maruti', 'torque'), (0, 'name_Maruti', 'max_power'),
     (2, 'name_Maruti', 'max_power'), (2, 'name_Maruti', 'year'), (0, 'name_Maruti', 'name_BMW'), (2, 'name_Maruti', 'name_BMW'),
     (2, 'name_Maruti', 'engine'), (2, 'name_Maruti', 'transmission_Manual'), (2, 'name_Volvo', 'torque'), (2, 'name_Volvo', 'max_power'),
     (0, 'name_Volvo', 'name_Audi'), (2, 'name_Volvo', 'year'), (0, 'name_Volvo', 'name_Mercedes-Benz'), (0, 'name_Volvo', 'name_BMW'),
     (2, 'name_Volvo', 'name_BMW'), (2, 'name_Volvo', 'engine'), (2, 'name_Volvo', 'transmission_Manual'), (2, 'name_Jaguar', 'torque'),
     (2, 'name_Jaguar', 'max_power'), (2, 'name_Jaguar', 'year'), (2, 'name_Jaguar', 'name_BMW'), (2, 'name_Jaguar', 'engine'),
     (2, 'name_Jaguar', 'transmission_Manual'), (2, 'seller_type_Trustmark Dealer', 'torque'), (2, 'seller_type_Trustmark Dealer', 'max_power'),
     (2, 'seller_type_Trustmark Dealer', 'year'), (2, 'seller_type_Trustmark Dealer', 'name_BMW'), (2, 'seller_type_Trustmark Dealer', 'engine'),
     (2, 'seller_type_Trustmark Dealer', 'transmission_Manual'), (2, 'name_Isuzu', 'torque'), (2, 'name_Isuzu', 'max_power'),
     (2, 'name_Isuzu', 'year'), (2, 'name_Isuzu', 'name_BMW'), (2, 'name_Isuzu', 'engine'), (2, 'name_Isuzu', 'transmission_Manual'),
     (2, 'name_Hyundai', 'torque'), (2, 'name_Hyundai', 'max_power'), (2, 'name_Hyundai', 'year'), (0, 'name_Hyundai', 'name_BMW'),
     (2, 'name_Hyundai', 'name_BMW'), (2, 'name_Hyundai', 'engine'), (2, 'name_Hyundai', 'transmission_Manual'),
     (0, 'torque', 'seller_type_Individual'), (1, 'torque', 'owner_Third Owner'), (0, 'torque', 'max_power'), (1, 'torque', 'name_Chevrolet'),
     (1, 'torque', 'name_Mitsubishi'), (1, 'torque', 'name_Fiat'), (1, 'torque', 'name_Kia'), (1, 'torque', 'owner_Second Owner'),
     (1, 'torque', 'name_Audi'), (1, 'torque', 'name_MG'), (1, 'torque', 'name_Peugeot'), (1, 'torque', 'name_Honda'),
     (1, 'torque', 'owner_Fourth & Above Owner'), (1, 'torque', 'name_Ford'), (1, 'torque', 'name_Volkswagen'),
     (1, 'torque', 'name_Mercedes-Benz'), (1, 'torque', 'name_Tata'), (1, 'torque', 'name_Nissan'), (1, 'torque', 'seats'),
     (1, 'torque', 'fuel_LPG'), (1, 'torque', 'name_Daewoo'), (1, 'torque', 'name_Mahindra'), (0, 'torque', 'name_BMW'),
     (1, 'torque', 'name_BMW'), (0, 'torque', 'engine'), (0, 'torque', 'transmission_Manual'), (1, 'torque', 'name_Renault'),
     (1, 'torque', 'name_Toyota'), (0, 'seller_type_Individual', 'max_power'), (2, 'seller_type_Individual', 'max_power'),
     (2, 'seller_type_Individual', 'year'), (0, 'seller_type_Individual', 'name_BMW'), (2, 'seller_type_Individual', 'engine'),
     (0, 'seller_type_Individual', 'transmission_Manual'), (2, 'owner_Third Owner', 'max_power'), (2, 'owner_Third Owner', 'year'),
     (2, 'owner_Third Owner', 'name_BMW'), (2, 'owner_Third Owner', 'engine'), (2, 'owner_Third Owner', 'transmission_Manual'), (0, 'max_power', 'max_power'),
     (1, 'max_power', 'name_Chevrolet'), (1, 'max_power', 'name_Mitsubishi'), (1, 'max_power', 'fuel_Diesel'), (1, 'max_power', 'name_Fiat'),
     (1, 'max_power', 'name_Kia'), (1, 'max_power', 'owner_Second Owner'), (1, 'max_power', 'fuel_Petrol'), (1, 'max_power', 'name_Audi'),
     (0, 'max_power', 'year'), (1, 'max_power', 'name_MG'), (1, 'max_power', 'name_Peugeot'), (1, 'max_power', 'name_Honda'),
     (1, 'max_power', 'owner_Fourth & Above Owner'), (1, 'max_power', 'name_Ford'), (1, 'max_power', 'name_Volkswagen'),
     (1, 'max_power', 'name_Mercedes-Benz'), (1, 'max_power', 'name_Tata'), (1, 'max_power', 'name_Nissan'), (1, 'max_power', 'seats'),
     (1, 'max_power', 'fuel_LPG'), (1, 'max_power', 'name_Daewoo'), (1, 'max_power', 'name_Mahindra'), (0, 'max_power', 'name_BMW'),
     (1, 'max_power', 'name_BMW'), (0, 'max_power', 'engine'), (0, 'max_power', 'transmission_Manual'), (0, 'max_power', 'km_driven'),
     (1, 'max_power', 'name_Renault'), (1, 'max_power', 'name_Toyota'), (2, 'name_Chevrolet', 'year'), (0, 'name_Chevrolet', 'name_BMW'),
     (2, 'name_Chevrolet', 'name_BMW'), (2, 'name_Chevrolet', 'engine'), (2, 'name_Chevrolet', 'transmission_Manual'), (2, 'name_Mitsubishi', 'year'),
     (2, 'name_Mitsubishi', 'name_BMW'), (2, 'name_Mitsubishi', 'engine'), (2, 'name_Mitsubishi', 'transmission_Manual'), (0, 'fuel_Diesel', 'name_BMW'),
     (2, 'fuel_Diesel', 'name_BMW'), (2, 'name_Fiat', 'year'), (2, 'name_Fiat', 'name_BMW'), (2, 'name_Fiat', 'engine'), (2, 'name_Fiat', 'transmission_Manual'),
     (2, 'name_Kia', 'year'), (2, 'name_Kia', 'name_BMW'), (2, 'name_Kia', 'engine'), (2, 'name_Kia', 'transmission_Manual'), (2, 'owner_Second Owner', 'year'),
     (2, 'owner_Second Owner', 'name_BMW'), (2, 'owner_Second Owner', 'engine'), (2, 'owner_Second Owner', 'transmission_Manual'), (0, 'fuel_Petrol', 'name_BMW'),
     (2, 'fuel_Petrol', 'name_BMW'), (2, 'name_Audi', 'year'), (0, 'name_Audi', 'name_Mercedes-Benz'), (0, 'name_Audi', 'name_BMW'), (2, 'name_Audi', 'name_BMW'),
     (2, 'name_Audi', 'engine'), (2, 'name_Audi', 'transmission_Manual'), (0, 'name_Audi', 'name_Toyota'), (1, 'year', 'name_MG'), (1, 'year', 'name_Peugeot'),
     (1, 'year', 'name_Honda'), (1, 'year', 'owner_Fourth & Above Owner'), (1, 'year', 'name_Ford'), (1, 'year', 'name_Volkswagen'), (1, 'year', 'name_Mercedes-Benz'),
     (1, 'year', 'name_Tata'), (1, 'year', 'name_Nissan'), (1, 'year', 'seats'), (1, 'year', 'fuel_LPG'), (1, 'year', 'name_Daewoo'), (1, 'year', 'name_Mahindra'),
     (1, 'year', 'name_BMW'), (1, 'year', 'transmission_Manual'), (1, 'year', 'name_Renault'), (1, 'year', 'name_Toyota'), (2, 'name_MG', 'name_BMW'), (2, 'name_MG', 'engine'),
     (2, 'name_MG', 'transmission_Manual'), (2, 'name_Peugeot', 'name_BMW'), (2, 'name_Peugeot', 'engine'), (2, 'name_Peugeot', 'transmission_Manual'),
     (0, 'name_Honda', 'name_BMW'), (2, 'name_Honda', 'name_BMW'), (2, 'name_Honda', 'engine'), (2, 'name_Honda', 'transmission_Manual'), (2, 'owner_Fourth & Above Owner', 'name_BMW'),
     (2, 'owner_Fourth & Above Owner', 'engine'), (2, 'owner_Fourth & Above Owner', 'transmission_Manual'), (0, 'name_Ford', 'name_BMW'), (2, 'name_Ford', 'name_BMW'), (2, 'name_Ford', 'engine'),
     (2, 'name_Ford', 'transmission_Manual'), (0, 'name_Volkswagen', 'name_BMW'), (2, 'name_Volkswagen', 'name_BMW'), (2, 'name_Volkswagen', 'engine'), (2, 'name_Volkswagen', 'transmission_Manual'),
     (0, 'name_Mercedes-Benz', 'name_BMW'), (2, 'name_Mercedes-Benz', 'name_BMW'), (2, 'name_Mercedes-Benz', 'engine'), (2, 'name_Mercedes-Benz', 'transmission_Manual'),
     (0, 'name_Mercedes-Benz', 'name_Toyota'), (0, 'name_Tata', 'name_BMW'), (2, 'name_Tata', 'name_BMW'), (2, 'name_Tata', 'engine'), (2, 'name_Tata', 'transmission_Manual'),
     (2, 'name_Nissan', 'name_BMW'), (2, 'name_Nissan', 'engine'), (2, 'name_Nissan', 'transmission_Manual'), (2, 'seats', 'name_BMW'), (2, 'seats', 'engine'),
     (2, 'seats', 'transmission_Manual'), (2, 'fuel_LPG', 'name_BMW'), (2, 'fuel_LPG', 'engine'), (2, 'fuel_LPG', 'transmission_Manual'), (2, 'name_Daewoo', 'name_BMW'),
     (2, 'name_Daewoo', 'engine'), (2, 'name_Daewoo', 'transmission_Manual'), (0, 'name_Mahindra', 'name_BMW'), (2, 'name_Mahindra', 'name_BMW'), (2, 'name_Mahindra', 'engine'),
     (2, 'name_Mahindra', 'transmission_Manual'), (0, 'name_BMW', 'name_BMW'), (0, 'name_BMW', 'engine'), (2, 'name_BMW', 'engine'), (0, 'name_BMW', 'transmission_Manual'),
     (2, 'name_BMW', 'transmission_Manual'), (0, 'name_BMW', 'km_driven'), (0, 'name_BMW', 'name_Renault'), (1, 'name_BMW', 'name_Renault'), (0, 'name_BMW', 'name_Toyota'),
     (1, 'name_BMW', 'name_Toyota'), (0, 'engine', 'engine'), (0, 'engine', 'transmission_Manual'), (1, 'engine', 'name_Renault'), (1, 'engine', 'name_Toyota'),
     (0, 'transmission_Manual', 'transmission_Manual'), (0, 'transmission_Manual', 'km_driven'), (1, 'transmission_Manual', 'name_Renault'), (1, 'transmission_Manual', 'name_Toyota')
    ]


def get_number(s):
  if isinstance(s, str):
    number = re.findall(r'\b\d+\b', s)
    if len(number) > 0:
      return float(number[0])

def get_rpm(s):
  if isinstance(s, str):
    number = re.findall(r'[\d.,]+', s)
    if len(number) >= 2:
      return float(number[1].replace(',', ''))

def get_torque(s):
  g = 9.8
  if isinstance(s, str):
    number = re.findall(r'[\d.,]+', s)
    if len(number) >= 1:
      n = float(number[0].replace(',', ''))
      if 'kgm' in s:
        n *= g
      return n

def get_mark(x):
  return x.split()[0]



def item_to_df(item):
    values = [(
        item.name,
        item.year,
        item.km_driven,
        item.fuel,
        item.seller_type,
        item.transmission,
        item.owner,
        item.mileage,
        item.engine,
        item.max_power,
        item.torque,
        item.seats
    )]
    keys = [
        'name',
        'year',
        'km_driven',
        'fuel',
        'seller_type',
        'transmission',
        'owner',
        'mileage',
        'engine',
        'max_power',
        'torque',
        'seats'
    ]
    df = pd.DataFrame(values, columns=keys)
    return df


def transform_df(df):
    df['mileage'] = df['mileage'].apply(get_number)
    df['engine'] = df['engine'].apply(get_number)
    df['max_power'] = df['max_power'].apply(get_number)
    df['max_torque_rpm'] = df['torque'].apply(get_rpm)
    df['torque'] = df['torque'].apply(get_torque)
    for c in 'engine', 'seats':
        df[c] = df[c].astype(int)

    for c in df.columns:
        if df[c].isnull().sum() > 0:
            m = df[c].median()
            df[c] = df[c].fillna(m)

    df['name'] = df['name'].apply(get_mark)
    df = pd.get_dummies(df)

    new_df = pd.DataFrame()
    for c in columns_after_get_dummies:
        if c not in df.columns:
            new_df[c] = [False] * df.shape[0]
        else:
            new_df[c] = df[c]

    for arr in feature_history:
        inx = arr[0]
        c1 = arr[1]
        c2 = arr[2]

        if inx == 0:
            new_df[f"{c1}_prod_{c2}"] = new_df[c1] * new_df[c2]
        elif inx == 1:
            try:
                new_df[f"{c1}_div_{c2}"] = new_df[c1] / new_df[c2]
            except BaseException:
                new_df[f"{c1}_div_{c2}"] = 0
        elif inx == 2:
            try:
                new_df[f"{c2}_div_{c1}"] = df[c2] / df[c1]
            except BaseException:
                new_df[f"{c2}_div_{c1}"] = 0

        new_df.replace([np.inf, -np.inf], 0, inplace=True)

    scaler = StandardScaler()
    columns = new_df.columns
    scaler.fit(new_df)
    new_df = pd.DataFrame(scaler.transform(new_df), columns=columns)

    return new_df

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
  df = transform_df(item_to_df(item))
  res = float(best_model.predict(df)[0])
  return res


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
  res = list()
  for item in items:
    pred = predict_item(item)
    res.append(pred)

  return res


if __name__ == "__main__":
     import uvicorn
     uvicorn.run(app, host="localhost", port=1024)

