# Put the code for your API here.
import pickle
import pandas as pd
from .starter.ml.model import inference
from .starter.ml.data import process_data
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os

# DVC on Heroku - required code
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc remote add -d myremote s3://udacity3bucket/storage/")
    if os.system("dvc pull -r myremote") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


model_pth = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'model/model.pickle')
encoder_pth = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'model/encoder.pickle')
lb_pth = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'model/lb.pickle')


class DataItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')
    salary: str

    class Config:
        schema_extra = {
            "example": {
                "age": 44,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2147,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
                "salary": "<=50K"
            }
        }


model = pickle.load(open(model_pth, 'rb'))
encoder = pickle.load(open(encoder_pth, 'rb'))
lb = pickle.load(open(lb_pth, 'rb'))


cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

app = FastAPI()

# Define a GET on the specified endpoint.


@app.get("/")
async def say_welcome():
    return {"greeting": "Welcome!"}


@app.post("/predict")
async def predict(item: DataItem):
    df = pd.DataFrame(item.dict(), index=[0])

    X_test, _, _, _ = process_data(
        df, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb)

    pred = lb.inverse_transform(inference(model, X_test))[0]

    return {"prediction": pred}
