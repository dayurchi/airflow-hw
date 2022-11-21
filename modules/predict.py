import os
import datetime
import dill
import pandas as pd
import json

from pydantic import BaseModel
from datetime import datetime


path = os.environ.get('PROJECT_PATH', '..')


# create class
class Form(BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str


# create predict function
def predict():
    # create function to load jsons for predictions
    def load_json():
        file_path = f'{path}/data/test'
        predict_list = []
        for root, dirs, files in os.walk(file_path):
            for f in files:
                with open(file_path + '/' + f) as num:
                    predict_list.append(json.load(num))
            return predict_list
    # get pickle for base model
    name = os.listdir(f'{path}/data/models')
    with open(f'{path}/data/models/{name[0]}', 'rb') as file:
        model = dill.load(file)

    # take class Form to create prediction frame
    def make_predict(form: Form):
        df = pd.DataFrame(form, index=[0])
        y = model.predict(df)
        return {
            'car_id': form['id'],
            'pred': y[0]
        }
    # make prediction with jsons and model
    predict_list = load_json()
    predict_df = pd.DataFrame(columns=['car_id', 'pred'], index=[0])
    for item in predict_list:
        predict_result_df = pd.DataFrame(make_predict(item),  index=[0])
        predict_df = pd.concat([predict_df, predict_result_df], ignore_index=True)
    predict_df = predict_df.drop(labels=[0], axis=0)
    # save result to csv
    filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    predict_df.to_csv(filename, index=False)


if __name__ == '__main__':
    predict()
