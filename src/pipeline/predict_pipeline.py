import sys
import os
import pandas as pd
from src.exception import customexception
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts',"preprocessor.pkl")
            model_path=os.path.join('artifacts',"model.pkl")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            scaled_data = preprocessor.transform(features)
            pred = model.predict(scaled_data)
            return pred
        except Exception as e:
            raise customexception(sys,e)


class CustomData:
    def __init__(self,
                 age: int,
                 sex : int,
                 chest_pain_type : int,
                 cholestrol:int,
                 resting_blood_pressure : int,
                 fasting_blood_sugar : int,
                 resting_electrocardiogram : int,
                 max_heart_rate_achieved:int,
                 exercise_induced_angina:int,
                 st_depression : float,
                 st_slope : int,
                 num_major_vessels : int,
                 thalassemia : int):
        logging.info("data entry in this function")

        self.age = age
        self.sex = sex
        self.chest_pain_type = chest_pain_type
        self.cholestrol = cholestrol
        self.resting_blood_pressure = resting_blood_pressure
        self.fasting_blood_sugar = fasting_blood_sugar
        self.resting_electrocardiogram = resting_electrocardiogram
        self.max_heart_rate_achieved = max_heart_rate_achieved
        self.exercise_induced_angina = exercise_induced_angina
        self.st_depression = st_depression
        self.st_slope = st_slope
        self.num_major_vessels = num_major_vessels
        self.thalassemia = thalassemia

    def get_data_as_dataframe(self):
        try:
            logging.info("entry in dataframe function")
            custom_data_input_dict = {

                'age':[self.age],
                'sex':[self.sex],
                'chest_pain_type':[self.chest_pain_type],
                'cholestrol':[self.cholestrol],
                'resting_blood_pressure':[self.resting_blood_pressure],
                'fasting_blood_sugar':[self.fasting_blood_sugar],
                'resting_electrocardiogram':[self.resting_electrocardiogram],
                'max_heart_rate_achieved':[self.max_heart_rate_achieved],
                'exercise_induced_angina':[self.exercise_induced_angina],
                'st_depression':[self.st_depression],
                'st_slope':[self.st_slope],
                'num_major_vessels':[self.num_major_vessels],
                'thalassemia':[self.thalassemia]

                       }
            logging.info("exit from dictinary")
            data = pd.DataFrame(custom_data_input_dict)
            logging.info("dataframe has been gathered")
            return data
        except Exception as e:
            raise customexception(sys,e)

