import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from src.logger import logging
from src.exception import customexception
from src.utils import save_object


@dataclass

class datatransformationconfig:
    preprocessor_obj_path = os.path.join("artifacts",'preprocessor.pkl')

class datatransformation:
    def __init__(self):
        self.data_transformation = datatransformationconfig()

    def get_data_transform_object(self):
        try:
            logging.info("Data Transformation initiated")
            numerical_cols = ['age', 'sex', 'chest_pain_type', 'cholestrol',
       'resting_blood_pressure', 'fasting_blood_sugar',
       'resting_electrocardiogram', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope',
       'num_major_vessels', 'thalassemia']
            logging.info("Numerical Pipeline Initiated")

            ## Numerical pipleine

            num_pipeline = Pipeline(
                steps=[('impute',SimpleImputer(strategy='most_frequent')),
                       ('scaler',StandardScaler(with_mean=False))]
            )

            preprocessor = ColumnTransformer(
                [("num_pipeline",num_pipeline,numerical_cols)]
            )
            return preprocessor
        except Exception as e:
            raise customexception(e,sys)

    def initiate_data_ingestion(self,train_path,test_path):
        try :
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")

            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transform_object()

             ## Extracting input feature (x) and target feature (y)
            target_column_name = "target"
            input_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_train_df = train_df[target_column_name]

            input_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_test_df = test_df[target_column_name]

            logging.info("splitting input and target features are done")

            ## Applying preproceesing object in these extracted features

            input_train_df_arr = preprocessing_obj.fit_transform(input_train_df)
            input_test_df_arr = preprocessing_obj.transform(input_test_df)
            logging.info("Applying preprocessing in training and test data")

             ## combining input features and target

            train_arr = np.c_[input_train_df_arr,np.array(target_train_df)]
            test_arr = np.c_[input_test_df_arr,np.array(target_test_df)]

            save_object(file_path=self.data_transformation.preprocessor_obj_path,obj=preprocessing_obj)
            logging.info("preprocessing pickle file saved")

            return train_arr,test_arr
        except Exception as e:
            raise customexception(e,sys)
