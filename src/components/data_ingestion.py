import os
import sys
from src.exception import customexception
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import datatransformation
from src.components.data_transformation import datatransformationconfig
from src.components.model_trainer import model_training_config
from src.components.model_trainer import modeltrainer

@dataclass
class dataingestionconfig:
    train_data_path : str = os.path.join("artifacts","train.csv")
    test_data_path : str = os.path.join("artifacts","test.csv")
    raw_data_path : str = os.path.join("artifacts","raw_data.csv")

class dataingestion:
    def __init__(self):
        self.ingestion_config = dataingestionconfig()

    def initiateingestion(self):
        logging.info("Entered the data ingestion method or componenet")
        try:
            data = pd.read_csv('notebook/data/cleaned_data.csv')
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index = False)
            logging.info("Created the raw data file")

            logging.info("splitting the data")
            train_data , test_data = train_test_split(data,test_size=0.2,random_state=40)
            logging.info("data is splitted")

            train_data.to_csv(self.ingestion_config.train_data_path,index = False)
            test_data.to_csv(self.ingestion_config.test_data_path,index = False)
            logging.info("train and test data files are created")
            logging.info("ingestion has been completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,

            )
        except Exception as e:
            raise customexception(e,sys)


if __name__ == "__main__":
    obj=dataingestion()
    train_data,test_data = obj.initiateingestion()

    transform_obj = datatransformation()
    train_arr,test_arr = transform_obj.initiate_data_ingestion(train_data,test_data)

    trainer_obj = modeltrainer()
    trainer_obj.initiatemodeltrainer(train_arr,test_arr)

