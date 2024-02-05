import os
import sys
from dataclasses import dataclass
from src.logger import logging
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from src.exception import customexception
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils import save_object,evaluate_model

@dataclass
class model_training_config:
    train_model_path = os.path.join("artifacts","model.pkl")

class modeltrainer:
    def __init__(self):
        self.model_trainer = model_training_config()

    def initiatemodeltrainer(self,train_array,test_array):
        try:
            logging.info("splitting testing and training data")

            X_train,Y_train,X_test,Y_test = (

                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
             )
            models = {

                "LogisticRegression": LogisticRegression(),
                "Random Forest Classifier": RandomForestClassifier(n_estimators=10,random_state=12,max_depth=5),
                "Naive Bayes": GaussianNB(),
                "Decision Tree": DecisionTreeClassifier(criterion="entropy",random_state=0,max_depth=6),
                "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=10)
            }

            model_report:dict = evaluate_model(X_train,Y_train,X_test,Y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report: {model_report}')

            # To get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}, Accuracy Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, Accuracy Score: {best_model_score}')

            save_object(
                 file_path=self.model_trainer.train_model_path,
                 obj=best_model)
            predicted = best_model.predict(X_test)





        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise customexception(e,sys)



