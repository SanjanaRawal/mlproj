import os 
import sys 
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig : 
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer : 
    def __init__(self) : 
        self.model_trainer_config = ModelTrainerConfig()
    logging.info("Model trainer initiated to find best model.")
    def initiate_model_trainer(self, train_array, test_array) : 
        logging.info("Model trainer initiated to find best model.")
        try : 
            logging.info("Split train and test input data.")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            #train and test array comes from data_transformation component
            #last feature that is added in train, test array is target feature
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            #Hyper parameter tuning the models
            params = {
                "Random Forest" : {
                    "n_estimators" : [8,16,32,64,128,256],
                    # "max_features" : ["sqrt" , "log2", None],
                    # "criterion" : ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                },
                "Decision Tree" : {
                    "criterion" : ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    #"splitter" : ["best", "random"],
                    #"max_features" : ["sqrt", "log2"]
                },
                "Gradient Boosting" : {
                    # "loss" : ["huber", "squared_error", "absolute_error", "quantile"],
                    "learning_rate" : [0.1, 0.01, 0.05, 0.001], 
                    "subsample" : [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # "criterion" : ["squared_error", "friedman_mse"],
                    # "max_features" : ["auto", "sqrt", "log2"],
                    "n_estimators" : [8,16,32,64,128,256]
                },
                "Linear Regression" : {
                    # "fit_intercept" : [True, False],
                    # "normalize" : [True, False]
                },
                "K-Neighbors Regressor" : {
                    "n_neighbors" : [5, 7, 9, 11],
                    "weights" : ["uniform", "distance"],
                    "algorithm" : ["auto", "ball_tree", "kd_tree", "brute"]
                },
                "XGBRegressor" : {
                    "learning_rate" : [0.1, 0.01, 0.05, 0.001], 
                    "n_estimators" : [8,16,32,64,128,256]
                },
                "CatBoosting Regressor" : {
                    "depth" : [6, 8, 10],
                    "learning_rate" : [0.1, 0.01, 0.05, 0.001], 
                    "iterations" : [30, 50, 100]
                },
                "AdaBoost Regressor" : {
                    "learning_rate" : [0.1, 0.01, 0.05, 0.001], 
                    # "loss" : ["linear", "square", "exponential"],
                    "n_estimators" : [8,16,32,64,128,256]
                }
            }
            
            model_report:dict=evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, param=params)
            #to get best model score from dict 
            best_model_score = max(sorted(model_report.values()))
            #to get best model name from dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score < 0.6 :
                raise CustomException("No best model found")
            logging.info("Best model found on both training and testing dataset.")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            #the best model will be saved as model.pkl file in artifacts folder
            y_pred = best_model.predict(x_test)
            r2_square = r2_score(y_test, y_pred)
            return r2_square
            
        except Exception as e : 
            raise CustomException(e, sys)    
    