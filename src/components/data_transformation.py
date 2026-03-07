import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer #to create pipeline
from sklearn.impute import SimpleImputer #to handle missing values 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object

@dataclass
class DataTransformationConfig: 
    #will give any path that is required in data transformation component
    #used to give input to data transformation component
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")
    #to create any model and sace it into pickle file/model file

class DataTransformation : 
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation based on different types of data.
        '''
        #to create all pickle files, which will be responsible for data type conversion, standard scaling etc
        try : 
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
            
            num_pipeline = Pipeline(
                #create a numerical pipeline that is doing 2 imp tasks
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    #1. responsible for handling missing values in numerical columns
                    ("scaler", StandardScaler())
                    #2. responsible for standard scaling of numerical columns
                ]
            ) # needs to be run on training dataset
            logging.info("Numerical columns encoding completed.")
            
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy= "most_frequent")),
                    #handling missing values in categorical features 
                    ("one_hot_encoder", OneHotEncoder()),
                    #converting categorical features into numerical features
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns encoding completed.")
            
            preprocessor = ColumnTransformer(
                #to combine both the pipelines
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor   
        except Exception as e : 
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path) : 
        try : 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading of train and test data is completed")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math score"
            numerical_columns = ["writing score", "reading score"]
            input_features_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            input_features_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]
            logging.info("Applying preprocessing object on training and testing dataframe.")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saved preprocessing object.")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e: 
            raise CustomException(e, sys)
