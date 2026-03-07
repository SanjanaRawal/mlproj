import os 
import sys
import pandas as pd 
from sklearn.model_selection import train_test_split 
from src.exception import CustomException
from src.logger import logging 
from dataclasses import dataclass #used to create class variables

@dataclass
# Any input required in data ingestion component can be given here
class DataIngestionConfig : 
    train_data_path : str = os.path.join("artifacts", "train.csv")
    test_data_path : str = os.path.join("artifacts", "test.csv")
    raw_data_path : str = os.path.join("artifacts", "data.csv")
    #tells the system to save the train, test and raw data in artifacts folder

class DataIngestion : 
        def __init__(self) : 
            self.ingestion_config = DataIngestionConfig()
        def initiate_data_ingestion(self) : 
            #to read data from various sources 
            logging.info("Entered the data ingestion method.")
            try : 
                df = pd.read_csv("notebook\\data\\stud.csv")
                #Above line can be changed when the data source is different
                logging.info("Read the dataset as a dataframe")
                os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
                df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
                logging.info("Train test split initiated")
                train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
                train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
                test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
                logging.info("Data ingestion is completed.")
                return (
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                )
            except Exception as e: 
                raise CustomException(e, sys)
            
if __name__ == "__main__" :
    obj = DataIngestion()
    obj.initiate_data_ingestion()     
            
    