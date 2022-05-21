#%%

import pandas as pd
import pickle
import os
import csv
import pdfplumber
from gama import GamaClassifier
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import f1_score

class main(object):
    def __init__(self, directory):
        self.directory = directory
        self.csv_df = None
        self.json_df = None
        self.pdf_df = None
        self.automl = None
    
    def __call__(self):
        self.csv_df = self._read_csv(self.directory)
        self.json_df = self._read_json(self.directory)
        self.pdf_df = self._read_pdf(self.directory)
        df_together = pd.concat([self.csv_df, self.json_df, self.pdf_df], ignore_index=True, sort=False)
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_preprocessing(df_together)
        self.challenge_GAMA_pipeline_train()
        
    def _read_csv(self, directory):
        df_list =[]
        for file in os.listdir(directory + 'data/' ):
            if file.endswith(".csv"):
                if 'train' in file:
                    try:
                        df = pd.read_csv(directory + 'data/' + file)
                    except:
                        df = pd.read_csv(directory + 'data/' + file, sep=";")
                    df_list.append(df)
        return pd.concat(df_list, ignore_index=True, sort=False)
    
    
    def _read_json(self, directory):
        json_df_list = [] 
        for file in os.listdir(directory + 'data/' ):
            if file.endswith(".json"):
                j_son_df = pd.read_json(directory  + 'data/' + file)
                print(len(j_son_df))
                json_df_list.append(j_son_df)
        final_json = pd.concat(json_df_list, ignore_index=True, sort=False)
        names_csv = list(self.csv_df.columns)                
        return final_json[names_csv]
    
    def _read_pdf(self, directory):

        df_list =[]
        for file in os.listdir(directory +"Entrega/" ):
            if file.endswith(".csv"):
                if 'train' in file:
                    try:
                        df = pd.read_csv(directory + file)
                    except:
                        df = pd.read_csv(directory + file, sep=";")
                    df_list.append(df)
        return pd.concat(df_list, ignore_index=True, sort=False)

    def data_preprocessing(self, df_raw):
        list_drop = ['FacilityInspireID',
                'facilityName',
                'targetRelease', 
                'CITY ID',
                'REPORTER NAME',
                'EPRTRAnnexIMainActivityLabel',
                'eprtrSectorName',
                'CONTINENT']
        ]
        
        df = df_raw.drop(list_drop, axis=1)
        print(df)
        df.to_csv('final.csv', index=False)
        y = df['pollutant']
        list_predictors = list(df.columns)
        list_predictors.remove('pollutant')
        X = df[list_predictors]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

        return X_train, X_test, y_train, y_test
        
    def challenge_GAMA_pipeline_train(self):
        # Useless columns

        automl = GamaClassifier(max_total_time=180, store="nothing")
        print("Starting `fit` which will take roughly 3 minutes.")
        automl.fit(self.X_train, self.y_train)
    
        label_predictions = automl.predict(self.X_test)
        probability_predictions = automl.predict_proba(self.X_test)
    
        print('accuracy:', accuracy_score(self.y_test, label_predictions))
        print('log loss:', log_loss(self.y_test, probability_predictions))
        # the `score` function outputs the score on the metric optimized towards (by default, `log_loss`)
        print('log_loss', automl.score(self.X_test, self.y_test))
        print('f1-score', f1_score(self.y_test, label_predictions, average='macro'))
        filename = 'classification.sav'
        pickle.dump(automl.model, open(filename, 'wb'))
        self.automl = automl
        return automl


    def generate_csv_from_PDF_files(self, directory):
        listColumnsInCSV = [
        'countryName', 'eprtrSectorName', 'EPRTRAnnexIMainActivityLabel',
        'FacilityInspireID', 'facilityName', 'City', 'targetRelease',
        'pollutant', 'reportingYear', 'MONTH', 'DAY', 'CONTINENT',
        'max_wind_speed', 'avg_wind_speed', 'min_wind_speed', 'max_temp',
        'avg_temp', 'min_temp', 'DAY WITH FOGS', 'REPORTER NAME', 'CITY ID'
        ]
        
        listPDFFiles = os.listdir(directory + "data/train6/")
        
        valuesForDict = list()
        # replace text
        listColumnsFromPDFToChange = [
                    ["FACILITY NAME", "facilityName"],
                    ["DAYS FOG", "DAY WITH FOGS"],
                    ["COUNTRY:", "countryName:"],
                    ["CITY",  "City"],
                    ["City_ID", "CITY ID:"], 
                    ["CONTINENT", "\nCONTINENT"],
                    ["MONTH", "\nMONTH"],
                    ["YEAR", "\nYEAR"],
                    ["min_temp", "\nmin_temp" ],
                    ["avg_temp", "\navg_temp"],
                    ["min_wind_speed", "\nmin_wind_speed"],
                    ["avg_wind_speed", "\navg_wind_speed"],
                    ["pollutant", "\npollutant"],
                ]
        
        # read pdf files
        for PDFFile in listPDFFiles:
            with pdfplumber.open(directory + "data/train6/" + PDFFile) as temp:
                first_page = temp.pages[0]
                text_first_page = first_page.extract_text()
                
                for column in listColumnsFromPDFToChange:
                    text_first_page = text_first_page.replace(column[0], column[1])
        
                keys, values = list(), list()
                txt_page_divided_by_line = text_first_page.split("\n")
                for val in txt_page_divided_by_line:
                    if val.startswith(tuple(listColumnsInCSV)):
                        keys.append(val.split(":")[0].strip())
                        #Get the data from the PDF plus strip the scientific notation
                        values.append(val.split(":")[1].strip().split("E+")[0])       
                valuesForDict.append(values)
        
        with open(directory + "Entrega/train6PDFs.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(valuesForDict)
        
        with open(directory + "Entrega/train6PDFs.csv", "r+") as f:
            texto=f.read()
            f.truncate(0)
            f.write(texto.replace("\n\n","\n"))

    def make_predictions(self, automl):
        columnsTest = [
            "countryName","eprtrSectorName","City","pollutant","reportingYear","MONTH","DAY",
            "CONTINENT","max_wind_speed","avg_wind_speed","min_wind_speed","max_temp","avg_temp",
            "min_temp","DAY WITH FOGS"
        ]     
        df = pd.read_csv("data/test_x.csv") 
        dfTest = df.filter(columnsTest)   

        predictions = automl.predict(dfTest)

        return predictions 


if __name__ == '__main__':
    path_use = os.getcwd()
    path = path_use.replace(os.sep, '/') + '/'
    challenge = main(path)
    model = challenge()
    automl = model.challenge_GAMA_pipeline_train(model.csv_df)
    automl.make_predictions()
# %%
