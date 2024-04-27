src: source code

EDA_and_feature_engineering.ipynb:
- This notebook contains exploratory data analysis and feature engineering
- it unzips the data, then converts it from csv to parquet
- it then loads the data into a pandas dataframe, 
- preprocesses it and then performs analysis & feature engineering
- saves as a parquet file for use in modelling

modelling_notebook.ipynb:
- run all the models from a single notebook
- main code for each model is in the individual package subdirectories
- we just call the model to either train or predict from the notebook

Historic Files:
- prophet_model.py is a single file version of the model and inference
- XGBoost single file
- LSTM single file is the code we got working
- NEM_etl.py is a script to download public data from from the AEMO website, unzip and assemble it.
- price_prep is the script to prepare the price data to add to our modelling dataframe
- 