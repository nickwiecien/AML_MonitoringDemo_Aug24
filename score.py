import pandas as pd
import json
import mlflow
import mlflow.xgboost
from azureml.ai.monitoring import Collector
import os
import os

def init():
    global inputs_collector, outputs_collector, inputs_outputs_collector, loaded_model
    model_path= 'model' 
    print(os.listdir('.'))
    # print(os.listdir('./azureml-models'))
    loaded_model = mlflow.xgboost.load_model(model_path)

    # instantiate collectors with appropriate names, make sure align with deployment spec
    inputs_collector = Collector(name='model_inputs')                    
    outputs_collector = Collector(name='model_outputs')

def run(data): 
  
    data = preprocess(data)
  
    input_df = pd.DataFrame(data)

    # collect inputs data, store correlation_context
    context = inputs_collector.collect(input_df)
    
    input_df = filter_data(input_df)

    # perform scoring with pandas Dataframe, return value is also pandas Dataframe
    output = loaded_model.predict(input_df) 

    input_df['predictions'] = output

    print(output)

    # collect outputs data, pass in correlation_context so inputs and outputs data can be correlated later
    outputs_collector.collect(input_df['predictions'].to_frame(), context)
  
    return json.dumps(output.tolist())
  
def preprocess(src_data):
    data = src_data
    try:
        data = json.loads(data)
    except Exception as e:
        pass
    data = data['data']
    try:
        data = (json.loads(data))
    except Exception as e:
        pass 
    return data

def filter_data(df):
    input_df = df.copy()
    zone_cols = [x for x in df.columns if 'Zone' in x]
    input_df = input_df.drop(columns=zone_cols)
    try:
        input_df = input_df.drop(columns=['Datetime'])
    except Exception as e:
        pass
    return input_df
    
    