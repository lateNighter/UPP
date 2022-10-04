import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import requests

class Model:
  def __init__(self, model):
    self.model = model

  def predict(self, sample):
    '''
    Parameters:
    sample (DataFrame): Year: int; Rooms: int; TotalArea: float; LivingArea: float; 
    Floor(actual floor/total floors): float.
  
    Returns:
    float: Predicted price
    '''
    sample=sample.to_numpy().reshape(1,-1)
    sample=(sample-np.mean(sample)) / np.std(sample)
    res = (self.model.predict(sample)*1000000)[0]
    return res

if __name__ == "__main__":
    path = "https://raw.github.com/lateNighter/UPP/main/model.pkl"
    response = requests.get(path, allow_redirects=True)
    if(response.status_code==requests.codes.ok):
        open('model.pkl', 'wb').write(response.content)

    pickled_model = pickle.load(open('model.pkl', 'rb'))

    df=pd.DataFrame([[2019,2,40.0,27.0,1]])
    print(pickled_model.predict(df))