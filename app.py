import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('kmeans_cluster.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('Social_Network_Ads.csv')
# Extracting independent variable:
X = dataset.iloc[:,[2,3]].values
# Encoding the Independent Variable
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(Age,EstimatedSalary):
  output= model.predict(sc.transform([[Age,EstimatedSalary]]))
  print(output)
  if output==[0]:
    prediction="Customer is careless"

  elif output==[1]:
    prediction="Customer is standard"
  elif output==[2]:
    prediction="Customer is Target"
  elif output==[3]:
    prediction="Customer is careful"

  else:
    prediction="Custmor is sensible" 

  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:Red;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">KMeans Clustering Example</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    Age = st.number_input('Insert a Age',18,100)
    EstimatedSalary=st.number_input('Insert a EstimatedSalary',100,10000000)
    result=""
    
    if st.button("Predict"):
      result=predict_note_authentication(Age,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Prashant Jain")
      st.subheader("Student , Poornima Group Of Insititution")

if __name__=='__main__':
  main()
