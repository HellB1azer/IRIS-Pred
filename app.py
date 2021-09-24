import streamlit as st
import sklearn
import pickle

st.title("IRIS ML Web APP")
classes=['setosa', 'versicolor', 'virginica']
sepal_l=st.slider("Enter Sepal Length",0,9)
sepal_w=st.slider("Enter Sepal Width",0,9)
petal_l=st.slider("Enter Petal Length",0,9)
petal_w=st.slider("Enter Petal Width",0,9)

loadedmodel=pickle.load(open('Final Model.pkl','rb'))
pred=loadedmodel.predict([[sepal_l,sepal_w,petal_l,petal_w]])

st.write("Class of IRIS Plant is: ",classes[pred[0]])
