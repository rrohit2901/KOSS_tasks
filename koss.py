import numpy as np
from sklearn import linear_model, preprocessing

data_path = 'imports-85'

# Generating 

array1 = np.genfromtxt(data_path, delimiter=',', autostrip=True, usecols=(0,1,9,10,11,12,13,16,18,19,20,21,22,23,24), filling_values=0, skip_footer=2)
n_array1 = preprocessing.normalize(array1)
array2 = np.genfromtxt(data_path, delimiter=',', autostrip=True, usecols=(25), filling_values=0, skip_footer=2)
reg = linear_model.LinearRegression()
reg.fit(n_array1,array2)
r_matrix = reg.coef_

#Testing model

t_array1 = np.genfromtxt(data_path, delimiter=',', autostrip=True, usecols=(0,1,9,10,11,12,13,16,18,19,20,21,22,23,24), filling_values=0, skip_header=203)
t_narray1 = preprocessing.normalize(t_array1)
t_array2 = np.genfromtxt(data_path, delimiter=',', autostrip=True, usecols=(25), filling_values=0, skip_header=203)
pred_price = reg.predict(t_narray1)


print("Predicted output:-")
print(pred_price)
print("Actual prices of cars:-")
print(t_array2)
print("Regression matrix:-")
print(r_matrix)