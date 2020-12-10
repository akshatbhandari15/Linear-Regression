# -*- coding: utf-8 -*-
import pandas as pd
import numpy as nm

#Loading training dataset
feature_set = pd.read_csv('CarPrice_Assignment.csv')
feature_set = feature_set.drop(['car_ID', 'CarName'], axis=1)
car_prices = feature_set.price
feature_set = feature_set.drop(['price'], axis = 1)

#assigning categorical variables a value
feature_set = feature_set.replace(to_replace="gas", value=1)
feature_set = feature_set.replace(to_replace="diesel", value=2)

feature_set = feature_set.replace(to_replace="std", value=1)
feature_set = feature_set.replace(to_replace="turbo", value=2)

feature_set = feature_set.replace(to_replace="std", value=1)
feature_set = feature_set.replace(to_replace="turbo", value=2)


feature_set = feature_set.replace(to_replace="two", value=1)
feature_set = feature_set.replace(to_replace="four", value=2)
feature_set = feature_set.replace(to_replace="six", value=3)
feature_set = feature_set.replace(to_replace="eight", value=4)
feature_set = feature_set.replace(to_replace="three", value=5)
feature_set = feature_set.replace(to_replace="five", value=6)
feature_set = feature_set.replace(to_replace="twelve", value=7)



feature_set = feature_set.replace(to_replace="convertible", value=1)
feature_set = feature_set.replace(to_replace="hatchback", value=2)
feature_set = feature_set.replace(to_replace="sedan", value=3)
feature_set = feature_set.replace(to_replace="wagon", value=4)
feature_set = feature_set.replace(to_replace="hardtop", value=5)

feature_set = feature_set.replace(to_replace="4wd", value=1)
feature_set = feature_set.replace(to_replace="fwd", value=2)
feature_set = feature_set.replace(to_replace="rwd", value=3)

feature_set = feature_set.replace(to_replace="front", value=1)
feature_set = feature_set.replace(to_replace="rear", value=2)

feature_set = feature_set.replace(to_replace="dohc", value=1)
feature_set = feature_set.replace(to_replace="ohcv", value=2)
feature_set = feature_set.replace(to_replace="ohc", value=3)
feature_set = feature_set.replace(to_replace="l", value=4)
feature_set = feature_set.replace(to_replace="rotor", value=5)
feature_set = feature_set.replace(to_replace="ohcf", value=6)
feature_set = feature_set.replace(to_replace="dohcv", value=7)

feature_set = feature_set.replace(to_replace="1bbl", value=1)
feature_set = feature_set.replace(to_replace="2bbl", value=2)
feature_set = feature_set.replace(to_replace="4bbl", value=3)
feature_set = feature_set.replace(to_replace="idi", value=4)
feature_set = feature_set.replace(to_replace="mfi", value=5)
feature_set = feature_set.replace(to_replace="mpfi", value=6)
feature_set = feature_set.replace(to_replace="spdi", value=7)
feature_set = feature_set.replace(to_replace="spfi", value=8)


#converting DataFrame and Series to NumPy Array
feature_set = feature_set.to_numpy()



print (car_prices)
print (feature_set)
#Hypothesis Function
def hyp(theta):
    h= nm.dot(feature_set, theta)
    return h
#Cost Function
def cost(m, theta):
    j = car_prices- hyp(theta)
    j = nm.square(j)
    j = j/(2*m)
    return j
#Function for calling Gradient Descent
def gradDescent(theta, alpha, m):
    print (cost(m, theta))
    term = car_prices- hyp(theta)
    term = nm.dot(term, feature_set)
    term = term*(alpha/m)
    theta = theta - term
    return theta
       
