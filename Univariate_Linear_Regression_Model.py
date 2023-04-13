import pandas as pd
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import math

# Reading the CSV Datapoints File from the command line input
filename = input("Enter name of input file: ")
sqftrentA = pd.read_csv(filename)

#Function to apply the line equation
def y_pred(x):
    a1 , a2 = sym.symbols('a1 a2')
    return a1*x+a2

#Applying the General (y=mx+c) equation for the Y values in sqftrentA
sqftrentP = sqftrentA.apply(y_pred)

#Calculating error value by subtraction Y actual from Y prediction
E = pd.Series(sqftrentA['Y'].subtract(sqftrentP['X']))

#Applying Square to remove the ambiguity in sign (+/-)
ET = pd.Series(E*E)

#Assigning Sympy symobols for the variable a1 and a2
a1= sym.symbols('a1')
a2= sym.symbols('a2')

#Calculating error summation
function= ET.sum()

#Applying partial derivative for a1 and a2
partialderiv_a1= sym.Derivative(function, a1)
partialderiv_a2= sym.Derivative(function, a2)

#Storing that value in ET_a1 and ET_a2
ET_a1 = partialderiv_a1.doit()
ET_a2 = partialderiv_a2.doit()

#Solving the equation to find a1 and a2
sym.solve((ET_a1,ET_a2), (a1, a2))

#Storing a1 and a2 value in a pandas frame
sol_a1_a2 = pd.Series(sym.solve((ET_a1, ET_a2), (a1, a2)))

#Calculating Accuracy
eq3 = ET.sum()/ET.count()
S = eq3.subs([(a1,sol_a1_a2[0]), (a2,sol_a1_a2[1])])
accuracy = math.sqrt(S)

#Applying X value in the equation y=mx+c to calulate the error percentage
#Getting the value of X from command line
b = input("Enter X value to calculate the Y value: ")
b1 = float(b)
y = sol_a1_a2[0]*b1+sol_a1_a2[1]

#changing the datatype to float
y1 = float(y)

#Calculating error percentage
error = (abs(accuracy-y1)/y1)*100

#plotting the graph using matplotlib
x = np.linspace(0, 16000, 50)
y = sol_a1_a2[0]*x+sol_a1_a2[1]
plt.scatter(sqftrentA.X, sqftrentA.Y, color="grey",label = 'Training set')
plt.plot(x, y, '-r', label='Curve Fit')
plt.title('Univariate Linear Regression')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()

#printing Y value and accuracy percentage of the model
print("Y Value", y1)
print("Accuracy Percentage of this Model", round(error,2),"%")
