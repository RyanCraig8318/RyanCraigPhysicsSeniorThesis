import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

e = 1.602e-19  # Coulombs
K_b=1.381e-23 #Joules/Kelvin
m_e=9.109e-31 #kilograms
file = pd.read_csv('/Users/ryancraig/Documents/Capstone/data/sampledata1.csv')
V = file['V']
I = file['I']
V_cut = file['V']
I_cut = file['I']
for i in range(2, len(I_cut)):
    if I_cut[i] < I_cut[i-1] < I_cut[i-2]:
        V_cut = V_cut[:i-2]
        I_cut = I_cut[:i-2]
        break

def InitialPlot():
    plt.figure(figsize=(10, 6))
    plt.plot(V, I, 'o', color='blue')  
    plt.xlabel('Voltage Input (Volts)')
    plt.ylabel('Current Output (Amperes)')
    plt.title('IV Curve from Dr. Thakur\'s Langmuir Probe Data')
    plt.legend()
    plt.grid(True)
    plt.show()

def Sigmoidcurvefit():
    plt.figure(figsize=(10, 6))
    plt.plot(V, I, 'o', color='red') 
    def sigmoid(V_cut, A, k, V_0, B):
        return A / (1 + np.exp(-k * (V_cut - V_0))) + B
    initial_guess = [max(I_cut), 1, np.median(V_cut), min(I_cut)]
    paramaters, covariance = curve_fit(sigmoid, V_cut, I_cut, p0=initial_guess)
    A, k, V_0, B = paramaters
    V_cut_fit = np.linspace(min(V_cut), max(V_cut), 100)
    I_cut_fit = sigmoid(V_cut_fit, *paramaters)
    plt.plot(V_cut_fit, I_cut_fit, label='Sigmoid Fit', color='blue')
    plt.xlabel('Voltage Input (Volts)')
    plt.ylabel('Current Output (Amperes)')
    plt.title('IV Curve with Sigmoidal Fit from Dr. Thakur\'s Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    print('Sigmoid fit A / (1 + np.exp(-k * (V_cut - V_0))) + B')
    print(f"Fitted Parameters:\n A = {A}\n k = {k}\n V_0 = {V_0}\n B = {B}")

def slope(V):
    def sigmoid(V_cut, A, k, V_0, B):
        return A / (1 + np.exp(-k * (V_cut - V_0))) + B
    initial_guess = [max(I_cut), 1, np.median(V_cut), min(I_cut)]
    paramaters, covariance = curve_fit(sigmoid, V_cut, I_cut, p0=initial_guess)
    A, k, V_0, B = paramaters
    return (A*k*(np.exp(-k*(V-V_0))))/((1+(np.exp(-k*(V-V_0))))**2)

def slopeatV0():
    def sigmoid(V_cut, A, k, V_0, B):
        return A / (1 + np.exp(-k * (V_cut - V_0))) + B
    initial_guess = [max(I_cut), 1, np.median(V_cut), min(I_cut)]
    paramaters, covariance = curve_fit(sigmoid, V_cut, I_cut, p0=initial_guess)
    A, k, V_0, B = paramaters
    return(slope(V_0))

def slopeatV0xint():
    def sigmoid(V_cut, A, k, V_0, B):
        return A / (1 + np.exp(-k * (V_cut - V_0))) + B
    initial_guess = [max(I_cut), 1, np.median(V_cut), min(I_cut)]
    paramaters, covariance = curve_fit(sigmoid, V_cut, I_cut, p0=initial_guess)
    A, k, V_0, B = paramaters
    x=(((A*k)/4)-B)**-1
    return V_0 - (np.log(A*x-1))/(k)

def slopeatV0yint():
    def sigmoid(V_cut, A, k, V_0, B):
        return A / (1 + np.exp(-k * (V_cut - V_0))) + B
    initial_guess = [max(I_cut), 1, np.median(V_cut), min(I_cut)]
    paramaters, covariance = curve_fit(sigmoid, V_cut, I_cut, p0=initial_guess)
    A, k, V_0, B = paramaters
    return -slopeatV0()*slopeatV0xint()

def slopeplot():
    def sigmoid(V_cut, A, k, V_0, B):
        return A / (1 + np.exp(-k * (V_cut - V_0))) + B
    initial_guess = [max(I_cut), 1, np.median(V_cut), min(I_cut)]
    paramaters, covariance = curve_fit(sigmoid, V_cut, I_cut, p0=initial_guess)
    A, k, V_0, B = paramaters
    plt.figure(figsize=(10, 6))
    plt.plot(V, I, 'o', label='IV Data', color='red')  
    plt.xlabel('Voltage Input (Volts)')
    plt.ylabel('Current Output (Amperes)')
    plt.title('IV Curve from Dr. Thakur\'s Data')
    plt.plot(np.linspace(-5, 20, 100)+slopeatV0xint(), slopeatV0()*(np.linspace(-5, 20, 100)), label='Slope Estimate', color='green')
    plt.legend()
    plt.grid(True)
    plt.show()

def I_sat():
    V_1=(V_cut <=-50)
    I_1=I_cut[V_1]
    return np.mean(I_1)
    
def I_sat_plot():
    plt.figure(figsize=(10, 6))
    plt.plot(V, I, 'o', label='IV Data', color='red')  
    plt.xlabel('Voltage Input (Volts)')
    plt.ylabel('Current Output (Amperes)')
    plt.title('Ion Saturation Overlayed on Top of Data')
    plt.axhline(y=I_sat(), color='green', linestyle='-', label=f'Ion Saturation')
    plt.legend()
    plt.grid(True)
    plt.show() 

def asymtotes_plot():
    plt.figure(figsize=(10, 6))
    plt.plot(V, I, 'o', label='IV Data', color='red')  
    plt.xlabel('Voltage Input (Volts)')
    plt.ylabel('Current Output (Amperes)')
    plt.title('Asymtotes')
    plt.axhline(y=I_sat(), color='green', linestyle='-', label=f'Ion Saturation')
    plt.plot(np.linspace(-5, 20, 100)+slopeatV0xint(), slopeatV0*(np.linspace(-5, 20, 100)), label='Slope Estimate', color='green')
    plt.legend()
    plt.grid(True)
    plt.show()

def V_f():
    return slopeatV0xint()

def V_f_plot():
    plt.figure(figsize=(10, 6))
    plt.plot(V, I, 'o', label='IV Data', color='red')  
    plt.xlabel('Voltage Input (Volts)')
    plt.ylabel('Current Output (Amperes)')
    plt.title('Floating Potential')
    plt.axvline(x=V_f(), color='green', linestyle='-', label=f'Floating Potential: {V_f():.2f} V')
    plt.legend()
    plt.grid(True)
    plt.show()

def lnIeplot():
    lnI=np.log(I_cut)
    plt.figure(figsize=(10, 6))
    V1 = V_cut[(V_cut >= V_f()) & (V_cut <= 25)]
    lnI1 = lnI[(V_cut >= V_f()) & (V_cut <= 25)]
    plt.plot(V1, lnI1, 'o', label='IV Data', color='red')  
    plt.xlabel('Voltage Input (Volts)')
    plt.ylabel('Natural Log of Current Output')
    plt.title('Natural Log Plot')
    plt.legend()
    plt.grid(True)
    plt.show()

def lnIeplotfit():
    lnI = np.log(I_cut)
    V1 = V_cut[(V_cut >= V_f()) & (V_cut <= 25)]
    lnI1 = lnI[(V_cut >= V_f()) & (V_cut <= 25)]
    V2 = V_cut[(V_cut >= V_f()) & (V_cut <= 10)]
    lnI2 = lnI[(V_cut >= V_f()) & (V_cut <= 10)]
    slope, intercept, r_value, p_value, std_err = stats.linregress(V2, lnI2)
    print('Linear Fit for Natural Log Plot')
    print(f"Fitted Parameters:\n slope={slope}\n intercept={intercept}\n ")
    plt.figure(figsize=(10, 6))
    plt.plot(V1, lnI1, 'o', label='IV Data', color='red')  
    plt.xlabel('Voltage Input (Volts)')
    plt.ylabel('Natural Log of Current Output (Amperes)')
    plt.title('Natural Log of $I_e$ plot with Slope Fit')
    plt.plot(np.linspace(4, 10, 100), slope * np.linspace(4, 10, 100) + intercept, label='Slope Estimate', color='green')
    plt.legend()
    plt.grid(True)
    plt.show()

def lnIeslope():
    lnI = np.log(I_cut)
    V2 = V_cut[(V_cut >= V_f()) & (V_cut <= 10)]
    lnI2 = lnI[(V_cut >= V_f()) & (V_cut <= 10)]
    slope, intercept, r_value, p_value, std_err = stats.linregress(V2, lnI2)
    return slope

def e_temperature():
    return 1/(lnIeslope())

def e_density(A_p,M): #A_p in m^2 and M in kg for argon M=6.62e-26
    return ((np.abs(I_sat()))/(e*A_p*np.exp(-.5)))*np.sqrt(M/(K_b*e_temperature()))


#InitialPlot()
#Sigmoidcurvefit()
#slopeplot()
#print(f'Slope of Ascent: {slope()} A/V')
#print(f'x intercept of slope of ascent: {slopeatV0xint()}')
#print(f'y intercept of slope of ascent: {slopeatV0yint()}')
#print(f'Ion Saturation: {I_sat()} A')
#print(f'electron saturation {e_sat} A')
#I_sat_plot()
#print(f'floating potential: {V_f()} V')
#V_f_plot()
#lnIeplot()
#print(f'slope of natural log ascent: {lnIeslope()} ln(A)/V')
#lnIeplotfit()
#print(f'Electron Temperature {e_temperature()} eV')
#print(f'Electron Density: {e_density(5e-6,6.62e-26)} electrons/m^3')