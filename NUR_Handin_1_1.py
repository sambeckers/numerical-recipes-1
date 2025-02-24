"""
NUR_Handin1_1
Created on 14-02-2025

@author(s): Sam Beckers
"""
import numpy as np
import itertools
import decimal
import warnings
warnings.filterwarnings("ignore")

def factorial(x):
    """Simple factorial function

    Args:
        x (float): input value

    Returns:
        f: factorial of x
    """
    f = 1
    for i in range(1, round(x)+1):
        f *= i
    return f

def poisson(lam, k):
    """Poisson distribution function

    Args:
        lam (float32): lambda value
        k (float32): k value

    Returns:
        P: P(lamda, k)
    """
    try: 
        P = lam**k * np.exp(-lam) / factorial(k) # Poisson equation
        if P == np.inf or P == 0.0: # Check for overflow
            raise OverflowError
        else: 
            return P
    except OverflowError: # In the case of overflow, use Ramanujan's approximation for log(k!)
        log_P = k * np.log(lam) - lam - k * np.log(k) + k - np.log(k*(1+(4*k*(1+2*k)))) / 6 - np.log(np.pi) / 2
        if np.exp(log_P) == 0.0 or np.exp(log_P) == np.inf:
            return np.exp(decimal.Decimal(log_P)) # if P overflows, we cannot prevent this except with the Decimal module
        elif log_P == np.inf: # If log_P overflows, something is wrong with the approximation
            raise OverflowError
        else:
            return np.exp(log_P)

# Test values (include k=339 to show overflow handling of P)
lam_arr = np.array([1,5,3,2.6,100,101, 1], dtype=np.float32)
k_arr = np.array([0,10,21,40,5,200, 339], dtype=np.float32)

print("Results for the given lambda and k values:")
for lam, k in zip(lam_arr, k_arr):
    print(f"lambda = {lam}, k = {k}, P = {poisson(lam, k)}")

# More test values
lam_extended = np.linspace(1, 101, 1000, dtype=np.float32)
k_extended = np.linspace(0, 200, 1000, dtype=np.float32)

P_extended = []
for lam, k in itertools.product(lam_extended, k_extended): # Create all combinations of lambda and k
    P_extended.append(poisson(lam, k))

count = 0
for p in P_extended:
    if p == 0.0 or p < 0.0:
        count += 1

print(f"\nPercentage of zero or negative values in {len(P_extended)} combinations of lambda and k: {count/len(P_extended)*100}%")


