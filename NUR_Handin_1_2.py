"""
NUR_Handin1_2
Created on 14-02-2025

@author(s): Sam Beckers
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

#### Q2.A ####

# Load data
data=np.genfromtxt(os.path.join(sys.path[0],"Vandermonde.txt"),comments='#',dtype=np.float64)
x=data[:,0]
y=data[:,1]
xx=np.linspace(x[0],x[-1],1001) #x values to interpolate at

# def crout(Matrix):
#     for i in range(len(Matrix[:,0])):
#         for j in range(len(Matrix[0,:])):
#             #All Beta_0j values are equal to the original matrix values
#             #no need to do anything
#             if i == 0:
#                 continue
#             if i<=j:
#                 Matrix[i][j] = Matrix[i][j]-sum(Matrix[i][k]*Matrix[k][j] for k in range(i))
#             if i>j:
#                 Matrix[i][j] = (1/Matrix[j][j])*(Matrix[i][j]-sum(Matrix[i][k]*Matrix[k][j] for k in range(j)))
#     return Matrix

# Create Vandermonde matrix
V_mat = np.zeros((len(x), len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        V_mat[i,j] = x[i]**j

def crout_improved(A):
    """An improved version of the Crout algorithm for LU decomposition

    Args:
        A (np.ndarray): NxN matrix

    Returns:
        LU (np.ndarray): LU decomposition matrix
    """
    LU = np.copy(A)  # LU = A
    n = len(LU)  # Number of rows/columns
    i_max_arr = np.zeros(n, dtype=np.int64)  # Keep track of what i_max was for each k

    # "Propagate" along row & column
    for k in range(n): 
        for i in range(n):  # Loop over rows i >= k
            if i >= k:
                # Find the row with the largest absolute pivot candidate in column k
                column = LU[:, k]
                i_max = np.argmax(np.abs(column[i:])) + i  # argmax returns index in sliced array, so add back i
                i_max_arr[k] = i_max
        
        # Swap rows if i_max != k
        if i_max_arr[k] != k:
            for j in range(n):  # Loop over columns
                LU[i_max_arr][j] = LU[k, j]
    
        # Find beta values and replace in LU matrix
        for i in range(n):
            if i > k:
                LU[i, k] = LU[i, k] / LU[k, k]  # =1/\beta_{jj}
                for j in range(n):
                    if j > k:
                        LU[i, j] = LU[i, j] - LU[i, k] * LU[k, j]  # \propto \beta
    
    return LU

def solve_LU(LU, b):
    """Solve the system of equations using forward
    and backward substitution with the LU decomposition matrix

    Args:
        LU (np.ndarray): LU decomposition matrix
        b (array) : RHS of the system of equations

    Returns:
        x (array): Solution to the system of equations
    """
    n = len(b) 

    # Forward substitution
    y = np.zeros(n)
    for i in range(n):
            alpha_y_sum = np.sum([LU[i,j]*y[j] for j in range(i)])
            y[i] = b[i] - alpha_y_sum
    
    # Backward substitution
    x = np.zeros(n)
    x[n-1] = y[n-1] / LU[n-1,n-1]
    for i in range(n-1, -1, -1): # Backwards loop
        if j > i: 
            beta_x_sum = np.sum([LU[i,j]*x[j] for j in range(n-1, -1, -1)]) # Another backwards loop
            x[i] = (1/LU[i,i]) * (y[i] - beta_x_sum)

    return x

c_arr = solve_LU(crout_improved(V_mat), y)
print("Values of c:\n",c_arr)

def solve_y(c, x_arr):
    """Solve the polynomial for the given x values. Eq. 2 of Hand-in 1

    Args:
        c_arr (array): Coefficients of the polynomial
        x_arr (array): x values to solve for
    Returns:
        y (array): y values of the polynomial
    """
    return np.array([sum([c[j]*i**j for j in range(len(c_arr))]) for i in x_arr])

y_LUD_interp = solve_y(c_arr, xx) # Interpolated values
y_LUD = solve_y(c_arr, x) # Values at sample points
abs_diff = np.abs(y_LUD - y)

plt.figure(dpi=200)
plt.plot(xx, y_LUD_interp, c='red', label='Interpolated polynomial (LU decomposition)', zorder=0)
plt.scatter(x, y, c='k', label='Samples', zorder=1)
plt.xlim(np.min(x)-2, np.max(x)+2)
plt.ylim(np.min(y)-100, np.max(y)+100)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

plt.figure(dpi=200)
plt.plot(x, abs_diff, '^', c='k', label='$|y(x) - y_i|$')
plt.xlabel('x')
plt.ylabel('Absolute difference')
plt.yscale('log')
plt.legend()
plt.show()
#### Q2.B ####

def bisection_v2(x, sample_points, M):
    """Bisection algorithm to find nearest sample point(s) to x

    Args:
        x (float): the value to search for
        sample_points (array): the grid of sample points to search in
        M (int): number of sample points around the edge points to consider 

    Returns:
        j_low (int): the index of the lowest sample point in the interpolation
    """
    # If x is smaller than the first sample point, j_low must be zero
    if x < sample_points[0]: 
        return 0 
    # If x is larger than last sample point, j_low must be at the edge
    if x > sample_points[-1]:
        return (len(sample_points) - 1) - M # -M to ensure there are enough j_low+1, ..., j_low+(M-1) points at the edge
    if x == sample_points[-1]: # If x is the last sample point, return the last index
        return len(sample_points) - M
    
    # Find edge points
    begin, end = 0, len(sample_points) - 1
    while end - begin > 1: # Whilst edge points are not adjacent
        mid = int((begin + end) * 0.5) # Find the middle point
        if x <= sample_points[mid]: # If x is in the left half
            end = mid # Shift the end point to the middle
        else: # If x is in the right half
            begin = mid # Shift the begin point to the middle

    # If the begin/end point is too close to the left/right edge for our order 
    if begin - M < 0: 
        return 0
    elif end + M > len(sample_points) - 1: 
        return (len(sample_points) - 1) - M
    
    return begin

def neville(x, x_grid, y_grid, M):
    """Neville's algorithm for polynomial interpolation

    Args:
        x (float): the value to interpolate for
        x_grid (array): the grid of x values
        y_grid (array): the grid of y values
        M (int): the order of interpolation
    
    Returns:
        P[0]: the interpolated value
    """
    j_low = bisection_v2(x, x_grid, M) 
    P = np.copy(y_grid[j_low:j_low + M]) # Set initial P_i values at M tabulated points around x
    x_grid = np.copy(x_grid[j_low:j_low + M])

    for k in range(1, M):
        for i in range(M - k):
            P[i] = ((x_grid[i+k] - x) * P[i] + (x - x_grid[i]) * P[i+1]) \
            / (x_grid[i+k] - x_grid[i]) # =H(x)

    dy = np.abs(P[0] - P[1]) # Error estimate

    return P[0], dy

y_neville = np.array([neville(i, x, y, 20)[0] for i in xx]) # Interpolate at all x values

plt.figure(dpi=200)
plt.plot(x, y, 'o', label='Samples')
plt.plot(xx, y_neville, label='Interpolated polynomial (Neville\'s algorithm)')
plt.xlim(np.min(x)-2, np.max(x)+2)
plt.ylim(np.min(y)-100, np.max(y)+100)
plt.legend()
plt.show()

diff = np.abs(y_LUD_interp - y_neville)
print(np.max(diff))

#### Q2.C ####
