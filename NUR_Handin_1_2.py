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

# Create Vandermonde matrix
V_mat = np.zeros((len(x), len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        V_mat[i,j] = x[i]**j

def plot(x, xx, y_1, y_2 = None, color_1 = None, color_2 = None, label_1 = None, label_2 = None, save = True, save_name = None,
        interp=False, abs_diff = False, plot_y2 = False):
    if interp: 
        plt.figure(dpi=200)
        plt.plot(xx, y_1, c=color_1, label=label_1, zorder=0)
        plt.scatter(x, y, c='k', label='Samples', zorder=1)
        plt.xlim(np.min(x)-2, np.max(x)+2)
        plt.ylim(np.min(y)-100, np.max(y)+100)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        if save:
            plt.savefig(save_name)
        plt.show()
    if abs_diff:
        plt.figure(dpi=200)
        plt.plot(x, y_1, '^', c=color_1, label=label_1)
        if plot_y2:
            plt.plot(x, y_2, '^', c=color_2, label=label_2)
        plt.xlabel('x')
        plt.ylabel('Absolute difference')
        plt.yscale('log')
        plt.legend()
        if save:
            plt.savefig(save_name)
        plt.show()

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

def solve_y(c_arr, x_arr):
    """Solve the polynomial for the given x values in LU decomposition. 
    Eq. 2 of Hand-in 1

    Args:
        c_arr (array): Coefficients of the polynomial
        x_arr (array): x values to solve for
    Returns:
        y (array): y values of the polynomial
    """
    return np.array([sum([c_arr[j]*i**j for j in range(len(c_arr))]) for i in x_arr])

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

def interpolate_neville(x_arr, x_grid, y_grid, order):
    return np.array([neville(i, x_grid, y_grid, order)[0] for i in x_arr])

#### Q2.C ####
"""
Ax' = b
delta_b = Ax' - b
but also Adelta_x = delta_b
and delta_x = LU(A, delta_b)
and x' = LU(A, b)
so delta_x = LU(A, b) - LU(A, Ax' - b)
"""
def matrix_mult(A, b):
    """Simple matrix multiplication

    Args:
        A (np.ndarray): NxN matrix
        b (np.ndarray): Nx1 vector

    Returns:
        A*b (np.ndarray): Nx1 vector
    """
    return np.array([np.sum(A[i] * b) for i in range(len(A))])


def solve_LU_iter(A, b, iter_num):
    x = solve_LU(crout_improved(A), b)
    for i in range(iter_num):
        x = x - solve_LU(crout_improved(A), matrix_mult(A, x) - b)
    return x

def main():
    ## Q2.A ##
    c_arr = solve_LU(crout_improved(V_mat), y)
    print("Values of c:\n",c_arr)

    y_LUD_interp = solve_y(c_arr, xx) # Interpolated values
    y_LUD = solve_y(c_arr, x) # Values at sample points
    abs_diff_LUD = np.abs(y_LUD - y)

    plot(x, xx, y_LUD_interp, color_1='r', label_1='Interpolated polynomial (LU decomposition)', interp=True, save_name='2A_LU_Decomposition_polynomial_fit.png')
    plot(x, xx, abs_diff_LUD, color_1='r', label_1='$|y(x) - y_i|$ (LUD)', abs_diff=True, save_name='2A_LU_Decomposition_absolute_difference.png')

    ## Q2.B ##
    y_neville_interp = interpolate_neville(xx, x, y, 20)
    y_neville = interpolate_neville(x, x, y, 20) 
    abs_diff_neville = np.abs(y_neville - y)

    plot(x, xx, y_neville_interp, color_1='magenta', label_1='Interpolated polynomial (Neville\'s algorithm)', interp=True, save_name='2B_Neville_interpolation.png')
    plot(x, xx, abs_diff_neville, abs_diff_LUD, color_1='magenta', color_2 = 'r', label_1='$|y(x) - y_i|$ (Neville)', label_2='$|y(x) - y_i|$ (LUD)', abs_diff=True, plot_y2=True, save_name='2B_AbsoluteDiff_LU_Neville.png')

    ## Q2.C ##
    y_LUD_iter_interp = solve_y(solve_LU_iter(V_mat, y, 10), xx) # Interpolated values
    y_LUD_iter = solve_y(solve_LU_iter(V_mat, y, 10), x) # Values at sample points
    abs_diff_LUD_iter = np.abs(y_LUD_iter - y)

    plot(x, xx, y_LUD_iter_interp, color_1='b', label_1='Interpolated polynomial (LU decomposition, 10 iterations)', interp=True, save_name='2C_LU_Decomposition_iterative_polynomial_fit.png')
    plot(x, xx, abs_diff_LUD_iter, abs_diff_LUD, color_1='b', color_2 = 'r', label_1='$|y(x) - y_i|$ (LUD, 10 iterations)', label_2='$|y(x) - y_i|$ (LUD)', abs_diff=True, plot_y2=True, save_name='2C_AbsoluteDiff_LU_iterative.png')

if __name__ == "__main__":
    main()