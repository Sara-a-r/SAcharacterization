import numpy as np
from scipy.optimize import minimize, root, fsolve, newton_krylov, dual_annealing

#-----------------------------A matrix---------------------------#
def matrix_A(K, M1, M2, M3, M4, M5, M6):
    A = np.array([[-(K[0] + K[1]) / M1, K[1] / M1, 0, 0, 0, 0],
                  [K[1] / M2, -(K[1] + K[2]) / M2, K[2] / M2, 0, 0, 0],
                  [0, K[2] / M3, -(K[2] + K[3]) / M3, K[3] / M3, 0, 0],
                  [0, 0, K[3] / M4, -(K[3] + K[4]) / M4, K[4] / M4, 0],
                  [0, 0, 0, K[4] / M5, -(K[4] + K[5]) / M5, K[5] / M5],
                  [0, 0, 0, 0, K[5] / M6, -K[5] / M6]])
    return A

#-----------------------------det equation-----------------------#
# the function to minimize is det(A+w**2) (il must be equal to 0)
def determinant_equation(K, M, w_values):
    determinant_values = []             # list of the equation that must be equal to zero
                                        # one for each value of w (five in this case)#
    for w in w_values:
        A = matrix_A(K, *M)
        determinant_values.append(np.linalg.det(A + w**2 * np.eye(len(A))))
    return determinant_values

if __name__ == '__main__':
    # define masses of the system
    M = [173, 165, 140, 118, 315, 125]  # M1, M2, M3, M4, M7, Mpayload

    # define normal frequencies of the system
    f_normal = np.array([0.107, 0.360, 0.483, 0.701, 1.131, 1.368])
    w_normal = 2 * np.pi * f_normal

    # define initial values for K
    initial_K = np.array([1, 1, 1, 1, 1, 1])

    # set the bounds of the variables (K must be positive)
    bounds = [(0, 20000)] * 6

    # numerical resolution
    #result = minimize(lambda K: np.linalg.norm(determinant_equation(K, M, w_normal)),
    #                  initial_K, method='SLSQP')#,bounds=bounds, options={'maxiter':1000})

    result = dual_annealing(lambda K: np.linalg.norm(determinant_equation(K, M, w_normal)), bounds, seed=1)

    solutions = result.x

    print('Did the routine ended well?',result.success)
    if not result.success:
        print(result.message)

    print("Solutions for K1, K2, K3, K4, K5:")
    print(solutions)
    print("The minimum value found is:", np.linalg.norm(determinant_equation(solutions, M, w_normal)))