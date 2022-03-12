import numpy as np
import cvxpy as cp
import time
import scipy


def rnn_project(AK_t, BK1_t, BK2_t, CK1_t, DK1_t, DK2_t, CK2_t, DK4_t, Q1_bar, Q2_bar,
            AG, BG, CG,
            eps, decay_factor):

    if Q1_bar is None:
        Q1_bar = init_q1(AG, BG, CG, AK_t.shape[0], rho = decay_factor, eps = eps)
    
    P_bar = np.linalg.inv(Q1_bar)
    Lambda_bar = np.linalg.inv(Q2_bar)

    originals = [AK_t, BK1_t, BK2_t, CK1_t, DK1_t, DK2_t, CK2_t, DK4_t, Q1_bar, Q2_bar]

    vAK_t  = cp.Variable(AK_t.shape)
    vBK1_t = cp.Variable(BK1_t.shape)
    vBK2_t = cp.Variable(BK2_t.shape)
    vCK1_t = cp.Variable(CK1_t.shape)
    vDK1_t = cp.Variable(DK1_t.shape)
    vDK2_t = cp.Variable(DK2_t.shape)
    vCK2_t = cp.Variable(CK2_t.shape)
    vDK4_t = cp.Variable(DK4_t.shape)
    vQ1 = cp.Variable(P_bar.shape, symmetric = True)
    vQ2 = cp.Variable(Lambda_bar.shape, diag = True)

    variables = [vAK_t, vBK1_t, vBK2_t, vCK1_t, vDK1_t, vDK2_t, vCK2_t, vDK4_t, vQ1, vQ2]

    condition = construct_condition(variables, P_bar, Lambda_bar, AG, BG, CG, decay_factor)

    constraints = [
        vQ1 - eps * np.eye(vQ1.shape[0]) >> 0,
        cp.diag(vQ2) >= eps,
        condition - eps * np.eye(condition.shape[0]) >> 0
    ]

    obj = sum([cp.sum_squares(var - vVar) for (var, vVar) in zip(originals, variables)])

    prob = cp.Problem(cp.Minimize(obj), constraints)

    t0 = time.process_time()
    try:
        prob.solve(solver = cp.MOSEK)
        if vAK_t.value is None:
            raise "Error"
    except Exception as e:
        print('error: ', str(e))
        print('solving with SCS')
        prob.solve(solver = cp.SCS)
    tf = time.process_time()

    print('Projection: Objective value: ', prob.value)
    print('Projection: Computed in: ', tf - t0, 'time')

    oAK_t  = vAK_t.value
    oBK1_t = vBK1_t.value
    oBK2_t = vBK2_t.value
    oCK1_t = vCK1_t.value
    oDK1_t = vDK1_t.value
    oDK2_t = vDK2_t.value
    oCK2_t = vCK2_t.value
    oDK4_t = vDK4_t.value
    oQ1_bar = vQ1.value
    oQ2_bar = vQ2.value.toarray()

    return oAK_t, oBK1_t, oBK2_t, oCK1_t, oDK1_t, oDK2_t, oCK2_t, oDK4_t, oQ1_bar, oQ2_bar

def construct_condition(variables, P_bar, Lambda_bar, AG, BG, CG, decay_factor):
    AK_t, BK1_t, BK2_t, CK1_t, DK1_t, DK2_t, CK2_t, DK4_t, Q1, Q2 = variables

    A = cp.bmat([[AG + BG @ DK2_t @ CG, BG @ CK1_t],
                 [BK2_t @ CG,           AK_t]])

    B = cp.bmat([[BG @ DK1_t],
                 [BK1_t]])

    C = cp.bmat([[DK4_t @ CG, CK2_t]])

    D = np.zeros((DK4_t.shape[0], DK4_t.shape[0]))

    block11 = cp.bmat([[decay_factor**2 * (2*P_bar - P_bar.T @ Q1 @ P_bar), np.zeros((P_bar.shape[0], Lambda_bar.shape[1]))],
                       [np.zeros((Lambda_bar.shape[0], P_bar.shape[1])),    2*Lambda_bar - Lambda_bar.T @ Q2 @ Lambda_bar]])

    block21 = cp.bmat([[A, B],
                       [C, D]])

    block22 = cp.bmat([[Q1, np.zeros((Q1.shape[0],            Q2.shape[1]))],
                       [np.zeros((Q2.shape[0], Q1.shape[0])), Q2]])

    condition = cp.bmat([[block11, block21.T],
                         [block21, block22]])

    return condition

def init_q1(AG, BG, CG, xi_dim, rho = 0.98, eps = 1e-5):
    x_dim = AG.shape[0]
    if xi_dim is None:
        xi_dim = x_dim
    obs_dim = CG.shape[0]
    ac_dim = BG.shape[1]

    vX  = cp.Variable((x_dim, x_dim), symmetric = True)
    vY  = cp.Variable((x_dim, x_dim), symmetric = True)
    vK  = cp.Variable((x_dim, x_dim))
    vL  = cp.Variable((x_dim, obs_dim))
    vM  = cp.Variable((ac_dim, x_dim))
    vN  = cp.Variable((ac_dim, obs_dim))

    yPAy = cp.bmat([
        [AG @ vY + BG @ vM, AG + BG @ vN @ CG],
        [vK,                vX @ AG + vL @ CG]
    ])

    yPy = cp.bmat([
        [vY,                np.eye(x_dim)],
        [np.eye(x_dim), vX]
    ])

    LMI1 = cp.bmat([
            [rho**2 * yPy,  yPAy.T],
            [yPAy,          yPy]
    ])

    LMI2 = yPy

    obj = 0
    # constraints = [LMI1 - eps * np.eye(LMI1.shape[0]) >> 0, LMI2 - eps * np.eye(LMI2.shape[0]) >> 0]
    constraints = [LMI1 >> 0, LMI2 >> 0]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver = cp.MOSEK)

    X = vX.value
    Y = vY.value

    # if xi_dim == x_dim:
        # Hand picked U and V
    U = X
    V = np.linalg.inv(X) - Y
    P = np.linalg.inv(np.block([[Y, V],[np.eye(x_dim), np.zeros((x_dim, x_dim))]])) @ \
        np.block([[np.eye(x_dim), np.zeros((x_dim, x_dim))],[X, U]])
    Q1 = np.linalg.inv(P)
    Q1 = scipy.linalg.block_diag(Q1, np.eye(xi_dim - x_dim))
    # else:
    #     # SVD based method to determine P
    #     u, s, vh = np.linalg.svd(np.eye(x_dim) - Y @ X, full_matrices=True)
    #     V = np.block([[u @ np.diag(s), np.zeros((x_dim, xi_dim - x_dim))]])
    #     U = np.block([[vh.T, np.zeros((x_dim, xi_dim - x_dim))]])
    #     xhat = np.linalg.lstsq(X @ V, U @ (V.T @ U - np.eye(xi_dim)))[0]
    #     # xhat = np.linalg.lstsq(V, -Y @ U)[0]
    #     P = np.block([[X, U],
    #                 [U.T, xhat]])

    # Q1init = np.linalg.inv(P)

    return Q1