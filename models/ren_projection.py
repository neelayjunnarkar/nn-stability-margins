import numpy as np
import cvxpy as cp
import time


def ren_project_nonlin(X, Y, N11, N12, N21, N22, Lambda_p, Lambda_c, N12_h, N21_h, DK1_t, DK3_h, DK4_h, \
        AG_t, BG1_t, BG2, CG1, CG2_t, DG3_t, **kwargs):
    return ren_project(X, Y, N11, N12, N21, N22, Lambda_c, N12_h, N21_h, DK1_t, DK3_h, DK4_h, \
        AG_t, BG2, CG1, nonlin = True, Lambda_p = Lambda_p, BG1_t = BG1_t, CG2_t = CG2_t, DG3_t = DG3_t, **kwargs)

def ren_project(X, Y, N11, N12, N21, N22, Lambda_c, N12_h, N21_h, DK1_t, DK3_h, DK4_h, AG_t, BG2, CG1, \
                nonlin = False, Lambda_p = False, BG1_t = None, CG2_t = None, DG3_t = None, \
                eps = 0.01, rnn = False, decay_factor=0.98, ):
    # All inputs in numpy
    # This section will use variable terminology of the REN (so DK3 and DK4 both exist)
    # If rnn is true, DK3 will be set to 0.

    # When plant is linear then AG_t = AG, BG2 = BG, CG1 = CG

    if rnn:
        DK3_h = np.zeros_like(Lambda_c)
    
    originals = [X,   Y,  N11,  N12,  N21,  N22,  Lambda_c,  N12_h,  N21_h,  DK1_t,  DK3_h,  DK4_h]
    if satisfy_lmi(originals, AG_t, BG2, CG1, eps, decay_factor, \
            nonlin = nonlin, Lambda_p = Lambda_p, BG1_t = BG1_t, CG2_t = CG2_t, DG3_t = DG3_t):
        return [None for _ in originals]

    vX = cp.Variable(X.shape, symmetric = True)
    vY = cp.Variable(Y.shape, symmetric = True)
    vN11 = cp.Variable(N11.shape)
    vN12 = cp.Variable(N12.shape)
    vN21 = cp.Variable(N21.shape)
    vN22 = cp.Variable(N22.shape)
    vLambda_c = cp.Variable(Lambda_c.shape, diag = True)
    vN12_h = cp.Variable(N12_h.shape)
    vN21_h = cp.Variable(N21_h.shape)
    vDK1_t = cp.Variable(DK1_t.shape)
    if rnn:
        vDK3_h = np.zeros_like(Lambda_c)
    else:
        vDK3_h = cp.Variable(DK3_h.shape)
    vDK4_h = cp.Variable(DK4_h.shape)

    variables = [vX, vY, vN11, vN12, vN21, vN22, vLambda_c, vN12_h, vN21_h, vDK1_t, vDK3_h, vDK4_h]

    condition = construct_condition(variables, AG_t, BG2, CG1, decay_factor, \
        nonlin = nonlin, Lambda_p = Lambda_p, BG1_t = BG1_t, CG2_t = CG2_t, DG3_t = DG3_t)
    
    constraints = [
        vX - eps*np.eye(vX.shape[0]) >> 0, # X positive definite
        vY - eps*np.eye(vY.shape[0]) >> 0, # Y positive definite
        vLambda_c - eps*np.eye(vLambda_c.shape[0]) >> 0, # Lambda positive definite
        condition - eps*np.eye(condition.shape[0]) >> 0 # LMI condition holds
    ]

    obj = sum([cp.sum_squares(var - vVar) for (var, vVar) in zip(originals, variables)])

    prob = cp.Problem(cp.Minimize(obj), constraints)
    t0 = time.process_time()
    prob.solve(solver = cp.MOSEK)
    tf = time.process_time()
    print('Projection: Objective value: ', prob.value)
    print('Projection: Computed in: ', tf - t0, 'time')

    oX   = vX.value
    oY   = vY.value
    oN11 = vN11.value
    oN12 = vN12.value
    oN21 = vN21.value
    oN22 = vN22.value
    oLambda_c = vLambda_c.value.toarray() # Needs 'toarray' since is a sparse matrix (because diagonal
    oN12_h  = vN12_h.value
    oN21_h  = vN21_h.value
    oDK1_t  = vDK1_t.value
    if rnn:
        oDK3_h = np.zeros_like(Lambda_c)
    else:
        oDK3_h  = vDK3_h.value
    oDK4_h  = vDK4_h.value

    print(f'REN proj: orig 2 DK3: {np.linalg.norm(np.linalg.inv(Lambda_c) @ DK3_h, 2)} to {np.linalg.norm(np.linalg.inv(oLambda_c) @ oDK3_h, 2)}')

    return oX, oY, oN11, oN12, oN21, oN22, oLambda_c, oN12_h, oN21_h, oDK1_t, oDK3_h, oDK4_h

def satisfy_lmi(variables, AG_t, BG2, CG1, eps, decay_factor, \
        nonlin = False, Lambda_p = None, BG1_t = None, CG2_t = None, DG3_t = None):
    
    X,   Y,  N11,  N12,  N21,  N22,  Lambda_c,  N12_h,  N21_h,  DK1_t,  DK3_h,  DK4_h = variables

    # Check that X is symmetric and positive definite
    if not np.allclose(X, X.T):
        print('REN proj: satisfy lmi: X not symmetric')
        return False
    try:
        np.linalg.cholesky(X)
    except:
        print('REN proj: satisfy lmi: X not PD')
        return False

    # Check that Y is positive definite
    if not np.allclose(Y, Y.T):
        print('REN proj: satisfy lmi: Y not symmetric')
        return False
    try:
        np.linalg.cholesky(Y)
    except:
        print('REN proj: satisfy lmi: Y not PD')
        return False

    # Check that the LMI is positive definite
    condition = construct_condition(variables, AG_t, BG2, CG1, decay_factor, stacker = 'numpy', \
        nonlin = nonlin, Lambda_p = Lambda_p, BG1_t = BG1_t, CG2_t = CG2_t, DG3_t = DG3_t)
    if not np.allclose(condition, condition.T):
        print('REN proj: satisfy lmi: condition not symmetric')
        return False
    try:
        np.linalg.cholesky(condition)
    except:
        print('REN proj: satisfy lmi: condition not PD')
        return False

    print('REN proj: satisfy lmi: curr params satisfy lmi')
    return True

def construct_condition(variables, AG_t, BG2, CG1, decay_factor, stacker = 'cvxpy', \
        nonlin = False, Lambda_p = None, BG1_t = None, CG2_t = None, DG3_t = None):
    
    if stacker == 'cvxpy':
        stacker = cp.bmat
    else:
        stacker = np.bmat
    
    X,   Y,  N11,  N12,  N21,  N22,  Lambda_c,  N12_h,  N21_h,  DK1_t,  DK3_h,  DK4_h = variables

    ytpy = stacker([[Y, np.eye(Y.shape[0])], [np.eye(Y.shape[0]), X]])

    if nonlin:
        Lambda = stacker([[Lambda_p, np.zeros((Lambda_p.shape[0], Lambda_c.shape[1]))], \
                          [np.zeros((Lambda_c.shape[0], Lambda_p.shape[1])), Lambda_c]])
    else:
        Lambda = Lambda_c

    block_11 = stacker([[decay_factor**2 * ytpy, np.zeros((ytpy.shape[1], Lambda.shape[0]))], \
        [np.zeros((Lambda.shape[1], ytpy.shape[0])), Lambda]])

    # block_22 = stacker([[ytpy, np.zeros((ytpy.shape[1], Lambda_c.shape[0]))], \
    #     [np.zeros((Lambda_c.shape[1], ytpy.shape[0])), Lambda_c]])

    block_22 = stacker([[ytpy, np.zeros((ytpy.shape[1], Lambda.shape[0]))], \
        [np.zeros((Lambda.shape[1], ytpy.shape[0])), Lambda]])

    ytpay = stacker([[AG_t @ Y + BG2 @ N21, AG_t + BG2 @ N22 @ CG1], \
        [N11, X @ AG_t + N12 @ CG1]])

    if nonlin:
        ytpb = stacker([[BG1_t, BG2 @ DK1_t], \
                        [X @ BG1_t, N12_h]])
    else:
        ytpb = stacker([[BG2 @ DK1_t], \
                        [N12_h]])

    if nonlin:
        lcy = stacker([[Lambda_p @ CG2_t @ Y, Lambda_p @ CG2_t], \
                       [N21_h, DK4_h @ CG1]])
    else:
        lcy = stacker([[N21_h, DK4_h @ CG1]])

    if nonlin:
        ld = stacker([[Lambda_p @ DG3_t, np.zeros((Lambda_p.shape[0], DK3_h.shape[1]))], \
                      [np.zeros((DK3_h.shape[0], DG3_t.shape[1])), DK3_h]])
    else:
        ld = DK3_h

    block_21 = stacker([[ytpay, ytpb], \
                        [lcy,    ld ]])
    
    condition = stacker([[block_11, block_21.T], \
                         [block_21, block_22]])

    return condition

# Uses Disciplined Parameterized Programming for a negligible speed up, but at least the code is cleaner.
class LinProjector:
    def __init__(self, AG, BG, CG, eps, decay_factor, state_size, hidden_size, ob_dim, ac_dim, rnn = False):
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.plant_state_size = AG.shape[0]
        self.eps = eps
        self.decay_factor = decay_factor

        self.rnn = rnn

        self.AG = AG
        self.BG = BG
        self.CG = CG

        self.pX   = cp.Parameter((self.plant_state_size, self.plant_state_size), symmetric = True)
        self.pY   = cp.Parameter((self.plant_state_size, self.plant_state_size), symmetric = True)
        self.pN11 = cp.Parameter((self.plant_state_size, self.plant_state_size))
        self.pN12 = cp.Parameter((self.plant_state_size, self.ob_dim))
        self.pN21 = cp.Parameter((self.ac_dim, self.plant_state_size))
        self.pN22 = cp.Parameter((self.ac_dim, self.ob_dim))
        self.pLambda_c = cp.Parameter((self.hidden_size, self.hidden_size), diag = True)
        self.pN12_h = cp.Parameter((self.plant_state_size, self.hidden_size))
        self.pN21_h = cp.Parameter((self.hidden_size, self.plant_state_size))
        self.pDK1_t = cp.Parameter((self.ac_dim, self.hidden_size))
        if self.rnn:
            self.pDK3_h = np.zeros((self.hidden_size, self.hidden_size))
        else:
            self.pDK3_h = cp.Parameter((self.hidden_size, self.hidden_size))
        self.pDK4_h = cp.Parameter((self.hidden_size, self.ob_dim))

        obj_params = [self.pX, self.pY, self.pN11, self.pN12, self.pN21, self.pN22, 
            self.pLambda_c, self.pN12_h, self.pN21_h, self.pDK1_t, self.pDK3_h, self.pDK4_h]

        self.vX = cp.Variable(self.pX.shape, symmetric = True)
        self.vY = cp.Variable(self.pY.shape, symmetric = True)
        self.vN11 = cp.Variable(self.pN11.shape)
        self.vN12 = cp.Variable(self.pN12.shape)
        self.vN21 = cp.Variable(self.pN21.shape)
        self.vN22 = cp.Variable(self.pN22.shape)
        self.vLambda_c = cp.Variable(self.pLambda_c.shape, diag = True)
        self.vN12_h = cp.Variable(self.pN12_h.shape)
        self.vN21_h = cp.Variable(self.pN21_h.shape)
        self.vDK1_t = cp.Variable(self.pDK1_t.shape)
        if self.rnn:
            self.vDK3_h = self.pDK3_h
        else:
            self.vDK3_h = cp.Variable(self.pDK3_h.shape)
        self.vDK4_h = cp.Variable(self.pDK4_h.shape)

        variables = [self.vX, self.vY, self.vN11, self.vN12, self.vN21, self.vN22, 
            self.vLambda_c, self.vN12_h, self.vN21_h, self.vDK1_t, self.vDK3_h, self.vDK4_h]
        
        condition = construct_condition(variables, self.AG, self.BG, self.CG, self.decay_factor)

        constraints = [
            self.vX - self.eps*np.eye(self.vX.shape[0]) >> 0, # X positive definite
            self.vY - self.eps*np.eye(self.vY.shape[0]) >> 0, # Y positive definite
            self.vLambda_c - self.eps*np.eye(self.vLambda_c.shape[0]) >> 0, # Lambda positive definite
            condition - self.eps*np.eye(condition.shape[0]) >> 0 # LMI condition holds
        ]

        obj = sum([cp.sum_squares(pVar - vVar) for (pVar, vVar) in zip(obj_params, variables)])

        self.prob = cp.Problem(cp.Minimize(obj), constraints)

    def project(self, X, Y, N11, N12, N21, N22, Lambda_c, N12_h, N21_h, DK1_t, DK3_h, DK4_h):
        if self.rnn:
            DK3_h = self.pDK3_h # Zero DK3_h out

        originals = [X,   Y,  N11,  N12,  N21,  N22,  Lambda_c,  N12_h,  N21_h,  DK1_t,  DK3_h,  DK4_h]
        if satisfy_lmi(originals, self.AG, self.BG, self.CG, self.eps, self.decay_factor):
            return [None for _ in originals]

        self.pX.value = X
        self.pY.value = Y
        self.pN11.value = N11
        self.pN12.value = N12
        self.pN21.value = N21
        self.pN22.value = N22
        self.pLambda_c.value = Lambda_c
        self.pN12_h.value = N12_h
        self.pN21_h.value = N21_h
        self.pDK1_t.value = DK1_t
        if not self.rnn:
            self.pDK3_h.value = DK3_h
        self.pDK4_h.value = DK4_h

        t0 = time.process_time()
        self.prob.solve(solver = cp.MOSEK)
        tf = time.process_time()
        print('REN Lin Projection: Objective value: ', self.prob.value)
        print('REN Lin Projection: Computed in: ', tf - t0, 'time')

        oX   = self.vX.value
        oY   = self.vY.value
        oN11 = self.vN11.value
        oN12 = self.vN12.value
        oN21 = self.vN21.value
        oN22 = self.vN22.value
        oLambda_c = self.vLambda_c.value.toarray() # Needs 'toarray' since is a sparse matrix (because diagonal
        oN12_h  = self.vN12_h.value
        oN21_h  = self.vN21_h.value
        oDK1_t  = self.vDK1_t.value
        if self.rnn:
            oDK3_h = self.vDK3_h
        else:
            oDK3_h  = self.vDK3_h.value
        oDK4_h  = self.vDK4_h.value

        return oX, oY, oN11, oN12, oN21, oN22, oLambda_c, oN12_h, oN21_h, oDK1_t, oDK3_h, oDK4_h

# class NonlinProjector:
#     def __init__(self, AG_t, BG1_t, BG2, CG1, CG2_t, DG3_t,
#                 eps, decay_factor, state_size, hidden_size, ob_dim, ac_dim, rnn = False):
#         self.ac_dim = ac_dim
#         self.ob_dim = ob_dim
#         self.state_size = state_size
#         self.hidden_size = hidden_size
#         self.plant_state_size = AG_t.shape[0]
#         self.eps = eps
#         self.decay_factor = decay_factor

#         self.rnn = rnn

#         self.AG_t = AG_t
#         self.BG1_t = BG1_t
#         self.BG2 = BG2
#         self.CG1 = CG1
#         self.CG2_t = CG2_t
#         self.DG3_t = DG3_t

#         self.pX   = cp.Parameter((self.plant_state_size, self.plant_state_size), symmetric = True)
#         self.pY   = cp.Parameter((self.plant_state_size, self.plant_state_size), symmetric = True)
#         self.pN11 = cp.Parameter((self.plant_state_size, self.plant_state_size))
#         self.pN12 = cp.Parameter((self.plant_state_size, self.ob_dim))
#         self.pN21 = cp.Parameter((self.ac_dim, self.plant_state_size))
#         self.pN22 = cp.Parameter((self.ac_dim, self.ob_dim))
#         self.pLambda_p = cp.Parameter((self.plant_state_size, self.plant_state_size), diag = True)
#         self.pLambda_p_CG2_t = cp.Parameter((self.plant_state_size, self.CG2_t.shape[1]))
#         self.pLambda_c = cp.Parameter((self.hidden_size, self.hidden_size), diag = True)
#         self.pN12_h = cp.Parameter((self.plant_state_size, self.hidden_size))
#         self.pN21_h = cp.Parameter((self.hidden_size, self.plant_state_size))
#         self.pDK1_t = cp.Parameter((self.ac_dim, self.hidden_size))
#         if self.rnn:
#             self.pDK3_h = np.zeros((self.hidden_size, self.hidden_size))
#         else:
#             self.pDK3_h = cp.Parameter((self.hidden_size, self.hidden_size))
#         self.pDK4_h = cp.Parameter((self.hidden_size, self.ob_dim))

#         obj_params = [self.pX, self.pY, self.pN11, self.pN12, self.pN21, self.pN22, 
#             self.pLambda_c, self.pN12_h, self.pN21_h, self.pDK1_t, self.pDK3_h, self.pDK4_h]

#         self.vX = cp.Variable(self.pX.shape, symmetric = True)
#         self.vY = cp.Variable(self.pY.shape, symmetric = True)
#         self.vN11 = cp.Variable(self.pN11.shape)
#         self.vN12 = cp.Variable(self.pN12.shape)
#         self.vN21 = cp.Variable(self.pN21.shape)
#         self.vN22 = cp.Variable(self.pN22.shape)
#         self.vLambda_c = cp.Variable(self.pLambda_c.shape, diag = True)
#         self.vN12_h = cp.Variable(self.pN12_h.shape)
#         self.vN21_h = cp.Variable(self.pN21_h.shape)
#         self.vDK1_t = cp.Variable(self.pDK1_t.shape)
#         if self.rnn:
#             self.vDK3_h = np.zeros_like(self.pDK3_h)
#         else:
#             self.vDK3_h = cp.Variable(self.pDK3_h.shape)
#         self.vDK4_h = cp.Variable(self.pDK4_h.shape)

#         variables = [self.vX, self.vY, self.vN11, self.vN12, self.vN21, self.vN22, 
#             self.vLambda_c, self.vN12_h, self.vN21_h, self.vDK1_t, self.vDK3_h, self.vDK4_h]

#         condition = construct_condition(variables, self.AG_t, self.BG2, self.CG1, self.decay_factor, \
#         nonlin = True, Lambda_p = Lambda_p, BG1_t = self.BG1_t, CG2_t = self.CG2_t, DG3_t = self.DG3_t)
        
#         self.pLambda_p.value = np.eye(self.plant_state_size)

#     def construct_parameterized_condition(self, variables, )