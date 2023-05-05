#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:45:05 2023

@author: yuriyz
(can try with 
 exclusive -C m 
 on a larger machine)
"""

from gurobipy import *
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.stats import pareto

# some Gurobi run-time params that seem to give better runs
LPmethod     = 2
LPcrossover  = 0
LPbarconvtol = 1e-8



"""
The value function take in mandatory arguments
- t, alpha, X (n-by-m sample values of n r.v. array), cdfLB, cdfUB (m**n arrays)
and optionally
- model (Gurobi object) and Z (n-tensor)
If the optional argumens are 'None', we construct and solve the value (LP) model,
and return the value (val), its derivative w.r.t. t (dVal), as well as
the respective model object and (instantiated) Z tensor.
If the optional arguments are supplied, that is, MODEL = [model,pmf,Z] is not 'None', 
we modify the objective only and resolve.

"""
def val_t(t,alpha,X,cdfLB,cdfUB,MODEL=None):
        
    # get the base dimensions
    [n,m] = X.shape
    varIdcs = []
    oneDIdx = [*range(m)]
    varIdcs = list(itertools.product(oneDIdx, repeat=n))   
    
    if MODEL is None:
        # preprocess for Z (and C)
        print(' ==> preprocessing for Z...')
        
        Z = np.zeros([m] * n)
        
        for j in range(n):
            idx        = [1] * n
            idx[n-1-j] = m
            replica    = X[j,:]
            replica    = replica.reshape(idx)
            idx        = [m] * n
            idx[n-1-j] = 1
            Z = Z + np.tile(replica,idx)
        
        # prep the model     
        model = Model('CVaR')
        
        # define pmf and cdf vars and objective
        print(' ==> forming vars...')
        pmf = model.addVars(varIdcs, ub=1, name='pmf')
        cdf = model.addVars(varIdcs, lb=cdfLB, ub=cdfUB, name='cdf')   
        
        # connect pmf and cdf via constraints
        print(' ==> forming constrs...')
        counter = 1
        for idx in itertools.product(oneDIdx, repeat=n):
            if (counter % 10000 == 0):
                print('- adding', counter, 'out of', m**n, 'cdf contsraints: current idx =', str(idx))
            LHS    = pmf[idx] - cdf[idx]
            idx    = list(idx)
            for k in range(1,2**n):
                # k encodes the combination of index offsets for each cdf term
                sign   = -1
                idxCDF = idx.copy()
                for j in range(n):
                    # check the respctive bits for offset of the j-th index
                    offset = 2**j & k
                    if offset:
                        idxCDF[j] += -1
                        sign      *= -1
                if all(i > -1 for i in idxCDF):
                    LHS += sign * cdf[tuple(idxCDF)]
            c = model.addConstr( LHS == 0, name='cdf'+str(idx).replace(' ',''))
            counter += 1
        
        print('- adding marginals')
        for j in range(n):
            idx = [m-1] * n
            idx[j] = 0
            c = model.addConstr( cdf[tuple(idx)] == 1/m, name=str(j)+'-marginal[0]')
            for i in range(1,m):
                idx       = [m-1] * n
                idx[j]    = i
                idxSUB    = [m-1] * n
                idxSUB[j] = i - 1
                # instead of (net) marginals below
                # c = model.addConstr( cdf[tuple(idx)] - cdf[tuple(idxSUB)] == 1/m, name=str(j)+'-marginal'+str([i]))
                # add cumulative marginall constraints :)
                if (i < m-1) or (j == n-1):
                    c = model.addConstr( cdf[tuple(idx)] == (i+1)/m, name=str(j)+'-cum.marginal'+str([i]))
    else:
        # extract the model and Z from the input
        model = MODEL[0]
        pmf   = MODEL[1]
        Z     = MODEL[2]
        
    # set the objective
    print(' ==> forming objective...')
    C = np.maximum(Z - t, 0.0)
    model.setObjective( quicksum(C[idx] * pmf[idx] for idx in varIdcs) )    
    
    #================debug===============
    # may test a trivial pmf init for debug (by fixing)
    #for idx in itertools.product(oneDIdx, repeat=n):
    #    pmf[idx].lb = pmf[idx].ub = 0
    #for i in range(m):
    #    pmf[tuple([i] * n)].lb = pmf[tuple([i] * n)].ub = 1/m
    #model.update()
    #model.write('cvar.lp')
    
    
    # solve
    #model.reset()
    model.params.logfile    = 'gurobi.log'
    model.params.method     = LPmethod
    model.params.barconvtol = LPbarconvtol
    model.params.crossover  = LPcrossover
    model.optimize()
    if model.status == GRB.OPTIMAL:
        val  = t + model.objval/(1-alpha)
        dC   = 1 * (Z >= t)
        aux  = quicksum( dC[idx] * pmf[idx] for idx in varIdcs )   
        dVal = 1 - aux.getValue()/(1-alpha)
    else:
        val  = None
        dVal = None
    
    
    # form the output
    MODEL = [model, pmf, Z]
    return [val, dVal, MODEL]




# Test, prep the data first
"""
Note input data arrays must be consistent with p.m.f./c.d.f. var indexing,
e.g., n=3 relies on

[(0, 0, 0),
 (0, 0, 1),
 (0, 0, 2),
...
 (0, 0, m-1),
 (0, 1, 0),
 (0, 1, 1),
 (0, 1, 2),
...
 (0, 1, m-1),
...
 (m-1, 0, 0),
...
 (m-1, m-1, m-1)]

and to test (e.g., n=3, m=5) can use 

IDCS = []
for j in varIdcs:
    IDCS.append(100*j[0]+10*j[1]+j[2])
res = np.reshape(IDCS, [5,5,5])


where the index '(x,y,z)' are packed into digits 'xyz',
giving res as

array([[[  0,   1,   2,   3,   4],
        [ 10,  11,  12,  13,  14],
        [ 20,  21,  22,  23,  24],
        [ 30,  31,  32,  33,  34],
        [ 40,  41,  42,  43,  44]],
...
       [[400, 401, 402, 403, 404],
        [410, 411, 412, 413, 414],
        [420, 421, 422, 423, 424],
        [430, 431, 432, 433, 434],
        [440, 441, 442, 443, 444]]])

and

res[0][0][0]
Out[267]: 0

res[4][0][0]
Out[268]: 400

res[4][0][3] or res[4,0,3]:)
Out[269]: 403


To this extent, we assume we are provided with 
- t-value
- alpha
- lower and upper bounds on c.d.f.
- the underlying r.v. X (ordered) values as n-by-m numpy array (rows store Xi)

where and can test forming of Z (C) with 

X = np.array([[0,1,2,3,4],[0,10,20,30,40],[0,100,200,300,400]])

that gives Z to be look-alike of 'res' (~IDCS)
"""

    


try: userSupplied
except NameError: userSupplied = None
# note if userSupplied is defined, the X, cdfLB, cdfU data will be loaded from
# inX.npz, inCdfLB, inCdfU (from the current directory) while if undefined,
# a compressed dump of X, cdfLB, cdfU will be created (inX.npz)

# set target runtime parameters
# we build the cdf bounds in accordance to genMethod, where
# genMethod = 0 or 1 are basic tests, and self-explanatory,and
# genMethod = 2 0r 3 give log-normal data as in the manuscript
# genMethod = 4 or 5 is Pareto (latest try)
genMethod = 5
alpha     = .8
n         = 3
m         = 4
t         = .5 * n
# to plot the value function over a range of tuse N sample points
N       = 10
# flex factor at t_min and t_max end points
eps     = 1e-5
# epigraph scheme termination epsilon
# since average total loss is on the order of 1e6, this can be relatively high:)
epsStop = 1e1




# set the dimensions (and init X, if none was provided)
if (userSupplied is None):
    print(' ==> generating random input data...')
    X = np.random.rand(n,m)
    X.sort()
    if genMethod == 0:
        # trivial bounds
        cdfLB = [0] * (m**n)
        cdfUB = [1] * (m**n)
    elif genMethod == 1:
        pmfRef = np.zeros([m] * n)
        # randomized (but still simple) bounds
        # can be based off, for example, a uniform
        # pmfRef = np.ones([m] * n) / m**n      
        # or a 'diagonal' pmf
        for i in range(m):
            pmfRef[tuple([i] * n)] = 1/m
        # next we construct and perturb the reference to construct bounds
        cdfRef = np.zeros([m] * n)
        counter = 1
        for idx in itertools.product([*range(m)], repeat=n):
            if (counter % 10000 == 0):
                print('- computing', counter, 'out of', m**n, 'cdf reference values: current idx =', str(idx))
            LHS    = pmfRef[idx]
            idx    = list(idx)
            for k in range(1,2**n):
                # k encodes the combination of index offsets for each cdf term
                sign   = -1
                idxCDF = idx.copy()
                for j in range(n):
                    # check the respctive bits for offset of the j-th index
                    offset = 2**j & k
                    if offset:
                        idxCDF[j] += -1
                        sign      *= -1
                if all(i > -1 for i in idxCDF):
                    LHS += sign * cdfRef[tuple(idxCDF)]
            cdfRef[tuple(idx)] = LHS
            counter += 1  
        cdfLB = np.maximum(0, cdfRef - 1/m * np.random.rand(m**n).reshape([m] * n))
        cdfUB = np.minimum(1, cdfRef + 1/m * np.random.rand(m**n).reshape([m] * n))
    elif genMethod == 2 or genMethod == 3:
        assert(n <= 3)
        # note we reverse the order of X(i) as compared to the manuscript 
        # to have a cleaner cdfUB build (~ F3 * max(F1,F2))
        if genMethod == 3:
            X = np.tile(.5/m + np.arange(0,1,1/m),[n,1])
        S = [3.25, 3.5, 3]
        S = S[0:n]
        X = lognorm.ppf(X, s = np.tile(np.reshape(S,[n,1]),[1,m]), scale = np.exp(10 * np.ones([n,m])))
        Fi    = 1/m + np.arange(0,1,1/m)
        cdfLB = np.ones([m] * n)
        cdfUB = np.zeros([m] * n)
        for j in range(n):
            idx        = [1] * n
            idx[n-1-j] = m
            replica    = Fi.reshape(idx)
            idx        = [m] * n
            idx[n-1-j] = 1
            layer      = np.tile(replica,idx)
            cdfLB     *= layer
            if j == 0:
                cdfUB = layer
            elif j == 1:
                cdfUB = np.minimum(cdfUB, layer)
            else:
                cdfUB *= layer
    elif genMethod == 4 or genMethod == 5:
        assert(n <= 3)
        # note we reverse the order of X(i) as compared to the manuscript 
        # to have a cleaner cdfUB build (~ F3 * max(F1,F2))
        # and use TX, FL, NY ordering for the parameters
        if genMethod == 5:
            X = np.tile(.5/m + np.arange(0,1,1/m),[n,1])
        b   = [2.7, 2.1, 5]
        loc = [-7.3610E+06, -1.1077E+07, -7.9200E+06]
        s   = list(-np.array(loc))
        b   = b[0:n]
        loc = loc[0:n]
        s   = s[0:n]
        X = pareto.ppf(X, np.tile(np.reshape(b,[n,1]),[1,m]), np.tile(np.reshape(loc,[n,1]),[1,m]), np.tile(np.reshape(s,[n,1]),[1,m]),)
        Fi    = 1/m + np.arange(0,1,1/m)
        cdfLB = np.ones([m] * n)
        cdfUB = np.zeros([m] * n)
        for j in range(n):
            idx        = [1] * n
            idx[n-1-j] = m
            replica    = Fi.reshape(idx)
            idx        = [m] * n
            idx[n-1-j] = 1
            layer      = np.tile(replica,idx)
            cdfLB     *= layer
            if j == 0:
                cdfUB = layer
            elif j == 1:
                cdfUB = np.minimum(cdfUB, layer)
            else:
                cdfUB *= layer
    # save the X, cdfLB, cdfUB data to a disk
    np.savez_compressed('inX.npz', X)
    np.savez_compressed('inCdfLB.npz', cdfLB)
    np.savez_compressed('inCdfUB.npz', cdfUB)
                
else:
    # load X, cdfLB, cdfUB from a disk
    X     = np.load('inX.npz')
    cdfLB = np.load('inCdfLB.npz')
    cdfUB = np.load('inCdfUB.npz')
    [n,m] = X.shape


# optionally stop here to check if means (of Xi) make sense, e.g., 
#np.mean(X, axis=1)
#sys.exit(0)


tStart = np.sum(X,axis=0)[0]
tEnd   = np.sum(X,axis=0)[m-1]
valOpt = np.inf
MODEL  = None

timeElapsed = -time.time()
if (0):
    # rough out the graph of the value function
    tVals = np.arange(tStart, (1+eps)*tEnd, (tEnd - tStart)/(N-1))
    vals  = []
    for t in tVals:
        [val, dVal, MODEL] = val_t(t, alpha, X, cdfLB, cdfUB, MODEL)
        # may want to switch to primal simplex for re-solve
        MODEL[0].params.method = 0
        # append the values to the list for later plotting
        vals.append(val)
        # rough out the optimum:)
        valOpt = min(valOpt, val)
        # add tangent to the plt
        t1 = t
        t2 = tEnd
        v1 = val
        v2 = val + dVal*(t2-t1)
        plt.plot([t1,t2], [v1,v2], color='r')
    # get the corresponding index and show the plot
    idx = np.where(vals == valOpt)
    tOpt = tVals[idx][0]
    plt.plot(tVals, vals)
    plt.show()
else:
    # try epigraph scheme
    # set the end points of the search interval (slightly inflated)
    tL = (1-eps)*tStart
    tR = (1+eps)*tEnd
    # get the respective values and derivatives
    [fL, dfL, MODEL] = val_t(tL, alpha, X, cdfLB, cdfUB, MODEL)
    # may want to switch to primal simplex for re-solve
    MODEL[0].params.method = 0
    [fR, dfR, MODEL] = val_t(tR, alpha, X, cdfLB, cdfUB, MODEL)
    # set the bounds on CVaR
    fUpper = min(fL,fR)
    tMid   = (fL-fR+dfR*tR-dfL*tL)/(dfR-dfL)
    fLower = fL + dfL*(tMid-tL)
    # enter the main bi-section loop
    counter = 1
    while fUpper-fLower > epsStop:
        print('- bisection iterate', counter, 'upper', fUpper, 'and lower', fLower)
        counter += 1
        # sample  a mid point
        t = (tL + tR)/2
        [f, df, MODEL] = val_t(t, alpha, X, cdfLB, cdfUB, MODEL)
        if df < 0:
            tL  = t
            fL  = f
            dfL = df
        else:
            tR  = t
            fR  = f
            dfR = df
        # recompute upper and lower bounds 
        fUpper = min(fL,fR)
        tMid   = (fL-fR+dfR*tR-dfL*tL)/(dfR-dfL)
        fLower = fL + dfL*(tMid-tL)
    # may as well use a mid-point as a surrogate
    valOpt = (fLower + fUpper)/2
    tOpt   = (tL + tR)/2
    
timeElapsed += time.time()
print('Best val found:',valOpt,'at t:',tOpt,'in',timeElapsed,'seconds')
    

    