import numpy as np
from scipy.optimize import nnls

class Result(object):
    '''Result object for storing input and output data for omp.  When called from 
    `omp`, runtime parameters are passed as keyword arguments and stored in the 
    `params` dictionary.

    Attributes:
        X:  Predictor array after (optional) standardization.
        y:  Response array after (optional) standarization.
        ypred:  Predicted response.
        residual:  Residual vector.
        coef:  Solution coefficients.
        active:  Indices of the active (non-zero) coefficient set.
        err:  Relative error per iteration.
        params:  Dictionary of runtime parameters passed as keyword args.   
    '''
    
    def __init__(self, **kwargs):
        
        # to be computed
        self.X = None
        self.y = None
        self.ypred = None
        self.residual = None
        self.coef = None
        self.active = None
        self.err = None
        
        # runtime parameters
        self.params = kwargs
            
    def update(self, coef, active, err, residual, ypred):
        '''Update the solution attributes.
        '''
        self.coef = coef
        self.active = active
        self.err = err
        self.residual = residual
        self.ypred = ypred

def omp(X, y, nonneg=True, ncoef=None, maxit=200, tol=1e-3, ztol=1e-12, verbose=True):
    '''Compute sparse orthogonal matching pursuit solution with unconstrained
    or non-negative coefficients.
    
    Args:
        X: Dictionary array of size n_samples x n_features. 
        y: Reponse array of size n_samples x 1.
        nonneg: Enforce non-negative coefficients.
        ncoef: Max number of coefficients.  Set to n_features/2 by default.
        tol: Convergence tolerance.  If relative error is less than
            tol * ||y||_2, exit.
        ztol: Residual covariance threshold.  If all coefficients are less 
            than ztol * ||y||_2, exit.
        verbose: Boolean, print some info at each iteration.
        
    Returns:
        result:  Result object.  See Result.__doc__
    '''
    
    def norm2(x):
        return np.linalg.norm(x) / np.sqrt(len(x))
    
    # initialize result object
    result = Result(nnoneg=nonneg, ncoef=ncoef, maxit=maxit,
                    tol=tol, ztol=ztol)
    if verbose:
        print(result.params)
    
    # check types, try to make somewhat user friendly
    if type(X) is not np.ndarray:
        X = np.array(X)
    if type(y) is not np.ndarray:
        y = np.array(y)
        
    # check that n_samples match
    if X.shape[0] != len(y):
        print('X and y must have same number of rows (samples)')
        return result
    
    # store arrays in result object    
    result.y = y
    result.X = X
    
    # for rest of call, want y to have ndim=1
    if np.ndim(y) > 1:
        y = np.reshape(y, (len(y),))
        
    # by default set max number of coef to half of total possible
    if ncoef is None:
        ncoef = int(X.shape[1]/2)
    
    # initialize things
    X_transpose = X.T                        # store for repeated use
    #active = np.array([], dtype=int)         # initialize list of active set
    active = []
    coef = np.zeros(X.shape[1], dtype=float) # solution vector
    residual = y                             # residual vector
    ypred = np.zeros(y.shape, dtype=float)
    ynorm = norm2(y)                         # store for computing relative err
    err = np.zeros(maxit, dtype=float)       # relative err vector
    
    # Check if response has zero norm, because then we're done. This can happen
    # in the corner case where the response is constant and you normalize it.
    if ynorm < tol:     # the same as ||residual|| < tol * ||residual||
        print('Norm of the response is less than convergence tolerance.')
        result.update(coef, active, err[0], residual, ypred)
        return result
    
    # convert tolerances to relative
    tol = tol * ynorm       # convergence tolerance
    ztol = ztol * ynorm     # threshold for residual covariance
    
    if verbose:
        print('\nIteration, relative error, number of non-zeros')
   
    # main iteration
    for it in range(maxit):
        
        # compute residual covariance vector and check threshold
        rcov = np.dot(X_transpose, residual)
        if nonneg:
            i = np.argmax(rcov)
            rc = rcov[i]
        else:
            i = np.argmax(np.abs(rcov))
            rc = np.abs(rcov[i])
        if rc < ztol:
            if verbose:
                print('All residual covariances are below threshold.')
            break
        
        # update active set
        if i not in active:
            #active = np.concatenate([active, [i]], axis=1)
            active.append(i)
            
        # solve for new coefficients on active set
        if nonneg:
            coefi, _ = nnls(X[:, active], y)
        else:
            coefi, _, _, _ = np.linalg.lstsq(X[:, active], y)
        coef[active] = coefi   # update solution
        
        # update residual vector and error
        residual = y - np.dot(X[:,active], coefi)
        ypred = y - residual
        err[it] = norm2(residual) / ynorm  
        
        # print status
        if verbose:
            print('{}, {}, {}'.format(it, err[it], len(active)))
            
        # check stopping criteria
        if err[it] < tol:  # converged
            if verbose:
                print('\nConverged.')
            break
        if len(active) >= ncoef:   # hit max coefficients
            if verbose:
                print('\nFound solution with max number of coefficients.')
            break
        if it == maxit-1:  # max iterations
            if verbose:
                print('\nHit max iterations.')
    
    result.update(coef, active, err[:(it+1)], residual, ypred)
    return result

if __name__ == '__main__':
    pass
