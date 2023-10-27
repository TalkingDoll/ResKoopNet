# Assuming necessary imports
import numpy as np

def find_lexicographic(lookfor, I, nocheck=None):
    """
    Finds specific rows of a matrix that is sorted lexicographically.
    """

    if nocheck is None and not np.array_equal(I, np.sort(I, axis=0)):
        raise ValueError("I is not lexicographically sorted")

    if nocheck is not None and nocheck != 'nocheck':
        raise ValueError("unknown 3rd input")

    nb_idx = I.shape[0]

    # Binary search
    idx = nb_idx // 2
    jj = I[idx, :]

    found = np.array_equal(jj, lookfor)

    iter = 1
    itermax = int(np.ceil(np.log2(nb_idx)))

    while not found and iter <= itermax:
        if np.lexsort([jj, lookfor])[0] == 0:  # Equivalent to islexico in MATLAB
            if idx < nb_idx - 1:
                idx = min(idx + nb_idx // 2**iter, nb_idx - 1)
                jj = I[idx, :]
                found = np.array_equal(lookfor, jj)
                iter += 1
            else:
                break
        else:
            if idx > 0:
                idx = max(idx - nb_idx // 2**iter, 0)
                jj = I[idx, :]
                found = np.array_equal(lookfor, jj)
                iter += 1
            else:
                break

    pos = None
    if found:
        pos = idx

    return found, pos, iter

# Note: The function islexico is not provided in the MATLAB code.
# The np.lexsort function is used as an equivalent in Python.



MATLAB_SPARSE_KIT_VERBOSE = 1

def multiidx_gen(N, rule, w, base=0, multiidx=None, MULTI_IDX=None):
    """
    Calculates all multi indexes M_I of length N with elements such that rule(M_I) <= w.
    M_I's are stored as rows of the matrix MULTI_IDX.
    Indices will start from base (either 0 or 1).
    """

    if multiidx is None:
        multiidx = []

    if MULTI_IDX is None:
        MULTI_IDX = []

    if len(multiidx) != N:
        # Recursive step: generates all possible leaves from the current node (i.e. all multiindexes with length le+1 starting from
        # the current multi_idx, which is of length le that are feasible w.r.t. rule)
        i = base
        while rule(multiidx + [i]) <= w:
            # If [multiidx, i] is feasible further explore the branch of the tree that comes out from it.
            MULTI_IDX = multiidx_gen(N, rule, w, base, multiidx + [i], MULTI_IDX)
            i += 1
    else:
        # Base step: if the length of the current multi-index is N then I store it in MULTI_IDX
        # (the check for feasibility was performed in the previous call)
        MULTI_IDX.append(multiidx)

    return MULTI_IDX

# Note: The function 'rule' is not provided in the MATLAB code.
# You'll need to define it in Python as well if it's used in your application.


def smolyak_grid(N, w, knots, lev2knots, idxset=None, arg6=None, weights_coeff=None):
    global MATLAB_SPARSE_KIT_VERBOSE

    if idxset is None:
        idxset = lambda i: sum(i) - 1

    if 'arg6' in locals():
        if callable(arg6):
            map = arg6
        elif issmolyak(arg6) or arg6 is None:
            S2 = arg6
        else:
            raise ValueError('SparseGKit:WrongInput', 'unknown type for 6th input')

    if callable(knots):
        fknots = knots
        knots = [fknots for _ in range(N)]

    if callable(lev2knots):
        f_lev2knots = lev2knots
        lev2knots = [f_lev2knots for _ in range(N)]

    if w == 0:
        i = np.ones(N, dtype=int)
        m = apply_lev2knots(i, lev2knots, N)
        S = [tensor_grid(N, m, knots)]
        S[0]['coeff'] = 1
        S[0]['idx'] = i
        C = i
    else:
        # ... (rest of the code, similar conversion)
        # Note: You'll need to convert other functions like multiidx_gen, tensor_grid, etc.
        C = np.array(multiidx_gen(N, idxset, w, 1))

        # Build the set of S2 if provided
        if S2 is not None:
            nb_idx_C2 = len(S2)
            C2 = np.array([s2['idx'] for s2 in S2]).T
        else:
            C2 = None

        nn = C.shape[0]
        coeff = np.ones(nn)

        _, bookmarks = np.unique(C[:, 0], return_index=True)
        bk = np.hstack((bookmarks[2:] - 1, [nn, nn]))

        for i in range(nn):
            cc = C[i]
            range_ = bk[cc[0]-1]
            for j in range(i + 1, range_):
                d = C[j] - cc
                if d.max() <= 1 and d.min() >= 0:
                    coeff[i] += (-1) ** d.sum()

        nb_grids = np.sum(coeff != 0)
        S = [{'knots': None, 'weights': None, 'size': None, 'knots_per_dim': None, 'm': None} for _ in range(nb_grids)]
        coeff_condensed = np.zeros(nb_grids)
        ss = 0

        if C2 is not None:
            for j in range(nn):
                if coeff[j] != 0:
                    i = C[j]
                    found, pos = find_lexicographic(i, C2)
                    if found:
                        for key in S[ss].keys():
                            S[ss][key] = S2[pos][key]
                        S[ss]['weights'] /= S2[pos]['coeff']
                    else:
                        m = apply_lev2knots(i, lev2knots, N)
                        S[ss] = tensor_grid(N, m, knots)
                    S[ss]['weights'] *= coeff[j]
                    coeff_condensed[ss] = coeff[j]
                    ss += 1
        else:
            for j in range(nn):
                if coeff[j] != 0:
                    i = C[j]
                    m = apply_lev2knots(i, lev2knots, N)
                    S[ss] = tensor_grid(N, m, knots)
                    S[ss]['weights'] *= coeff[j]
                    coeff_condensed[ss] = coeff[j]
                    ss += 1

        if map is not None:
            for ss in range(nb_grids):
                S[ss]['knots'] = map(S[ss]['knots'])

        if weights_coeff is not None:
            for ss in range(nb_grids):
                S[ss]['weights'] *= weights_coeff

        for ss in range(nb_grids):
            S[ss]['coeff'] = coeff_condensed[ss]

        ss = 0
        for j in range(nn):
            if coeff[j] != 0:
                i = C[j]
                S[ss]['idx'] = i
                ss += 1

    return S, C

def apply_lev2knots(i, lev2knots, N):
    m = np.zeros_like(i)
    for n in range(N):
        m[n] = lev2knots[n](i[n])
    return m

# Additional helper functions like tensor_grid, multiidx_gen, etc. need to be converted as well.

def issmolyak(S):
    """
    ISSMOLYAK(S) returns True if S is a smolyak sparse grid. 
    A smolyak sparse grid is a list of dictionaries with keys 
    'knots', 'weights', 'size', 'knots_per_dim', 'm', 'coeff', 'idx'.
    """

    required_keys = {'knots', 'weights', 'size', 'knots_per_dim', 'm', 'coeff', 'idx'}

    if isinstance(S, list) and len(S) >= 1 and all(isinstance(item, dict) for item in S):
        return all(required_keys == set(item.keys()) for item in S)
    else:
        return False


def tensor_grid(N, m, knots):
    """
    TENSOR_GRID generates a tensor grid and computes the corresponding weights.

    S = TENSOR_GRID(N, M, KNOTS) creates a tensor grid in N dimensions with M=[m1, m2, ... , m_N] points 
    in each direction. KNOTS is either a list containing the functions to be used 
    to generate the knots in each direction, i.e.         
        KNOTS=[knots_function1, knots_function2, ... ]
    or a single function, to be used to generate the 1D knots in every direction, i.e.
        KNOTS=knots_function1
    In both cases, the header of knots_function is x, w = knots_function(m)

    The output S is a dictionary containing the information on the tensor grid:
        S['knots']:        list containing the tensor grid knots
        S['weights']:      list containing the corresponding weights
        S['size']:         size of the tensor grid, S['size'] = prod(m)
        S['knots_per_dim']: list (N components), each component is the set of 1D knots used
                            to build the tensor grid    
        S['m']:            input list m, m[i] = len(S['knots_per_dim'][i])
    """

    # if knots is a simple function, we replicate it in a list
    if callable(knots):
        fknots = knots
        knots = [fknots for _ in range(N)]

    sz = 1
    for mi in m:
        sz *= mi

    S = {
        'knots': [[0] * sz for _ in range(N)],
        'weights': [1] * sz,
        'size': sz,
        'knots_per_dim': [None] * N,
        'm': m
    }

    # generate the pattern that will be used for knots and weights matrices
    pattern = generate_pattern(m)

    for n in range(N):
        xx, ww = knots[n](m[n])
        S['knots_per_dim'][n] = xx
        for j in range(sz):
            S['knots'][n][j] = xx[pattern[n][j] - 1]  # -1 because Python uses 0-based indexing
            S['weights'][j] *= ww[pattern[n][j] - 1]

    return S

def generate_pattern(m):
    """
    generate_pattern(m)
    
    Given m=[m1, m2, m3, m4, ... , mN], generates a matrix that can be used to generate the cartesian product
    of {1,2,...,m1} x {1,2,...,m2} x {1,2,...m3} x ....
    
    e.g.
    generate_pattern([3, 2, 2])
    
    pattern =
    [[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
     [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
     [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]]
    """

    N = len(m)
    pattern = np.zeros((N, np.prod(m)), dtype=int)

    for k in range(N):
        p = int(np.prod(m[:k]))
        q = int(np.prod(m[k+1:] if k+1 < N else [1]))
        
        lb = p * m[k]
        base = np.zeros(lb, dtype=int)
        
        bb = 0
        for j in range(m[k]):
            base[bb:bb+p] = j + 1  # +1 because Python uses 0-based indexing
            bb += p
        
        pp = 0
        for j in range(q):
            pattern[k, pp:pp+lb] = base
            pp += lb

    return pattern.tolist()

# Example usage:
print(generate_pattern([3, 2, 2]))
