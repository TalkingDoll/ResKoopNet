import numpy as np 

def lev2knots_lin(i):
    m = i
    return mslnl


def lev2knots_doubling(i):
    """
    Relation level / number of points:
    m = 2^{i-1}+1, for i>1
    m=1            for i=1
    m=0            for i=0

    Args:
    - i (list or numpy array): Input levels.

    Returns:
    - m (numpy array): Number of points corresponding to each level.v/v/bc/ 
    """
    
    i = np.array(i)  # Convert input to numpy array for vectorized operations
    m = [2 ** (jx - 1) + 1 for jx in i]
    
    print(i)
    print(m)
    for k in range(m.shape[0]):
        if i[k] == 1:
            m[k] = 1
        m[i == 1] = 1
        if i[k] == 0:
            m[k] = 0
    
    return m


def define_functions_for_rule(rule, input2):
    """
    Sets the functions lev2nodes and idxset to use in smolyak_grid to build the desired sparse grid.

    Args:
    - rule (str): Rule type, can be 'TP', 'TD', 'HC', or 'SM'.
    - input2 (int or list): Number of variables (if scalar) or rates vector.

    Returns:
    - lev2nodes (function): Corresponding function based on the rule.
    - idxset (function): Corresponding function based on the rule.
    """

    # Check if input2 is scalar (int) or list
    if isinstance(input2, int):
        N = input2
        rates = [1] * N
    else:
        rates = input2

    # Define the lev2nodes and idxset functions based on the rule
    if rule == 'TP':
        lev2nodes = lev2knots_lin
        idxset = lambda i: max([a*b for a, b in zip(rates[:len(i)], [x-1 for x in i])])
    elif rule == 'TD':
        lev2nodes = lev2knots_lin
        idxset = lambda i: sum([a*b for a, b in zip(rates[:len(i)], [x-1 for x in i])])
    elif rule == 'HC':
        lev2nodes = lev2knots_lin
        idxset = lambda i: np.prod(np.array(i) ** rates[:len(i)])
    elif rule == 'SM':
        lev2nodes = lev2knots_doubling
        idxset = lambda i: sum([a*b for a, b in zip(rates[:len(i)], [x-1 for x in i])])
    else:
        raise ValueError("Invalid rule type")

    return lev2nodes, idxset

