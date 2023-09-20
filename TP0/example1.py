"""
    Possible solutions for calculating the factorial. One uses recursion,
    the other uses a loop.
"""

def factorial_recurse(n):
    """
        Given the integer `n`, return the value `n!`. 
        Calculate via recursion. Note, to prevent
        infinite loops, this version returns 1 for all
        cases `n < 2`.

        :param n: Integer argument
        :type n: int
        :return: The factorial n!
    """    
    if n < 2: 
        return 1
    else:
        return n * factorial(n-1)

def factorial_loop(n):
    """
        Given the integer `n`, return the value `n!`. 
        Calculate via recursion.

        :param n: Integer argument
        :type n: int
        :return: The factorial n!
    """ 
    nfact = 1
    for x in range(2,n+1):
        nfact *= x
    return nfact

def factorial(n, method = 'loop'):
    """
        A front-facing function for calculating the factorial 
        of an integer using the specified algorithm.

        :param n: Integer argument
        :type n: int
        :param method: Specify 'loop' or 'recurse' for the 
                       two possible algorithms.
        :return: The factorial n!
    """
    # Input Checking
    if n < 0:
        raise ValueError('Cannot take factorial of n < 0.')

    if isinstance(n,int) == False:
        raise ValueError('Must have intenger input.')

    # Run the algorithms
    if method == 'loop':
        return factorial_loop(n)
    elif method == 'recurse':
        return factorial_recurse(n)
    else:
        raise ValueError('Uknown algorithm.')
