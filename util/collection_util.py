from collections import defaultdict
from itertools import groupby
from typing import Dict, Iterable, TypeVar, Callable, List
from functools import singledispatch

T1 = TypeVar('T1')
T2 = TypeVar('T2')

def groupbydefault(l: Iterable[T1], f: Callable[[T1],T2]) -> Dict[T2,List[T1]]:
    """group elements by key from a list into a defaultdict of lists.
    
    Arguments:
        l {List[T1]} -- List of elements
        f {Callable[[T1],T2]} -- Key to group element by
    
    Returns:
        Dict[T2,List[T1]] -- the defaultdict of lists
    """
    return defaultdict(list, ((k, list(g)) for k, g in groupby(l, key=f)))

def lambdamap(f: Callable[[T1],T2]) -> Callable[[Iterable[T1]],List[T2]]:
    """return a function that apply a mapping to the elements of some Iterable
    
    Arguments:
        ls {List[T1]} -- [description]
        f {Callable[[T1],T2]} -- [description]
    
    Returns:
        Callable[List[T1],List[T2]] -- [description]
    """
    return lambda l: list(map(f,l))


    
