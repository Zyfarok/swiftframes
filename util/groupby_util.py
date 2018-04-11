from collections import defaultdict
from itertools import groupby
from typing import Dict, List, TypeVar, Callable

T1 = TypeVar('T1')
T2 = TypeVar('T2')

def groupbydefault(l: List[T1], f: Callable[[T1],T2]) -> Dict[T2,List[T1]]:
    """group elements by key from a list into a defaultdict of lists.
    
    Arguments:
        l {List[T1]} -- List of elements
        f {Callable[[T1],T2]} -- Key to group element by
    
    Returns:
        Dict[T2,List[T1]] -- the defaultdict of lists
    """
    d = defaultdict(list)
    grouped = groupby(l, key=f)
    for k, v in grouped:
        d[k].append(v)
    return d