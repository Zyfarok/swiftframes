import sys

#from typing import Tuple

def init() -> str:
    if(len(sys.argv) < 2):
        print('Error : You need to specify the name of the png sequence')
        exit(1)

    framesName = sys.argv[1]

    # TODO : Accept more arguments.
    
    return framesName