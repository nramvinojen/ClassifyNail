# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 09:11:35 2020

@author: Ramvinojen
"""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
def main():
    install('keras')
    install('tensorflow==2.0')
    install('numpy')
    install('sklearn')
    install('scikit-learn')
    install('Pillow')
    #install('opencv-python')
    install('opencv-python-headless')
    install('argparse')   
    install('flask')

        

if __name__ == '__main__':
    main()
        
