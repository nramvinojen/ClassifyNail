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
    install('numpy')
    install('Pillow')
    install('opencv-python')
    install('keras')
    install('argparse')   
    install('flask')
    install('tensorflow')

        

if __name__ == '__main__':
    main()
        