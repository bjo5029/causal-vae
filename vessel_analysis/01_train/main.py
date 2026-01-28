import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_core')))

from train import main

if __name__ == "__main__":
    main()
