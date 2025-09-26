import sys
import os

# Add the current directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the GUI
from gui.gui_app import main

if __name__ == "__main__":
    main()
