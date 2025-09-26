import os
import shutil
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        'core',
        'strategies',
        'gui',
        'utils',
        'config',
        'data',
        'outputs',
        'tests'
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def move_files():
    """Move files to their appropriate directories."""
    # Core functionality
    core_files = ['main.py', 'tuner.py']
    for file in core_files:
        if os.path.exists(file):
            shutil.move(file, f'core/{file}')
    
    # GUI files
    gui_files = ['main_window.py', 'gui_app.py', 'test_gui.py', 'ui_components.py']
    for file in gui_files:
        if os.path.exists(file):
            shutil.move(file, f'gui/{file}')
    
    # Strategy files
    if os.path.exists('strategy.py'):
        shutil.move('strategy.py', 'strategies/__init__.py')
    
    # Config files
    if os.path.exists('config.py'):
        shutil.move('config.py', 'config/__init__.py')
    
    # Create empty __init__.py files
    for dir_name in ['core', 'strategies', 'gui', 'utils', 'config']:
        Path(f"{dir_name}/__init__.py").touch()

def main():
    print("Starting file organization...")
    create_directories()
    move_files()
    print("File organization complete!")

if __name__ == "__main__":
    main()
