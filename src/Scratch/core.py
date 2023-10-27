# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-20

import os
import inspect
import warnings


def find_and_create_scratch():
    original_path = os.getcwd()
    scratch_path = None
    while True:
        current_folder = os.path.basename(os.getcwd())
        if current_folder == 'Soft-Info':
            scratch_path = os.path.join(os.getcwd(), '.Scratch')
            if not os.path.exists('.Scratch'):
                os.mkdir('.Scratch')
            break
        else:
            os.chdir('..')
            if os.getcwd() == '/':  # Stop if we reach the root directory
                print("Soft-Info folder not found.")
                break
    os.chdir(original_path)  # Navigate back to original directory
    return scratch_path


def get_notebook_name():
    frame = inspect.currentframe()
    while frame:
        notebook_name = frame.f_globals.get('__vsc_ipynb_file__')
        if notebook_name:
            return os.path.basename(notebook_name).replace('.ipynb', '').replace(' ', '_')
        frame = frame.f_back  # Move to the previous frame in the call stack
    warnings.warn("Notebook name not found.")
    return 'not_defined_'  # Return a default value if the notebook name wasn't found
