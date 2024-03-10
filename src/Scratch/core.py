# Maurice Hanisch mhanisc@ethz.ch
# Created 2023-10-20

import os
import inspect
import warnings

def find_and_create_scratch(max_traversal_depth=20):
    original_path = os.getcwd()
    scratch_path = None
    traversal_count = 0  # Counter to limit the number of upward traversals

    try:
        while True:
            # Check if all specified directories and files exist in the current directory
            if all(os.path.exists(item) for item in ['.git', 'README.md', 'src', 'libs']):
                scratch_path = os.path.join(os.getcwd(), '.Scratch')
                if not os.path.exists('.Scratch'):
                    os.mkdir('.Scratch')
                break

            # Move up a directory level
            os.chdir('..')
            traversal_count += 1

            # Check if we reach the root directory or maximum traversal depth
            if os.getcwd() == os.path.abspath(os.sep) or traversal_count >= max_traversal_depth:
                raise FileNotFoundError("Project root with .git, README.md, src, and libs not found.")
                break
    except Exception as e:
        print(f"Error encountered: {e}")
        scratch_path = None
    finally:
        # Ensure we navigate back to the original directory
        os.chdir(original_path)

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
