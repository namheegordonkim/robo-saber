import os


def find_parent_dir(path, dir_name):
    while path != os.path.dirname(path):
        if os.path.basename(path) == dir_name:
            return path
        path = os.path.dirname(path)
    return None


proj_name = "Beaty"
proj_dir = find_parent_dir(os.path.abspath(__file__), proj_name)
src_dir = find_parent_dir(os.path.abspath(__file__), "src")
