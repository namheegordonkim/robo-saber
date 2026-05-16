import pickle
import sys
import os
import inspect
import importlib
from hashlib import sha256
import gzip
import atexit
from shutil import rmtree


# Get dependencies local to root in import order (root has to be normalized!)
def find_dependencies(module, root):
    def find(module, seen):
        seen.add(module)
        imports = filter(
            lambda m: hasattr(m, "__file__")
            and os.path.normpath(str(m.__file__)).startswith(root),
            [inspect.getmodule(m[1]) for m in inspect.getmembers(module)],
        )
        return [x for m in imports if m not in seen for x in find(m, seen)] + [module]

    return find(module, set())


# Temporarily replace system modules with alternative ones
class MonkeyPatch:
    def __init__(self, modules):
        self.modules = modules

    def __enter__(self):
        self.originals = {k: v for k, v in sys.modules.items()}
        sys.modules.update(self.modules)

    def __exit__(self, type, value, traceback):
        sys.modules.update(self.originals)
        for name in self.modules:
            if name not in self.originals:
                sys.modules.pop(name)


# Create a fresh module given a name and the path to source
def create_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Place source in temporary file and create module.
# In-memory would feel cleaner but doesn't show up in debugger :/
def place_module(name, path, source):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    open(path, "wb").write(source)
    print(f"created {path}")
    return create_module(name, path)


# Save model to path, store dependencies local to a specified root
def dump(sourceable, root=os.path.abspath(os.getcwd())):

    # If the model was previously loaded, use original source
    modules = {}
    if id(sourceable) in stored_modules:
        dir, modules = stored_modules[id(sourceable)]
        root = f"{root}/{dir}/"

    with MonkeyPatch(modules):
        # Create an import-order list of (name, path, file contents)
        root = os.path.normpath(root)
        modules = find_dependencies(
            importlib.import_module(sourceable.__module__), root
        )
        info = lambda n, f: [n, os.path.relpath(f, root), open(f, "rb").read()]
        sources = [info(m.__name__, m.__file__) for m in modules]

        # Point the object to a non-__main__ version of the class
        replace = {}
        if sourceable.__module__ == "__main__":
            sourceable.__module__ = os.path.splitext(os.path.split(sources[-1][1])[1])[
                0
            ]
            sources[-1][0] = sourceable.__module__
            module = create_module(sourceable.__module__, sources[-1][1])
            sourceable.__class__ = module.__dict__[sourceable.__class__.__name__]
            replace = {sourceable.__module__: module}

        with MonkeyPatch(replace):
            sourceable_compressed = gzip.compress(pickle.dumps(sourceable))
            return {
                "sources": sources,
                "sourceable": sourceable_compressed,
            }


# Load model from path
def load(d: dict):
    sources = d["sources"]
    sourceable_compressed = d["sourceable"]
    dir = "pickle_source/" + sha256(sourceable_compressed).hexdigest()

    # Monkey patch sys modules to look like they did when pickle was made
    modules = {
        name: place_module(name, f"{dir}/{path}", source)
        for name, path, source in sources
    }
    # with MonkeyPatch(modules):
    #     sourceable = pickle.loads(gzip.decompress(sourceable_compressed))
    #     stored_modules[id(sourceable)] = (dir, {k: v for k, v in sys.modules.items()})
    sourceable = pickle.loads(gzip.decompress(sourceable_compressed))
    return sourceable


# Remove contents of pickle_source at exit to avoid it growing without bound
def clean():
    if os.path.exists("pickle_source"):
        for _, dirs, _ in os.walk("pickle_source"):
            for d in dirs:
                rmtree(f"pickle_source/{d}")


atexit.register(clean)
stored_modules = {}
