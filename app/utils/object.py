import sys

def singleton(cls):
    if "pytest" in sys.modules:
        return cls 
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance