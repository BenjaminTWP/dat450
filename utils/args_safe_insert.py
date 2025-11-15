import inspect

def safe_create_class(CompClass, args):
    valid_keys = inspect.signature(CompClass).parameters.keys()
    args_dict = {k: v for k, v in vars(args).items() if k in valid_keys}
    return CompClass(**args_dict)