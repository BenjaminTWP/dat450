import argparse
import yaml
from paths import BASE_PATH

TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool
}

def get_runtime_args():
    parser = argparse.ArgumentParser()

    with open(BASE_PATH+"/args.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    for arg_name, attributes in cfg["arguments"].items():
        arg_type = TYPE_MAP.get(attributes.get("type"), str)

        parser.add_argument(
            f"--{arg_name}",
            type=arg_type,
            help=attributes.get("help"),
            default=attributes.get("default")
        )

    return parser.parse_args()

