import argparse
import yaml

TYPE_MAP = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool
}

def get_runtime_args(run_dir):
    parser = argparse.ArgumentParser()

    file_paths = [
        run_dir + "/args.yaml",
        "utils/tokenizer_args.yaml",
        "utils/training_args.yaml"
    ]

    for file_path in file_paths:

        with open(file_path, "r") as f:
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

