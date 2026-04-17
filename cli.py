import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="When using torch backend")

import bayesflow
from torch import device,cuda
import yaml
import copy
from bayesflow_models.discovery import discover_model_specs
from bayesflow_models.interfaces import WorkflowContext, WorkflowResult, ModelSpec
import bayesflow_models.workflow as wf
from bayesflow_models.discovery import discover_workflows
import logging
from typing import Union, Any

DEFAULT_CONFIG = {
    "device": device('cuda' if cuda.is_available() else 'cpu'),
    "training": {
        "n_sim": 100,          # Simulations per epoch
        "epochs": 10,          # Total epochs to train
        "batch_size": 32,
        "resume_epochs": 3,    # Additional epochs when resuming
    },
    "recovery": {
        "n_test_sims": 50,    # Simulations for recovery evaluation
        "n_posterior_samples": 50,  # MCMC samples for posterior
        "mode": "visualize" # visualize | save_only
    },
    "paths": {
        "checkpoints": "trained_model1/checkpoints",
        "results": "results",
        "real_data": "real_data",
        "logs": "logs"
    }
}

def setup_directories(config: dict):
    """Ensure all required directories exist"""
    assert config["paths"] != None
    for key, path in config["paths"].items():
        Path(path).mkdir(parents=True, exist_ok=True)
    print("✓ Directories setup complete")

def build_runtime_config(args):
    config = copy.deepcopy(DEFAULT_CONFIG)

    cwd = Path.cwd()

    config["paths"]["checkpoints"] = str(cwd / "checkpoints")
    config["paths"]["results"] = str(cwd / "results")
    config["paths"]["logs"] = str(cwd / "logs")

    if args.device is not None:
        config["device"] = device(args.device)

    if args.checkpoint_dir is not None:
        config["paths"]["checkpoints"] = args.checkpoint_dir
    if args.results_dir is not None:
        config["paths"]["results"] = args.results_dir
    if args.logs_dir is not None:
        config["paths"]["logs"] = args.logs_dir

    if args.command == "train":
        config["training"]["n_sim"] = args.n_sim
        config["training"]["epochs"] = args.epochs
        config["training"]["batch_size"] = args.batch_size

    if args.command == "resume":
        config["training"]["resume_epochs"] = args.epochs
        config["training"]["batch_size"] = args.batch_size

    if args.command == "recovery":
        config["recovery"]["n_test_sims"] = args.n_test_sims
        config["recovery"]["n_posterior_samples"] = args.n_posterior_samples
        config["recovery"]["mode"] = args.mode
    
    return config

def update_config(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value,dict) and isinstance(base.get(key), dict):
            update_config(base[key],value)
        else:
            base[key] = value
    return base

# There is no need for load config as all the details of the implmentation will be determined via terminal.
def load_config(path):
    if path is None:
        raise ValueError("The path input to load config is None")
    
    path = Path(path)
    with open(path, "r") as f:
        if path.suffix in {".yml", ".yaml"}:
            user_config = yaml.safe_load(f)
        elif path.suffix == ".json":
            user_config = json.load(f)
        else:
            raise ValueError("Unsupported config format: {path.suffix}")
    return user_config


## Helper function for terminal-based of visualization of data stored in format of list of dictionary
def get_value(obj: Union[dict,object], key: str, default=None)-> Any:
    """Unified getter for dicts and objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def display_header(headers: list[str], header_widths: list):
    header = "  ".join(h.ljust(w) for h, w in zip(headers, header_widths))
    print(header)

def dict_to_list(dict: dict)-> list:
    a = []
    for k,v in dict.items():
        a.append(v)
    return a

def extract_header_widths(headers: list[str],records: list[Union[dict,object]], keys:list[str])-> list[int]:
    header_widths = [max(len(header), max(len(get_value(r,keys[i])) for r in records)) for i,header in enumerate(headers)]
    return header_widths
def display_rows(records: list[Union[dict,object]],header_widths: list[int], keys:list[str]):
    for row in records:
        text = "  ".join(get_value(row,k).ljust(w) for k, w in zip(keys, header_widths))
        print(text)


def display_table(headers: list[str], records: list[dict], keys: list[str]):
    header_widths = extract_header_widths(headers=headers,records=records,keys=keys)
    # header_widths = [max(len(header), max(len(r[keys[i]]) for r in records)) for i,header in enumerate(headers)]
    display_header(headers=headers,header_widths= header_widths)
    display_rows(records=records,header_widths=header_widths, keys=keys)


def print_checkpoints_table(records: list[dict]) -> None:
      if not records:
          print("No checkpoints found.")
          return

      headers = ["ALIAS", "SPEC NAME", "STATUS", "CREATED","RESUME"]
      keys = ["alias","spec_name","status","created_display","resumed_display"]
      display_table(headers=headers,records=records,keys=keys)

def print_models_table(specs: dict) -> None:
      if not specs:
          print("No spec found.")
          return

      headers = ["SPEC NAME","FAMILY","DESCRIPTION"]
      keys = ["name","family","description"]
      records = dict_to_list(specs)
      display_table(headers=headers, records=records, keys=keys)



def handle_checkpoints_list(config):
      records = wf.list_checkpoint_records(config)
      print_checkpoints_table(records)

def handle_models_list(specs):
    print_models_table(specs=specs)

def handle_config(args) -> dict:
    user_args_dict = None
    if args.config:
          if args.plugin_path == None or args.workflow == "builtin":
             raise ValueError("In case of optional config you need to define workflow of your own")
          user_args_dict = load_config(args)
    if user_args_dict:
            return user_args_dict
    else:
            return build_runtime_config(args)   


def handle_train(args):
    config = build_runtime_config(args)
    specs = discover_model_specs(args.plugin_path)

    for spec_name in args.models:
        spec = specs[spec_name]
        wf.train_from_spec(spec, config)


def build_parser():
    parser = argparse.ArgumentParser(prog="normalized flow")

    parser.add_argument("--device", "-d", default=device('cpu'), help="Select the device of the torch")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument("--plugin-path",action="append",default=[],help="Extra directory containing external model spec modules.")
    parser.add_argument("--config", type=str,help="You can input your config assuming that you import your workflow plugin")

    subparsers = parser.add_subparsers(dest="command", required=True)

    models_parser = subparsers.add_parser("models", help="Model registry commands")
    models_sub = models_parser.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list", help="List available model specs")

    checkpoints_parser = subparsers.add_parser("checkpoints", help="it allow monitoring or processing checkpoints files")
    checkpoints_sub = checkpoints_parser.add_subparsers(dest="checkpoints_command",required=True)
    checkpoints_sub.add_parser("list", help="show all information of the save checkpoint files")


    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--models", nargs="+", required=True)
    train_parser.add_argument("--n-sim", type=int, default=100)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--artifacts", nargs="+", required=True)
    resume_parser.add_argument("--n-sim", type=int, default=100)
    resume_parser.add_argument("--epochs", type=int, default=3)
    resume_parser.add_argument("--batch-size", type=int, default=32)
    resume_parser.add_argument("--mode",choices=["new","old"],default="old")

    recovery_parser = subparsers.add_parser("recovery")
    recovery_parser.add_argument("--artifacts", nargs="+", required=True)
    recovery_parser.add_argument("--n-test-sims", type=int, default=50)
    recovery_parser.add_argument("--n-posterior-samples", type=int, default=50)
    recovery_parser.add_argument("--mode", choices=["save_only", "visualize"], default="save_only")

    return parser



def main():
      parser = build_parser()
      args = parser.parse_args()
      # discoveries from plugin path and logs if there is no plugin paths or failed discoveries
      # discover specs from plugin path 
      specs = discover_model_specs(plugin_paths=args.plugin_path)
      if args.command == "models" and args.models_command == "list":
        handle_models_list(specs=specs)

        return
      # discover workflow
      workflows = discover_workflows(plugin_paths=args.plugin_path)

      # Find out optional config which matches the user WorkFlow Plugin
      config = handle_config(args)

      # Set up directories
      setup_directories(config)



      
      if args.command == "checkpoints" and args.checkpoints_command == "list":
        handle_checkpoints_list(config=config)
        return
      if args.command == "train":
        for spec_name in args.models:
            if spec_name not in specs.keys():
                raise ValueError(f"Model {spec_name} has not found in detected models")
            spec = specs[spec_name]
            if spec.workflow not in workflows.keys():
                raise ValueError(f"workflow {spec.workflow} has not found in detected workflows")
            workflow = workflows[spec.workflow]
            workflow.train_fn(spec, config)
      
      if args.command == "resume":
        for ref in args.artifacts:
            metadata = wf.resolve_artifact_ref(ref, config)
            spec_name = metadata["spec_name"]
            if spec_name not in specs.keys():
                raise ValueError(f"Model {spec_name} has not found in detected models")
            spec = specs[spec_name]
            config_added_alia = copy.deepcopy(config)
            config_added_alia["metadata"] = metadata
            config_added_alia["mode"] = args.mode
            if spec.workflow not in workflows.keys():
                raise ValueError(f"workflow {spec.workflow} has not found in detected workflows")
            workflow = workflows[spec.workflow]
            workflow.resume_fn(spec, config_added_alia)  


      if args.command == "recovery":
        for ref in args.artifacts:
            metadata = wf.resolve_artifact_ref(ref, config)
            spec_name = metadata["spec_name"]
            if spec_name not in specs.keys():
                raise ValueError(f"Model {spec_name} has not found in detected models")
            spec = specs[spec_name]
            config_added_alia = copy.deepcopy(config)
            config_added_alia["metadata"] = metadata
            config_added_alia["mode"] = args.mode
            if spec.workflow not in workflows.keys():
                raise ValueError(f"workflow {spec.workflow} has not found in detected workflows")
            workflow = workflows[spec.workflow]
            workflow.recovery_fn(spec, config_added_alia)
           

   

if __name__ == "__main__":
    main()
   
