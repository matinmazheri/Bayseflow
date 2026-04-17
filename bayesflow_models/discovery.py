from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from pkgutil import iter_modules
from bayesflow_models.interfaces import ModelSpec
from bayesflow_models.interfaces import Workflow
from typing import Callable, Any

# I have to change the the discovery startegy in a way that it does not impose the attribute name whether it Workflows or ModelSpec. It should be more type-dependent 
def _collect_specs_from_module(module):
    if hasattr(module, "MODEL_SPECS"):
        specs = module.MODEL_SPECS
        
        # 1. Check if it is a list
        if not isinstance(specs, list):
            raise ValueError(f"MODEL_SPECS in {module.__name__} must be a list.")
            
        # 2. Check if everything INSIDE the list is a ModelSpec
        if not all(isinstance(spec, ModelSpec) for spec in specs):
            raise ValueError(f"All items in MODEL_SPECS in {module.__name__} must be ModelSpec instances.")
            
        return specs

    if hasattr(module, "register_models"):
        # Use the built-in callable() check
        if not isinstance(module.register_models, Callable[...]):
            raise ValueError(f"register_models in {module.__name__} must be a function.")
            
        return module.register_models()
    return []

def discover_model_specs(plugin_paths=None) -> dict[str,ModelSpec]:
    specs = {}

    package = import_module("bayesflow_models.internalpluginpath")
    for module_info in iter_modules(package.__path__):
        module = import_module(f"bayesflow_models.internalpluginpath.{module_info.name}")
        for spec in _collect_specs_from_module(module):
            if spec.name in specs:
                raise ValueError(f"Duplicate model spec name: {spec.name}")
            specs[spec.name] = spec

    for plugin_path in plugin_paths or []:
        for py_file in Path(plugin_path).glob("*.py"):
            mod_spec = spec_from_file_location(py_file.stem, py_file)
            module = module_from_spec(mod_spec)
            mod_spec.loader.exec_module(module)

            for spec in _collect_specs_from_module(module):
                if spec.name in specs:
                    raise ValueError(f"Duplicate model spec name: {spec.name}")
                specs[spec.name] = spec

    return specs


def _collect_workflow_from_module(module) -> list:
    if hasattr(module, "WORKFLOWS"):
        workflows = module.WORKFLOWS
        
        # 1. Check if it is a list
        if not isinstance(workflows, list):
            raise ValueError(f"WORKFLOWS in {module.__name__} must be a list.")
            
        # 2. Check if everything INSIDE the list is a ModelSpec
        if not all(isinstance(workflow, Workflow) for workflow in workflows):
            raise ValueError(f"All items in MODEL_SPECS in {module.__name__} must be ModelSpec instances.")
            
        return workflows

    if hasattr(module, "register_workflow"):
        # Use the built-in callable() check
        if not isinstance(module.register_workflow, Callable[...,Workflow]):
            raise ValueError(f"register_workflow in {module.__name__} must be a function returing workflow plugin.")
            
        return module.register_workflow()
    return []

def discover_workflows(plugin_paths:list =[]) -> dict[str,Workflow]:
    workflows  = {}
    package = import_module("bayesflow_models.internalpluginpath")
    for module_info in iter_modules(package.__path__):
        module = import_module(f"bayesflow_models.internalpluginpath.{module_info.name}")
        for workflow in _collect_workflow_from_module(module):
            workflows[workflow.name] = workflow
  

    for plugin_path in plugin_paths:
        for py_file in Path(plugin_path).glob("*.py"):
            mod_spec = spec_from_file_location(py_file.stem, py_file)
            module = module_from_spec(mod_spec)
            mod_spec.loader.exec_module(module)
            for workflow in _collect_workflow_from_module(module):
                workflows[workflow.name] = workflow

    if len(workflows)==0:
        raise ValueError("No workflow is found")
    return workflows