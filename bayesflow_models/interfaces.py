from dataclasses import dataclass
from typing import Callable, Any
from pathlib import Path


@dataclass
class ModelSpec:
    name: str
    workflow: str
    simulator_factory: Callable[[], Any]
    adapter_factory: Callable[[], Any]
    family: str
    par_names: list[str]
    version: str
    description: str = ""

@dataclass(frozen=True)
class WorkflowContext:
    artifact_id: str
    checkpoint_path: Path       # e.g. checkpoints/model_A.keras
    metadata_path: Path         # e.g. checkpoints/model_A.meta.json
    results_dir: Path
    logs_dir: Path
    device: Any

@dataclass
class WorkflowResult:
    artifact_id: str
    spec_name: str
    workflow: str
    checkpoint_path: str
    metadata_path: str
    results_dir: str
    logs_dir: str
    status: str
    history: Any = None

TrainFn = Callable[[ModelSpec, dict], WorkflowResult]
ResumeFn = Callable[[str, dict], WorkflowResult]
RecoveryFn = Callable[[str, dict, WorkflowContext], WorkflowResult]

@dataclass(frozen=True)
class Workflow:
    name: str
    train_fn: TrainFn 
    resume_fn: ResumeFn
    recovery_fn: RecoveryFn


