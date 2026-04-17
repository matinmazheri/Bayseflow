from bayesflow_models.interfaces import ModelSpec
from bayesflow_models.DDM_DC_Pedestrain_TrialWise import (
      model_DC_TrialWise,
      model_DC,
      get_adapter,
      get_adapter_trialwise,
      PAR_NAMES
  )

MODEL_SPECS = [
    ModelSpec(
        name="model_DC_TwoBoundary_Simplest",
        workflow="builtin",
        simulator_factory=lambda: model_DC,
        adapter_factory=get_adapter,
        family="ddm_dc",
        par_names = PAR_NAMES,
        version="1.0",
        description="Built-in simplest DDM with the whole TTA in the simulator",
    ),
        ModelSpec(
        name="model_DC_TwoBoundary_TiralWise_Simplest",
        workflow="builtin",
        simulator_factory=lambda: model_DC_TrialWise,
        adapter_factory=get_adapter_trialwise,
        family="ddm_dc",
        par_names = PAR_NAMES,
        version="1.0",
        description="Built-in simplest DDM with TiralWise TA conditions in the simulator",
    ),

]