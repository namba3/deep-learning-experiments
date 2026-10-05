"""Project optimizer implementations and learning-rate controllers."""

from .apollo import (
    APOLLO,
    APOLLOConfidence,
    APOLLOFallbackPolicy,
    APOLLOADAMW,
    APOLLOADAMWAutoSchedule,
    APOLLOLion,
    DualRotAPOLLO,
    RotAPOLLO,
    APOLLOAutoSchedule,
    APOLLOCAME,
    APOLLOCAMEAutoSchedule,
    APOLLOMini,
)
from .adamw import AdamWAutoSchedule, AdamWFP32State
from .adamw_lr_ema import (
    AdamWLowRankGradientEMA,
    AdamWLowRankGradientEMAConfidence,
)
from .adamw_lr_ema_conf_lrsf import AdamWLRSEMAConfLRSF
from .adamw_lrsf import AdamWLRSF
from .adamw_lrsf_lr import AdamWLRSLowRankPreconditioner
from .adamw_sf_lr import AdamWSFLowRankPreconditioner
from .came import CAME, CAMEAutoSchedule
from .came_lrsf import CAMESF, CAMELRSF
from .apollo_lrsf import APOLLOCAMELRSF
from .apollo_sf import APOLLOSF, APOLLOScheduleFree
from .projection_refresh import OrthogonalRefreshPolicy, ProjectionRefreshPolicy
from .muon import SingleDeviceMuon
from .muon_variants import AdaMuon, NorMuon
from .soap import SOAP
from .lion import Lion
from .schedulefree import AdamWScheduleFree, RAdamScheduleFree
from .factory import (
    CORE_OPTIMIZER_CHOICES,
    add_optimizer_argument,
    add_optimizer_override_argument,
    build_optimizer,
    is_schedule_free_optimizer,
)

__all__ = [
    "CAME",
    "CAMEAutoSchedule",
    "CAMELRSF",
    "CAMESF",
    "APOLLOCAMELRSF",
    "APOLLOSF",
    "APOLLOScheduleFree",
    "ProjectionRefreshPolicy",
    "OrthogonalRefreshPolicy",
    "AdamWAutoSchedule",
    "AdamWFP32State",
    "AdamWLowRankGradientEMA",
    "AdamWLowRankGradientEMAConfidence",
    "AdamWLRSEMAConfLRSF",
    "AdamWLRSF",
    "AdamWLRSLowRankPreconditioner",
    "AdamWSFLowRankPreconditioner",
    "AdamWScheduleFree",
    "RAdamScheduleFree",
    "SingleDeviceMuon",
    "Lion",
    "SOAP",
    "NorMuon",
    "AdaMuon",
    "APOLLO",
    "APOLLOConfidence",
    "APOLLOFallbackPolicy",
    "APOLLOADAMW",
    "APOLLOADAMWAutoSchedule",
    "APOLLOLion",
    "DualRotAPOLLO",
    "RotAPOLLO",
    "APOLLOAutoSchedule",
    "APOLLOMini",
    "APOLLOCAME",
    "APOLLOCAMEAutoSchedule",
    "CORE_OPTIMIZER_CHOICES",
    "add_optimizer_argument",
    "add_optimizer_override_argument",
    "build_optimizer",
    "is_schedule_free_optimizer",
]
