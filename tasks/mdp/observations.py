from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab .assets import Articulation ,RigidObject
from isaaclab .managers import SceneEntityCfg
from isaaclab .managers .manager_base import ManagerTermBase
from isaaclab .managers .manager_term_cfg import RewardTermCfg
from isaaclab .sensors import ContactSensor ,RayCaster

if TYPE_CHECKING :
    from isaaclab .envs import ManagerBasedRLEnv

def undesired_contacts (env :ManagerBasedRLEnv ,threshold :float ,sensor_cfg :SceneEntityCfg )->torch .Tensor :
    """Penalize undesired contacts as the number of violations that are above a threshold."""

    contact_sensor :ContactSensor =env .scene .sensors [sensor_cfg .name ]

    net_contact_forces =contact_sensor .data .net_forces_w_history
    is_contact =torch .max (torch .norm (net_contact_forces [:,:,sensor_cfg .body_ids ],dim =-1 ),dim =1 )[0 ]>threshold

    return torch .sum (is_contact ,dim =1 )
