import isaaclab .sim as sim_utils
from isaaclab .actuators import ImplicitActuatorCfg
from isaaclab .assets .articulation import ArticulationCfg
from isaaclab .sensors import RayCasterCfg
from isaaclab .utils .assets import ISAACLAB_NUCLEUS_DIR



NexusTrex_CONFIG =ArticulationCfg (
spawn =sim_utils .UsdFileCfg (
usd_path =r"C:\Users\ROG\Desktop\FYP\IsaacLab\NexusTrex\assets\NexusTrex.usd",
activate_contact_sensors =True ,
rigid_props =sim_utils .RigidBodyPropertiesCfg (
rigid_body_enabled =True ,
max_linear_velocity =1000.0 ,
max_angular_velocity =1000.0 ,
max_depenetration_velocity =100.0 ,
enable_gyroscopic_forces =True ,
),
articulation_props =sim_utils .ArticulationRootPropertiesCfg (
enabled_self_collisions =False ,
solver_position_iteration_count =4 ,
solver_velocity_iteration_count =0 ,
sleep_threshold =0.005 ,
stabilization_threshold =0.001 ,
),
),
init_state =ArticulationCfg .InitialStateCfg (
pos =(0.0 ,0.0 ,0.35 ),
joint_pos ={".*":0.0 },
joint_vel ={".*":0.0 }
),
actuators ={

"left_legs_actuator":ImplicitActuatorCfg (
joint_names_expr =['DriveLeftMotor','DriveLeftPassive'],
effort_limit_sim =150.0 ,
velocity_limit_sim =15.0 ,
stiffness =300.0 ,
damping =30.0 ,
),
"right_legs_actuator":ImplicitActuatorCfg (
joint_names_expr =['DriveRightMotor','DriveRighttPassive'],
effort_limit_sim =150.0 ,
velocity_limit_sim =15.0 ,
stiffness =300.0 ,
damping =30.0 ,
),
"wheel_actuator":ImplicitActuatorCfg (
joint_names_expr =['MotorRight','MotorLeft'],
effort_limit_sim =50.0 ,
velocity_limit_sim =20.0 ,

stiffness =0.0 ,
damping =10.0 ,
),
},
)
