import numpy as np
import os

##################################################
### Scene specific parameters
##################################################
scene_name = "office_0"
split_name = "tmp"
left_right = "left"
location_idx = 0

##################################################
### Directory
##################################################
dirs = dict(
    output_dir="data/replica_sim/{}/{}/{}/{:02}".format(
        scene_name,
        split_name,
        left_right,
        location_idx,
    ),
    data_dir="data/replica_v1/{}".format(scene_name),
)

##################################################
### Simulator
##################################################
simulator = dict(
    physics=dict(
        enable=True,
        gravity=[0.0, -10.0, 0.0],
    ),
    scene_id=os.path.join(dirs["data_dir"], "habitat/replicaSDK_stage.stage_config.json"),
    duration=100,
    FPS=20,
)

##################################################
### Agent
##################################################
agent = dict(
    position=[0.0, 0.0, 0.0],
    rotation=[0.0, 0.0, 0.0],
    motion_profile=dict(
        radius=1,
        motion_type="predefined",
    ),
)

##################################################
### Camera
##################################################
fov = lambda size, focal: np.rad2deg(np.arctan((size / 2) / focal)) * 2

camera = dict(
    fps=simulator["FPS"],
    pinhole=dict(
        enable=True,
        cam_type=["color", "depth", "semantic"],
        resolution_hw=[256, 512],
        orientation_type="horizontal",
        horizontal=dict(
            num_rot=1,
        ),
        fov=(fov(256, 512), fov(512, 512)),
    ),
    equirectangular=dict(
        enable=True,
        cam_type=["color", "depth", "semantic"],
        resolution_hw=[512, 1024],
        poses=[
            [0, 0, 0],
        ],
    ),
)

##################################################
### Simulation output storage
##################################################
sim_output = dict(
    save_video=True,
    save_frame=True,
    save_pose=True,
    save_K=True,
    frame_suffix=".png",
    depth_png_scale=6553.5,
    clear_old_output=True,
    force_clear=True,
)
