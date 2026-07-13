import numpy as np
import os

_base_ = "../../default.py"

##################################################
### General
##################################################
general = dict(
    dataset="Replica",
    scene="room2",
    num_iter=2000,
    device='cuda'
)

##################################################
### Directories
##################################################
dirs = dict(
    data_dir="data/",
    result_dir="results/",
    cfg_dir=os.path.join("configs", general['dataset'], general['scene'])
)

##################################################
### Simulator
##################################################
sim = dict(method="habitat_v2")
if sim["method"] == "habitat_v2":
    sim.update(
        habitat_cfg=os.path.join(dirs['cfg_dir'], "habitat.py")
    )

##################################################
### SLAM (Safe route: non-semantic SplaTAM)
##################################################
slam = dict(method="splatam")
if slam["method"] == "splatam":
    slam.update(
        room_cfg=f"{dirs['cfg_dir']}/../replica_splatam_s.py",
        enable_active_planning=False,
        dataset_eval_basedir="data/Replica",
        bbox_bound=[[-2.1, 2.5], [-3.2, 2], [-1.3, 2.0]],
        bbox_voxel_size=0.05,
        surface_dist_thre=0.5,
        refine_map_iter=30,
        use_global_keyframe=False,
        override=dict(
            map_every=5,
            report_global_progress_every=5,
            tracking=dict(
                use_gt_poses=True,
            )
        )
    )

##################################################
### Planner (safe passive mapping)
##################################################
planner = dict(
    method="predefined_traj",
    trans_step_size=0.1,
    rot_step_size=10,
    up_dir=np.array([0, 0, 1]),
    use_traj_pose=True,
    SLAMData_dir=os.path.join(dirs["data_dir"], "Replica", general['scene']),
    local_planner_method="RRTNaruto",
)

if planner["local_planner_method"] == "RRTNaruto":
    planner.update(
        rrt_step_size=planner['trans_step_size'] / slam['bbox_voxel_size'],
        rrt_step_amplifier=10,
        rrt_maxz=100,
        rrt_max_iter=None,
        rrt_z_levels=None,
        enable_eval=False,
        enable_direct_line=True,
    )

##################################################
### Visualization
##################################################
visualizer = dict(
    method="active_gs",
    vis_rgbd=False,
    vis_rgbd_max_depth=10,
    save_rendered_rgbd=False,
)
