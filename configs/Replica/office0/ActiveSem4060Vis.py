import numpy as np
import os

_base_ = "../../default.py"

##################################################
### General
##################################################
general = dict(
    dataset="Replica",
    scene="office0",
    num_iter=300,
    device="cuda",
)

##################################################
### Directories
##################################################
dirs = dict(
    data_dir="data/",
    result_dir="results/",
    cfg_dir=os.path.join("configs", general["dataset"], general["scene"]),
)

##################################################
### Simulator
##################################################
sim = dict(
    method="habitat_v2",
)

if sim["method"] == "habitat_v2":
    sim.update(
        habitat_cfg=os.path.join(dirs["cfg_dir"], "habitat_4060_vis.py"),
    )

##################################################
### SLAM
##################################################
slam = dict(
    method="semsplatam",
)

if slam["method"] == "semsplatam":
    slam.update(
        room_cfg=f"{dirs['cfg_dir']}/../replica_splatam_s.py",
        enable_active_planning=True,
        dataset_eval_basedir="data/Replica",
        bbox_bound=[[-2.1, 2.5], [-3.2, 2], [-1.3, 2.0]],
        bbox_voxel_size=0.05,
        surface_dist_thre=0.5,
        find_free_indices_bs=128,
        find_free_indices_occ_bs=512,
        refine_map_iter=20,
        use_global_keyframe=False,
        num_topk_logits=16,
        num_semantic_classes=102,
        lambda_hel=0.8,
        lambda_cosine=0.2,
        uncert_mask_thres=3.0,
        semantic_dir="./data/replica_v1/office_0/habitat/",
        class_info_file="./configs/Replica/office0/class_info_file.json",
        semantic_device="cuda:0",
        oneformer_checkpoint="lly00412/oneformer-replica-finetune",
        coco_checkpoint="shi-labs/oneformer_coco_swin_large",
        ade20k_checkpoint="shi-labs/oneformer_ade20k_swin_large",
        override=dict(
            map_every=10,
            keyframe_every=10,
            mapping_window_size=2,
            report_global_progress_every=10,
            tracking=dict(
                use_gt_poses=True,
                num_iters=3,
            ),
            mapping=dict(
                num_iters=3,
            ),
            data=dict(
                desired_image_height=256,
                desired_image_width=512,
                tracking_image_height=256,
                tracking_image_width=512,
                densification_image_height=256,
                densification_image_width=512,
            ),
        ),
    )

##################################################
### Planner
##################################################
planner = dict(
    method="active_gsv2",
    max_exploration_steps=300,
    post_refine_steps=50,
    max_refinement_steps=50,
    num_exploration_stage=1,
    gs_z_levels=[
        [35],
    ],
    num_dir_samples=[
        5,
    ],
    xy_sampling_step=[
        1.0,
    ],
    trans_step_size=0.1,
    rot_step_size=10,
    surface_dist_thre=slam["surface_dist_thre"],
    topk_cls_confidence=[8, 3],
    explore_thre=0.005,
    recognize_thre=0.3,
    color_ig_thre=34,
    depth_ig_thre=0.01,
    post_refinement_eval_freq=100,
    up_dir=np.array([0, 0, 1]),
    use_traj_pose=True,
    SLAMData_dir=os.path.join(dirs["data_dir"], "Replica", general["scene"]),
    local_planner_method="RRTNaruto",
)

if planner["local_planner_method"] == "RRTNaruto":
    planner.update(
        rrt_step_size=planner["trans_step_size"] / slam["bbox_voxel_size"],
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
    method="active_lang",
    vis_rgbd=True,
    vis_rgbd_max_depth=10,
    save_rgbd=False,
    save_rendered_rgbd=False,
    save_pose=False,
    save_state=False,
    save_planning_path=False,
    save_lookat_tgts=False,
    save_information_gain=False,
)
