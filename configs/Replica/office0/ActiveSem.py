import numpy as np
import os

_base_ = "../../default.py"

# 这是 Replica/office0 的推荐“完整 ActiveSem”配置：
# Habitat 仿真器 + SemSplaTAM 建图 + ActiveGSv2 规划器。
##################################################
### General
##################################################
general = dict(
    dataset = "Replica",
    scene = "office0",
    num_iter = 2000,
    device = 'cuda'
)

##################################################
### Directories
##################################################
dirs = dict(
    data_dir = "data/",
    result_dir = "results/",
    cfg_dir = os.path.join("configs", general['dataset'], general['scene'])
)


##################################################
### Simulator
##################################################
sim = dict(
    # 这里使用 habitat_v2：主循环期望 simulator.simulate()
    # 返回字典格式输出，例如 {"color": ..., "depth": ...}。
    method = "habitat_v2"                                  # simulator method
)

if sim["method"] == "habitat_v2":
    sim.update(
        habitat_cfg = os.path.join(dirs['cfg_dir'], "habitat.py")
    )

##################################################
### SLAM
##################################################
slam = dict(
    method="semsplatam"                                     # SLAM backbone method
)

if slam["method"] == "semsplatam":
    slam.update(
        # SplaTAM 的基础配置入口；本文件会覆盖主动建图与语义相关参数。
        room_cfg        = f"{dirs['cfg_dir']}/../replica_splatam_s.py",   # SplaTAM room configuration
        # room_cfg        = f"{dirs['cfg_dir']}/../replica_splatam.py",   # SplaTAM room configuration
        enable_active_planning = True,                             # enable/disable active planning
        dataset_eval_basedir = "data/Replica",
        # dataset_eval_basedir="data/Replica",

        ### 探索地图与规划采样空间的边界框 ###
        # bbox_bound = [[-2.2,2.6],[-3.4,2.1],[-1.4,2.0]],
        bbox_bound = [[-2.1,2.5],[-3.2,2],[-1.3,2.0]],
        bbox_voxel_size = 0.05,

        surface_dist_thre=0.5,
        find_free_indices_bs=256,
        find_free_indices_occ_bs=1024,

        ### Refinement step ###
        # 在 post-refinement 阶段，主循环可能把 mapping 迭代数切到 refine_map_iter。
        # explore_map_iter = 1,
        refine_map_iter = 60,
        use_global_keyframe = True,
        global_keyframe = dict(
            completeness_thre = 0.1,
            color_thre = 34, # smaller than this thre, add to global keyframe
            depth_thre = 0.01, # larger than this thre, add to global keyframe [NOT USED]
            seman_thre = 0.9, # smaller than this thre, add to global keyframe
            quality_method = "relative", # absolute: abs color_thre; relative: percentile
            quality_freq = 100, # eval every quality_freq
            quality_perc_thre = 30, # frames lower than this percentile are added to global KF
        ),

        ##### Semantic Network #######
        # top-k logits 用于控制语义渲染时保留的稀疏语义通道数。
        num_topk_logits = 16,
        num_semantic_classes = 102,
        lambda_hel=0.8,
        lambda_cosine=0.2,
        uncert_mask_thres=3.0,

        semantic_dir= "./data/replica_v1/office_0/habitat/",
        class_info_file='./configs/Replica/office0/class_info_file.json',
        semantic_device="cuda:0",
        oneformer_checkpoint='lly00412/oneformer-replica-finetune',
        coco_checkpoint='shi-labs/oneformer_coco_swin_large',
        ade20k_checkpoint="shi-labs/oneformer_ade20k_swin_large",

        ### override ###
        override = dict(
            map_every = 5,
            report_global_progress_every = 5,
            tracking = dict(
                use_gt_poses=True, # Use GT Poses for Tracking
            ),
            data = dict(
                desired_image_height=340,
                desired_image_width=600,
                tracking_image_height=340,
                tracking_image_width=600,
                densification_image_height=340,
                densification_image_width=600,
            )
        )
    )

##################################################
### Planner
##################################################
planner = dict(
    method= "active_gsv2",                           # planner method [predefined_traj, active_gs]
    # method = "predefined_traj",

    ### active_gs params ###
    # 两阶段探索：先粗粒度采样候选位姿，再细粒度加密采样。
    # gs_z_levels = [20, 30, 40, 50], #[20,30,40],
    max_exploration_steps = 1500,
    post_refine_steps = 200,
    max_refinement_steps = 200,
    num_exploration_stage = 2,
    gs_z_levels = [
        [35], 
        [20, 50],
        # [20, 30, 40, 50]
    ],
    num_dir_samples = [ # viewing direction sample number
        5, 
        15,
    ],

    xy_sampling_step = [
        1.0,
        0.5,
    ], # Unit: meter

    trans_step_size = 0.1, # meter
    rot_step_size = 10, # degree

    surface_dist_thre = slam['surface_dist_thre'],
        topk_cls_confidence = [8, 3],

    ### Stop Criteria ###
    # 当地图增益低于阈值时，规划器会从 exploration 切换到 refinement/done。
    explore_thre = 0.005,
    recognize_thre = 0.3,
    color_ig_thre = 34,
    depth_ig_thre = 0.01,
    post_refinement_eval_freq = 100,


    up_dir = np.array([0, 0, 1]), # up direction for planning pose
    use_traj_pose = True,                          # use pre-defined trajectory pose
    SLAMData_dir = os.path.join(                    # SLAM Data directory (for passive mapping or pre-defined trajectory pose)
        dirs["data_dir"], 
        "Replica", general['scene']
        ),

    ### RRT ###
    local_planner_method = "RRTNaruto",             # RRT method
)

if planner["local_planner_method"] == "RRTNaruto":
    planner.update(
        rrt_step_size = planner['trans_step_size'] / slam['bbox_voxel_size'], # Unit: voxel
        rrt_step_amplifier = 10,                    # rrt step amplifier to fast expansion
        rrt_maxz = 100,                             # Maximum Z-level to limit the RRT nodes. Unit: voxel
        rrt_max_iter = None,                        # maximum iterations for RRT
        rrt_z_levels = None,                        # Z levels for sampling RRT nodes. Unit: voxel. Min and Max level
        enable_eval = False,                        # enable RRT evaluation
        enable_direct_line = True,                  # enable direct connection attempt
    )

##################################################
### Visualization
##################################################
visualizer = dict(
    method = "active_lang",
    vis_rgbd        = True,                             # visualize RGB-D
    vis_rgbd_max_depth = 10,

    ### visualization save flags ###
    save_rgbd = False,                                  # save GT RGB-D
    save_rendered_rgbd = False,                         # save rendered RGB-D
    save_pose = False,                                  # save camera poses
    save_state = False,                                 # save planner state
    save_planning_path = False,                         # save planning path
    save_lookat_tgts = False,                           # save lookat targets
    save_information_gain = False,                      # save information gain

    ### mesh related ###
    # mesh_vis_freq = 500,                                # mesh save frequency
)

