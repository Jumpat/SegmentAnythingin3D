import argparse

### configs
def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_opt", default=None, type=str, 
                        choices=['train', 'test', 'video'], help='rendering mode')

    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')

    # arguments for feature distillation
    parser.add_argument("--freeze_density", action='store_true',
                        help='freeze density grid')
    parser.add_argument("--freeze_rgb", action='store_true',
                        help='freeze rgb grid and mlp')
    parser.add_argument("--freeze_feature", action='store_true',
                        help='freeze feature grid')
    parser.add_argument("--only_distill_loss", action='store_true',
                        help='train on only loss of features')
    parser.add_argument("--weighted_distill_loss", action='store_true',
                        help='train on weighted loss')
    parser.add_argument("--seg_mask", action='store_true',
                        help='generate segmentation mask')
    parser.add_argument("--segment", action='store_true',
                        help='interactively set threshold and re-render until stopped.')

    parser.add_argument("--segment_everything", action='store_true',
                        help='if true, adopt SamAutomaticMaskGenerator to generate masks for all possible objects in the first frame.')
    parser.add_argument("--stop_at", type=int, default=1000000,
                        help='at what iteration to stop training.')
    parser.add_argument("--get_cam_trajectory", action='store_true',
                        help='Get the camera poses of the training set as the trajectory.')
    
    # prompt options
    parser.add_argument("--num_prompts", type=int, default=3, help='number of prompts')
    parser.add_argument("--num_epochs", type=int, default=1, help='number of training epochs')
    parser.add_argument("--lamb", type=float, default=1., help='the negative force in seg loss')
    parser.add_argument("--tau", type=float, default=0.5, help='the iou threshold')
    parser.add_argument('--prompt_type', type=str, default='scene', choices=['scene', 'file', 'input', 'interactive', 'text'], 
                        help='the type of prompt, point or box')
    # type 1: scene property
    parser.add_argument("--scene", type=str, default=None,
                        help='scene, used for querying point and box coords')
    # type 2: json file
    parser.add_argument("--prompt_file", type=str, default=None,
                        help='json file containing prompts')
    # type 3: directly input the prompts
    parser.add_argument('--coords', metavar='N', type=int, nargs='+', help='get prompts from command line')
    # type 4: interactive backend
    parser.add_argument("--interactive", action='store_true', help='interactive backend, \
                        input points are from interactive GUI') # TODO
    # type 5: text discription
    parser.add_argument("--text", type=str, default=None,
                        help='text discription of the prompt')
    
    # sp_name for instance
    parser.add_argument("--sp_name", type=str, default=None, help="if None, use default, else use this as e_flag")
    # seg training
    parser.add_argument("--use_fine_stage", action='store_true',
                        help='fine stage can be used when IoU is low')
    parser.add_argument("--seg_poses", default='train', type=str,
                        choices=['train', 'video'], help='which poses are used for segmentation')

    # seg testing
    parser.add_argument('--seg_type', nargs = '+', type=str, default=['seg_img', 'seg_density'],
                        help='segmentation type in inference')
    parser.add_argument("--save_ckpt", action='store_true',
                        help='save segmentation ckpt')
    parser.add_argument("--mobile_sam", action='store_true', help='Replace the original SAM encoder with MobileSAM to accelerate segmentation')
    return parser



