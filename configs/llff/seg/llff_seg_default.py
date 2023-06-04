_base_ = '../../seg_default.py'

basedir = './logs/llff'

data = dict(
    dataset_type='llff',
    ndc=True,
#    width=1008,
#    height=756,
    factor=4,
)

coarse_train = dict(
    N_iters=0,
)

coarse_model_and_render = dict(
    num_voxels=320**3,
    num_voxels_base=320**3,
    density_type='DenseGrid',
    density_config=dict(n_comp=1),
    k0_type='TensoRFGrid',
    k0_config=dict(n_comp=48),
)

fine_train = dict(
    N_iters=30000,
    #N_iters=60000,
    N_rand=4096 * 1,
    #weight_distortion=0.01,
    pg_scale=[2000,4000,6000,8000],
    ray_sampler='flatten',
    tv_before=1e9,
    tv_dense_before=10000,
    weight_tv_density=1e-5,
    weight_tv_k0=1e-6,
)

fine_model_and_render = dict(
    num_voxels=320**3,
    num_voxels_base=320**3,
    density_type='DenseGrid',
    density_config=dict(n_comp=1),
    k0_type='TensoRFGrid',
    k0_config=dict(n_comp=48),

    mpi_depth=128,
    rgbnet_dim=9,
    rgbnet_width=64,
    world_bound_scale=1,
    fast_color_thres=1e-1,
)
