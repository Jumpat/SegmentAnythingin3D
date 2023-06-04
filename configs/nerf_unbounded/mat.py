_base_ = './nerf_unbounded_default.py'

expname = 'dcvgo_mat_unbounded'

data = dict(
    datadir='./data/360_v2/mat',
    factor=2, # 1297x840
    movie_render_kwargs=dict(
        shift_x=0.0,  # positive right
        #shift_y=-0.0, # negative down
        shift_y=-0.10, # negative down
        shift_z=0.0,
        scale_r=1.0,
        pitch_deg=-40,
    ),
)
