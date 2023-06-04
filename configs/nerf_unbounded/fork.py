_base_ = './nerf_unbounded_default.py'

expname = 'dcvgo_fork_unbounded'

data = dict(
    datadir='./data/fork/dense',
    factor=8, # 1558x1038
    bd_factor=None,
    movie_render_kwargs=dict(
        shift_x=0.0,  # positive right
        shift_y=0.0, # negative down
        shift_z=0,
        scale_r=0.9,
        pitch_deg=-30, # negative look downward
    ),
)

