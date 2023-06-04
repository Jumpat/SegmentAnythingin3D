_base_ = './nerf_unbounded_default.py'

expname = 'dcvgo_garden_unbounded'

data = dict(
    datadir='./data/360_v2/garden',
    factor=8,
    movie_render_kwargs=dict(
        shift_x=0.0,  # positive right
        shift_y=-0.0, # negative down
        shift_z=0,
        scale_r=0.9,
        pitch_deg=-30,
    ),
)
