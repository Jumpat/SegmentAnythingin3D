_base_ = './nerf_unbounded_default.py'

expname = 'dcvgo_legohouse_unbounded'

data = dict(
    datadir='./data/360_v2/legohouse',
    factor=8,
    movie_render_kwargs=dict(
        shift_y=-0.0,
        scale_r=0.9,
        pitch_deg=-40,
    ),
)
