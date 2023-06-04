_base_ = './llff_default.py'

expname = '360_dvgo_kitchen_unbounded'

data = dict(
    datadir='./data/360_v2/kitchen',
    factor=4, # 1558x1039
    movie_render_kwargs=dict(
        shift_y=-0.0,
        scale_r=0.9,
        pitch_deg=-40,
    ),
)
