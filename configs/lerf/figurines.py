_base_ = './lerf_default.py'

expname = 'dcvgo_figurines'

data = dict(
    datadir='./data/lerf_data/figurines',
    factor=2, # 497 * 369
    movie_render_kwargs=dict(
        shift_x=0.0,  
        shift_y=0.0, 
        shift_z=0.0,
        scale_r=1.0,
        pitch_deg=55,
    ),
)