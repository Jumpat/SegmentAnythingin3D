_base_ = './lerf_default.py'

expname = 'dcvgo_donuts'

data = dict(
    datadir='./data/lerf_data/donuts',
    factor=1, # 497 * 369
    # factor=4,
    movie_render_kwargs=dict(
        shift_x=-0.2,  
        shift_y=0.2, 
        shift_z=0.1,
        scale_r=1.3,
        pitch_deg=60,
    ),
)