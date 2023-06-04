_base_ = './lerf_default.py'

expname = 'dcvgo_book_store'

data = dict(
    datadir='./data/lerf_data/book_store',
    factor=2, # 497 * 369
    # factor=4,
    movie_render_kwargs=dict(
        shift_x=0.5,  # positive right
        shift_y=0.5, # negative down
        shift_z=1,
        scale_r=0,
        pitch_deg=0, # negative look downward
    ),
)