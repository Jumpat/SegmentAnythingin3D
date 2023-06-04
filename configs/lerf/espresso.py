_base_ = './lerf_default.py'

expname = 'dcvgo_espresso'

data = dict(
    datadir='./data/lerf_data/espresso',
    factor=2, # 497 * 369
    # factor=4,
#     movie_render_kwargs=dict(
#         shift_x=0.0,  # positive right
#         shift_y=-0.3, # negative down
#         shift_z=0,
#         scale_r=0.2,
#         pitch_deg=-40, # negative look downward
#     ),
)