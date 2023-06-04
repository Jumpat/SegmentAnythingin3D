_base_ = './seg_nerf_unbounded_default.py'

expname = 'dcvgo_pinecone_unbounded'

data = dict(
    datadir='./data/nerf/nerf_real_360/pinecone',
    factor=8, # 484x363
    movie_render_kwargs=dict(
        shift_x=0.0,
        shift_y=0.0,
        shift_z=0.0,
        scale_r=0.9,
        pitch_deg=-40,
    ),
)
