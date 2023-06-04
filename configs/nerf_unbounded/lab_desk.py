_base_ = './nerf_unbounded_default.py'

expname = 'lab_desk'

data = dict(
    datadir='./data/nerf_llff_data/lab_desk',
    factor=2,
)
