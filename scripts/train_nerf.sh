export CUDA_VISIBLE_DEVICES=0

python run.py --config=configs/llff/fern.py --stop_at=20000 --render_video --i_weights=10000

# python run.py --config=configs/nerf_unbounded/garden.py --stop_at=40000 --render_video --i_weights=20000

# python run.py --config=configs/lerf/figurines.py --stop_at=40000 --render_video --i_weights=20000