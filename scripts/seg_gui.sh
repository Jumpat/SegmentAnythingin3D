export CUDA_VISIBLE_DEVICES=3

python run_seg_gui.py --config=configs/llff/seg/seg_fern.py --segment \
    --sp_name=_gui --num_prompts=20 \
    --render_opt=train 

# python run_seg_gui.py --config=configs/llff/seg/seg_fern.py --segment \
#     --sp_name=_gui --num_prompts=20 \
#     --render_opt=train \
#     --mobile_sam

# python run_seg_gui.py --config=configs/nerf_unbounded/seg_garden.py --segment \
#     --sp_name=_gui --num_prompts=10 \
#     --render_opt=video

# python run_seg_gui.py --config=configs/lerf/seg_lerf/figurines.py --segment \
#     --sp_name=_gui --num_prompts=10 \
#     --render_opt=video
