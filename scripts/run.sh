export CUDA_VISIBLE_DEVICES=0

ns-train sa3d --data {data-dir} \
    --load-dir {ckpt-dir} \
    --pipeline.text_prompt {text-prompt} \
    --pipeline.network.num_prompts {num-prompts} \
    # --pipeline.network.neg_lamda 0.5 \
    # --pipeline.model.mask_fields.mask_threshold 0.1
    # --pipeline.model.mask_fields.base_res 128 \
    # --pipeline.model.mask_fields.num_levels 2 \
    # --pipeline.model.mask_fields.max_res 256 \
    # --pipeline.model.mask_fields.log2_hashmap_size 24
