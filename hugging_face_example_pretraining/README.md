# For Benchmarking ViTMAE

[Example homepage](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining)

You must install the Hugging Face Transformers library from source: 

```
pip install git+https://github.com/huggingface/transformers

```

You can run the model with the following command:

```
python run_mae.py \
    --dataset_name cifar10 \
    --output_dir ./vit-mae-demo \
    --remove_unused_columns False \
    --label_names pixel_values \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --num_train_epochs 800 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --report_to "wandb"

```

[A notebook with a helpful example of visualizing reconstruction](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTMAE/ViT_MAE_visualization_demo.ipynb)

