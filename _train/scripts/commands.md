
# проверка папок на битые изображения (запускать ПЕРЕД созданием jsonl)
# dry-run: только показать список битых, ничего не удаляя
python _train/scripts/clean_broken_images.py \
  /mnt/images/a.gainetdionov/Datasets/es_hunt_only/ref_person \
  /mnt/images/a.gainetdionov/Datasets/es_hunt_only/img_normalized_for_qwen_image \
  /mnt/images/a.gainetdionov/Datasets/sft_dataset/babesource/ref_person \
  /mnt/images/a.gainetdionov/Datasets/sft_dataset/babesource/img_normalized_for_qwen_image

# удаление битых (--delete), после этого перегенерировать jsonl ниже
python _train/scripts/clean_broken_images.py \
  /mnt/images/a.gainetdionov/Datasets/es_hunt_only/ref_person \
  /mnt/images/a.gainetdionov/Datasets/es_hunt_only/img_normalized_for_qwen_image \
  /mnt/images/a.gainetdionov/Datasets/sft_dataset/babesource/ref_person \
  /mnt/images/a.gainetdionov/Datasets/sft_dataset/babesource/img_normalized_for_qwen_image \
  --delete


# создание jsonl для es_hunt_only
python _train/scripts/build_jsonl.py \
  --image-dir   /mnt/images/a.gainetdionov/Datasets/es_hunt_only/img_normalized_for_qwen_image \
  --control-dir /mnt/images/a.gainetdionov/Datasets/es_hunt_only/ref_person \
  --caption-dir /mnt/images/a.gainetdionov/Datasets/es_hunt_only/caption_grok_fused_json \
  --output-dir  /mnt/images/a.gainetdionov/Datasets/es_hunt_only \
  --output-name es_hunt_qwen_image_reference.jsonl


# создание jsonl для babesource (имена вида 0000000.raw.png -> ключ 0000000; тексты в .json)
python _train/scripts/build_jsonl.py \
  --image-dir   /mnt/images/a.gainetdionov/Datasets/sft_dataset/babesource/img_normalized_for_qwen_image \
  --control-dir /mnt/images/a.gainetdionov/Datasets/sft_dataset/babesource/ref_person \
  --caption-dir /mnt/images/a.gainetdionov/Datasets/sft_dataset/babesource/jsons_grok4 \
  --caption-ext .json \
  --output-dir  /mnt/images/a.gainetdionov/Datasets/sft_dataset/babesource \
  --output-name babesource_qwen_image_reference.jsonl

# Латенты изображений (VAE)
python src/musubi_tuner/qwen_image_cache_latents.py \
  --dataset_config _train/dataset_configs/dataset_config.toml \
  --vae _train/models/qwen_image_vae.safetensors \
  --model_version edit-2509
  --skip_existing

# Выходы текстового энкодера (Qwen2.5-VL)
python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
  --dataset_config _train/dataset_configs/dataset_config.toml \
  --text_encoder _train/models/qwen_2.5_vl_7b.safetensors \
  --batch_size 1 \
  --model_version edit-2509
  --skip_existing


# === ОБУЧЕНИЕ (два эксперимента, A/B; останавливать вручную) ===

# A) с VAE-латентами control (стандартный edit)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  --multi_gpu --num_processes 4 --main_process_port 29500 \
  src/musubi_tuner/qwen_image_train_network.py --config_file _train/exp/with_vae/config

# B) без VAE-латентов control (только VLM, vlm_only_edit=true)
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  --multi_gpu --num_processes 4 --main_process_port 29501 \
  src/musubi_tuner/qwen_image_train_network.py --config_file _train/exp/without_vae/config


# === ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ (resume) ===
# Требует save_state=true в конфиге. State-папки пишутся в output_dir как
# {output_name}-step{шаг:08d}-state. Подставь номер последнего сохранённого шага
# (см. ls _train/exp/with_vae/output | grep state).

# A) resume
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  --multi_gpu --num_processes 4 --main_process_port 29500 \
  src/musubi_tuner/qwen_image_train_network.py --config_file _train/exp/with_vae/config \
  --resume _train/exp/with_vae/output/qwen_edit_with_vae-step00000350-state

# B) resume
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  --multi_gpu --num_processes 4 --main_process_port 29501 \
  src/musubi_tuner/qwen_image_train_network.py --config_file _train/exp/without_vae/config \
  --resume _train/exp/without_vae/output/qwen_edit_vlm_only-step00000600-state




Подготовка (один раз)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate musubi
cd /home/a.gainetdinov/Github/musubi-tuner
mkdir -p logs
SET=/mnt/images/a.gainetdionov/Datasets/es_hunt_only
mkdir -p "$SET"/cache_qwen_image_reference_{1328,640,320}
4 команды — VAE-латенты, все 3 разрешения (GPU 0–3)

CUDA_VISIBLE_DEVICES=0 python src/musubi_tuner/qwen_image_cache_latents.py \
  --dataset_config _train/dataset_configs/shards/es_hunt_shard_0.toml \
  --vae _train/models/qwen_image_vae.safetensors \
  --model_version edit-2509 --skip_existing --keep_cache \
  > logs/cache_lat_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python src/musubi_tuner/qwen_image_cache_latents.py \
  --dataset_config _train/dataset_configs/shards/es_hunt_shard_1.toml \
  --vae _train/models/qwen_image_vae.safetensors \
  --model_version edit-2509 --skip_existing --keep_cache \
  > logs/cache_lat_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python src/musubi_tuner/qwen_image_cache_latents.py \
  --dataset_config _train/dataset_configs/shards/es_hunt_shard_2.toml \
  --vae _train/models/qwen_image_vae.safetensors \
  --model_version edit-2509 --skip_existing --keep_cache \
  > logs/cache_lat_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python src/musubi_tuner/qwen_image_cache_latents.py \
  --dataset_config _train/dataset_configs/shards/es_hunt_shard_3.toml \
  --vae _train/models/qwen_image_vae.safetensors \
  --model_version edit-2509 --skip_existing --keep_cache \
  > logs/cache_lat_3.log 2>&1 &
4 команды — TE / Qwen2.5‑VL, один раз в папку 1328 (GPU 4–7)
CUDA_VISIBLE_DEVICES=4 python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
  --dataset_config _train/dataset_configs/shards/es_hunt_shard_0_te.toml \
  --text_encoder _train/models/qwen_2.5_vl_7b.safetensors \
  --batch_size 1 --model_version edit-2509 --skip_existing --keep_cache \
  > logs/cache_te_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
  --dataset_config _train/dataset_configs/shards/es_hunt_shard_1_te.toml \
  --text_encoder _train/models/qwen_2.5_vl_7b.safetensors \
  --batch_size 1 --model_version edit-2509 --skip_existing --keep_cache \
  > logs/cache_te_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
  --dataset_config _train/dataset_configs/shards/es_hunt_shard_2_te.toml \
  --text_encoder _train/models/qwen_2.5_vl_7b.safetensors \
  --batch_size 1 --model_version edit-2509 --skip_existing --keep_cache \
  > logs/cache_te_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
  --dataset_config _train/dataset_configs/shards/es_hunt_shard_3_te.toml \
  --text_encoder _train/models/qwen_2.5_vl_7b.safetensors \
  --batch_size 1 --model_version edit-2509 --skip_existing --keep_cache \
  > logs/cache_te_3.log 2>&1 &


# === ОБУЧЕНИЕ всех 8 экспериментов (0–7) ===
# 1 GPU на эксперимент -> все 8 запускаются параллельно. Логи в logs/train_<N>.log.
# Перед запуском кэш cache_qwen_image_reference_{1328,640,320} должен быть заполнен (см. секцию выше).
# Резюм: добавь к нужной команде --resume _train/exp/<папка>/output/<output_name>-step<NNNNNNNN>-state

# 0) LoKr, lr=5e-5, 1328
CUDA_VISIBLE_DEVICES=0 python src/musubi_tuner/qwen_image_train_network.py \
  --config_file _train/exp/0.without_vae_lokr_lr_5e-5_1328/config \
  > logs/train_0_lokr_lr5e-5_1328.log 2>&1 &

# 1) LoKr, lr=1e-4, 1328
CUDA_VISIBLE_DEVICES=1 python src/musubi_tuner/qwen_image_train_network.py \
  --config_file _train/exp/1.without_vae_lokr_lr_1e-4_1328/config \
  > logs/train_1_lokr_lr1e-4_1328.log 2>&1 &

# 2) LoKr, lr=5e-5, 1328+640
CUDA_VISIBLE_DEVICES=2 python src/musubi_tuner/qwen_image_train_network.py \
  --config_file _train/exp/2.without_vae_lokr_lr_5e-5_1328_640/config \
  > logs/train_2_lokr_lr5e-5_1328_640.log 2>&1 &

# 3) LoKr, lr=5e-5, 1328+640+320
CUDA_VISIBLE_DEVICES=3 python src/musubi_tuner/qwen_image_train_network.py \
  --config_file _train/exp/3.without_vae_lokr_lr_5e-5_1328_640_320/config \
  > logs/train_3_lokr_lr5e-5_1328_640_320.log 2>&1 &

# 4) LoRA, lr=5e-5, 1328
CUDA_VISIBLE_DEVICES=4 python src/musubi_tuner/qwen_image_train_network.py \
  --config_file _train/exp/4.without_vae_lora_lr_5e-5_1328/config \
  > logs/train_4_lora_lr5e-5_1328.log 2>&1 &

# 5) LoHa, lr=1e-4, 1328
CUDA_VISIBLE_DEVICES=5 python src/musubi_tuner/qwen_image_train_network.py \
  --config_file _train/exp/5.without_vae_loha_lr_1e-4_1328/config \
  > logs/train_5_loha_lr1e-4_1328.log 2>&1 &

# 6) LoRA, lr=5e-5, 1328+640
CUDA_VISIBLE_DEVICES=6 python src/musubi_tuner/qwen_image_train_network.py \
  --config_file _train/exp/6.without_vae_lora_lr_5e-5_1328_640/config \
  > logs/train_6_lora_lr5e-5_1328_640.log 2>&1 &

# 7) LoRA, lr=5e-5, 1328+640+320
CUDA_VISIBLE_DEVICES=7 python src/musubi_tuner/qwen_image_train_network.py \
  --config_file _train/exp/7.without_vae_lora_lr_5e-5_1328_640_320/config \
  > logs/train_7_lora_lr5e-5_1328_640_320.log 2>&1 &