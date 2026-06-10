
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
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  --multi_gpu --num_processes 4 \
  src/musubi_tuner/qwen_image_train_network.py \
  --config_file _train/exp/with_vae/config

# B) без VAE-латентов control (только VLM, vlm_only_edit=true)
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  --multi_gpu --num_processes 4 \
  src/musubi_tuner/qwen_image_train_network.py \
  --config_file _train/exp/without_vae/config