python resnet_main.py \
--train_steps=4720592 \
--train_batch_size=10000 \
--eval_batch_size=10000 \
--num_train_images=4720592 \
--num_eval_images=127306 \
--steps_per_eval=100 \
--iterations_per_loop=100 \
--resnet_depth=50 \
--use_tpu=Ture \
--data_dir=${STORAGE_BUCKET}/data \
--model_dir=${STORAGE_BUCKET}/resnet \
--tpu=${TPU_NAME} \
--precision="bfloat16" \
--data_format="channels_last"
pause
