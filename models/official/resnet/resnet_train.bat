python resnet_main.py ^
--train_steps=120 ^
--train_batch_size=6 ^
--eval_batch_size=6 ^
--num_train_images=6 ^
--num_eval_images=6 ^
--steps_per_eval=1 ^
--iterations_per_loop=1 ^
--resnet_depth=50 ^
--use_tpu=False ^
--data_dir="G:\TPU\data" ^
--model_dir="G:\TPU\model" ^
--tpu="" ^
--precision="float32" ^
--data_format="channels_last"
pause
