python resnet_main.py ^
--train_steps=7000 ^
--train_batch_size=1101 ^
--eval_batch_size=1101 ^
--num_train_images=2203 ^
--num_eval_images=2203 ^
--steps_per_eval=50 ^
--iterations_per_loop=50 ^
--resnet_depth=50 ^
--use_tpu=False ^
--data_dir="G:\TPU\data" ^
--model_dir="G:\TPU\model" ^
--export_dir="G:\TPU\export" ^
--tpu="" ^
--precision="float32" ^
--data_format="channels_last"
pause
