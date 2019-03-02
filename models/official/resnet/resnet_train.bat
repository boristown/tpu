python resnet_main.py ^
--train_steps=7000 ^
--train_batch_size=2 ^
--eval_batch_size=2 ^
--num_train_images=4720592 ^
--num_eval_images=127306 ^
--steps_per_eval=5 ^
--iterations_per_loop=5 ^
--resnet_depth=50 ^
--use_tpu=False ^
--data_dir="C:\TPU\data" ^
--model_dir="C:\TPU\model" ^
--export_dir="C:\TPU\export" ^
--tpu="" ^
--precision="float32" ^
--data_format="channels_last"
pause
