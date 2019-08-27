python resnet_main.py ^
--mode=predict ^
--train_steps=7000 ^
--train_batch_size=100 ^
--eval_batch_size=100 ^
--num_train_images=100 ^
--num_eval_images=100 ^
--steps_per_eval=100 ^
--iterations_per_loop=100 ^
--resnet_depth=169 ^
--use_tpu=False ^
--data_dir="C:\TPU\data" ^
--prices_dir="C:\TPU\prices" ^
--predict_dir="C:\TPU\predict" ^
--model_dir="C:\TPU\model" ^
--export_dir="C:\TPU\export" ^
--tpu="" ^
--precision="float32" ^
--data_format="channels_last"
pause
