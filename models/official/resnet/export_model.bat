python resnet_main.py ^
--mode=save_model ^
--train_steps=36400 ^
--train_batch_size=100 ^
--eval_batch_size=100 ^
--num_train_images=2000 ^
--num_eval_images=2000 ^
--steps_per_eval=100 ^
--iterations_per_loop=100 ^
--resnet_depth=201 ^
--use_tpu=False ^
--data_dir="D:\TPU\data" ^
--prices_dir="D:\TPU\data" ^
--predict_dir="D:\TPU\data" ^
--model_dir="D:\TPU\model\turtlex" ^
--export_dir="D:\TPU\export\saved_model_turtlex" ^
--tpu="" ^
--precision="float32" ^
--data_format="channels_last"
pause
