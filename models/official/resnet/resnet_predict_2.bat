python resnet_main.py ^
--mode=predict ^
--train_steps=7000 ^
--train_batch_size=100 ^
--eval_batch_size=100 ^
--num_train_images=529 ^
--num_eval_images=529 ^
--steps_per_eval=100 ^
--iterations_per_loop=100 ^
--resnet_depth=50 ^
--use_tpu=False ^
--data_dir="G:\TPU\data" ^
--prices_dir="G:\TPU\prices" ^
--predict_dir="G:\TPU\predict" ^
--model_dir="G:\TPU\model" ^
--export_dir="G:\TPU\export" ^
--tpu="" ^
--precision="float32" ^
--data_format="channels_last"
pause
