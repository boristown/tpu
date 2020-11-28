python resnet_main.py ^
--mode=save_model ^
--train_steps=36400 ^
--train_batch_size=100 ^
--eval_batch_size=100 ^
--num_train_images=2000 ^
--num_eval_images=2000 ^
--steps_per_eval=100 ^
--iterations_per_loop=100 ^
--resnet_depth=169 ^
--use_tpu=False ^
--data_dir="D:\TPU\data" ^
--prices_dir="D:\Robot\WX_BG\Output\prices" ^
--predict_dir="D:\Robot\WX_BG\Output\predict" ^
--model_dir="D:\TPU\model" ^
--export_dir="D:/saved_model/turtle6" ^
--tpu="" ^
--precision="float32" ^
--data_format="channels_last"
pause
