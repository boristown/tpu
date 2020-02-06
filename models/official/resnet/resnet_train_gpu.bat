python resnet_main.py ^
--train_steps=34172  ^
--train_batch_size=16  ^
--eval_batch_size=16  ^
--num_train_images=546767  ^
--num_eval_images=2261  ^
--steps_per_eval=500  ^
--iterations_per_loop=500  ^
--dropblock_groups=""  ^
--dropblock_keep_prob="0.5"  ^
--dropblock_size="3"  ^
--resnet_depth=169  ^
--use_tpu=False ^
--data_dir="G:\TPU\data" ^
--model_dir="G:\TPU\model" ^
--export_dir="G:\TPU\export" ^
--tpu="" ^
--precision="float32" ^
--data_format="channels_last"
pause
