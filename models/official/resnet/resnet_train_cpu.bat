python resnet_main.py ^
--train_steps=500  ^
--train_batch_size=12  ^
--eval_batch_size=12  ^
--input_batch_size=400  ^
--num_train_images=12  ^
--num_eval_images=12  ^
--steps_per_eval=100  ^
--iterations_per_loop=100  ^
--dropblock_groups=""  ^
--dropblock_keep_prob="0.5"  ^
--dropblock_size="3"  ^
--resnet_depth=169  ^
--use_tpu=False ^
--data_dir="D:\TPU\data" ^
--model_dir="D:\TPU\model" ^
--export_dir="D:\TPU\export" ^
--tpu="" ^
--precision="float32" ^
--data_format="channels_last" ^
--num_label_classes=12
pause
