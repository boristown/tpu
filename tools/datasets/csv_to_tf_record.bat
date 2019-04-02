python -m csv_to_tf_record.py ^
       --train_csv="G:\TPU\data\train_set.csv" ^
       --validation_csv="G:\TPU\data\eval_set.csv" ^
       --labels_file="G:\TPU\data\labels.txt" ^
       --project_id="local" ^
       --output_dir="G:\TPU\data\output\"
pause