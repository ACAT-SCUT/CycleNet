model_name=CycleNet

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom


model_type='mlp'
seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2024 2025 2026 2027 2028
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --cycle 168 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 64 --learning_rate 0.002 --random_seed $random_seed
done
done
