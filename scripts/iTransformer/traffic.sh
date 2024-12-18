model_name=CycleiTransformer

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom


seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2024
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
      --d_model 512 \
      --d_ff 512 \
      --e_layers 4 \
      --dropout 0.1 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 16 --learning_rate 0.001 --random_seed $random_seed
done
done
