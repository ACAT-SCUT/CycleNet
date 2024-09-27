root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

model_name=CycleNet
model_type='linear'
seq_len=336
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
      --model_type $model_type \
      --enc_in 7 \
      --cycle 24 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed $random_seed
done
done

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

model_name=CycleNet
model_type='linear'
seq_len=336
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
      --model_type $model_type \
      --enc_in 7 \
      --cycle 24 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed $random_seed
done
done

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

model_name=CycleNet
model_type='linear'
seq_len=336
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
      --model_type $model_type \
      --enc_in 7 \
      --cycle 96 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed $random_seed
done
done

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

model_name=CycleNet
model_type='linear'
seq_len=336
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
      --model_type $model_type \
      --enc_in 7 \
      --cycle 96 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed $random_seed
done
done

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

model_name=CycleNet
model_type='linear'
seq_len=336
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
      --model_type $model_type \
      --enc_in 21 \
      --cycle 144 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed
done
done

root_path_name=./dataset/
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar


model_name=CycleNet
model_type='linear'
seq_len=336
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
      --model_type $model_type \
      --enc_in 137 \
      --cycle 144 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed $random_seed
done
done


root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

model_name=CycleNet
model_type='linear'
seq_len=336
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
      --model_type $model_type \
      --enc_in 321 \
      --cycle 168 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed $random_seed
done
done



root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

model_name=CycleNet
model_type='linear'
seq_len=336
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
      --model_type $model_type \
      --enc_in 862 \
      --cycle 168 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --itr 1 --batch_size 256 --learning_rate 0.01 --random_seed $random_seed
done
done
