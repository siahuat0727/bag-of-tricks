flag="--gpu --epochs 400 --schedule 133 266"
prefix="epoch400_"

export CUDA_VISIBLE_DEVICES="6"
python main.py $flag &> ${prefix}default.out &

export CUDA_VISIBLE_DEVICES="1"
python main.py $flag --cosine &> ${prefix}cosine.out &

export CUDA_VISIBLE_DEVICES="2"
python main.py $flag --mixup 0.3 &> ${prefix}mixup.out &

export CUDA_VISIBLE_DEVICES="3"
python main.py $flag --smooth 0.1 &> ${prefix}smooth.out &

export CUDA_VISIBLE_DEVICES="4"
python main.py $flag --warmup 10 &> ${prefix}warmup.out &

export CUDA_VISIBLE_DEVICES="5"
python main.py $flag --warmup 10 --smooth 0.1 --mixup 0.3 --cosine &> ${prefix}alltricks.out &
