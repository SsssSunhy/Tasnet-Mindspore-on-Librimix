if [ $# != 2 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "cd Tasnet/scripts"
  echo "Usage: bash run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE] "
  echo "bash run_distribute_train.sh 8 ./hccl_8p.json "
  echo "Using absolute path is recommended"
  echo "==========================================================================="
fi

export RANK_TABLE_FILE=$2
export RANK_START_ID=0
export RANK_SIZE=$1
echo "train Tasnet with distribute"

for((i=0;i<$1;i++))
do
        export DEVICE_ID=$((i + RANK_START_ID))
        export RANK_ID=$i
        echo "start training for rank $i, device $DEVICE_ID"
        env > env.log

        rm -rf ./train_parallel$i
        mkdir ./train_parallel$i
        cp -r ../*.py ./train_parallel$i
        cd ./train_parallel$i || exit
        python train.py  --device_id=$DEVICE_ID > paralletrain.log 2>&1 &
        cd ..
done
