if [ $# != 1 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "cd Tasnet/scripts"
  echo "Usage: bash run_standalone_train.sh [DEVICE_ID] "
  echo "bash run_standalone_train.sh 1 "
  echo "Using absolute path is recommended"
  echo "==========================================================================="
fi

export DEVICE_ID=$1
export RANK_ID=0
export RANK_SIZE=1
export SLOG_PRINT_TO_STDOUT=0


rm -rf ./train_single_tasnet
mkdir ./train_single_tasnet
cp -r ../*.py ./train_single_tasnet
cd ./train_single_tasnet || exit
python train.py --device_id=$DEVICE_ID  > train.log 2>&1 &
