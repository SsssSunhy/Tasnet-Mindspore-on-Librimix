if [ $# != 1 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "cd Tasnet/scripts"
  echo "Usage: bash run_eval.sh [DEVICE_ID] "
  echo "bash run_eval.sh 1 "
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export DEVICE_ID=$1
export RANK_SIZE=1

rm -rf ./eval
mkdir ./eval
cp -r ../*.py ./eval

env > env.log
cd ./eval || exit
python eval.py  --device_id=$1  > eval.log 2>&1 &