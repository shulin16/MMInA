# example bash script to run the code
conda init
conda activate MMInA

RESULT_DIR="results/"

export PYTHONPATH="./MMInA:$PYTHONPATH"
export MODEL_NAME="cogagent-9b"
export HF_MODEL_ID="THUDM/cogagent-9b-20241220"


python run.py \
    --model $MODEL_NAME \
    --domain shopping \
    --result_dir "$RESULT_DIR"
    # --hist