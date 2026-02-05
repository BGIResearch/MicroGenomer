#!/bin/bash
set -euo pipefail
PROJECT_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# echo "Working directory: $(pwd)"
echo "Project working directory: $PROJECT_HOME"

# ===================== 1. Define command line argument parsing function (with renaming + new args) =====================
parse_args() {
    input_path="./res/demo.csv"
    output_dir="./results"
    task="extract_embed"
    batch_size=1
    max_seq_len=2000
    model_path="./models/HuggingFace"
    ckpt_path="./weights/MicroGenomer_470M/model_states.pt"
    output_attentions="False"
    output_hidden_states="True"
    extra_args=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --input_path)
                input_path="$2"
                shift 2
                ;;
            --output_dir)
                output_dir="$2"
                shift 2
                ;;
            --task)
                task="$2"
                shift 2
                ;;
            --batch_size)
                batch_size="$2"
                shift 2
                ;;
            --max_seq_len)
                max_seq_len="$2"
                shift 2
                ;;
            --model_path)
                model_path="$2"
                shift 2
                ;;
            --ckpt_path)
                ckpt_path="$2"
                shift 2
                ;;
            --output_attentions)
                output_attentions="$2"
                shift 2
                ;;
            --output_hidden_states)
                output_hidden_states="$2"
                shift 2
                ;;
            *)
                extra_args+="$1 "
                shift 1
                ;;
        esac
    done

    # Trim leading and trailing spaces of extra arguments
    extra_args=$(echo "$extra_args" | xargs)
}

# ===================== 2. Define script usage instructions =====================
usage() {
    echo "======================================= Script Usage Instructions ======================================="
    echo "Basic Usage: $0 --input_path <input path> --output_dir <output directory> --task <task type> [other arguments...]"
    echo ""
    echo "[Embedding Extraction Task]"
    echo "  --task extract_embed    - DNA feature/embedding extraction ｜ Example: --task extract_embed"
    echo ""
    echo "[Prediction Tasks]"
    echo "  --task test_growth      - Growth rate prediction ｜ Example: --task test_growth"
    echo "  --task test_ph          - Optimal pH prediction    ｜ Example: --task test_ph"
    echo "  --task test_salinity    - Salt tolerance prediction    ｜ Example: --task test_salinity"
    echo "  --task test_oxygen      - Oxygen tolerance prediction  ｜ Example: --task test_oxygen"
    echo "  --task test_temperature - Optimal temperature prediction  ｜ Example: --task test_temperature"
    echo "  --task test_probiotic   - Probiotic authenticity prediction ｜ Example: --task test_probiotic"
    echo "======================================================================================================"
}

# ===================== 3. Define core CSV processing function =====================
process_single_csv() {
    local input_csv="$1"
    local output_dir="$2"
    local batch_size="$3"
    local max_seq_len="$4"
    local model_path="$5"
    local ckpt_path="$6"
    local output_attentions="$7"
    local output_hidden_states="$8"
    
    echo -e "========================================"
    echo "Start processing CSV file: $input_csv"
    echo -e "========================================"
    
    mkdir -p "$output_dir"
    echo "Output directory created: $output_dir"
    
    cmd="python models/dnaglm.py \
--csv_path \"$input_csv\" \
--max_seq_len \"$max_seq_len\" \
--model_path \"$model_path\" \
--ckpt_path \"$ckpt_path\" \
--save_dir \"$output_dir\" \
--batch_size \"$batch_size\" \
--output_attentions \"$output_attentions\" \
--output_hidden_states \"$output_hidden_states\""

    eval "$cmd"
}

process_multiple_csv() {
    local input_csv="$1"
    local output_dir="$2"
    local batch_size="$3"
    local max_seq_len="$4"
    local model_path="$5"
    local ckpt_path="$6"
    local output_attentions="$7"
    local output_hidden_states="$8"

    # Single file processing mode
    if [ -f "$input_path" ]; then
        echo "Single CSV file input mode detected, starting feature extraction..."
        process_single_csv "$input_path" "$output_dir" "$batch_size" "$max_seq_len" "$model_path" "$ckpt_path" "$output_attentions" "$output_hidden_states"

    # Directory batch processing mode
    elif [ -d "$input_path" ]; then
        echo "Directory batch input mode detected, starting feature extraction..."
        csv_count=$(find "$input_path" -maxdepth 1 -name "*.csv" -type f | wc -l)
        echo "Found $csv_count CSV files in $input_path"

        if [ $csv_count -eq 0 ]; then
            echo "ERROR: No .csv files found in directory $input_path!"
            exit 1
        fi

        for csv_file in "$input_path"/*.csv; do
            if [ -f "$csv_file" ]; then
                process_single_csv "$input_path" "$output_dir" "$batch_size" "$max_seq_len" "$model_path" "$ckpt_path" "$output_attentions" "$output_hidden_states"
            fi
        done
    fi
    python scripts/proess_data_to_csv.py --input_dir=$output_dir
}

# ===================== 4. Main Process: Arg Parsing → Data Preprocessing → Task Distribution → Core Processing =====================
# Step 1: Parse command line arguments
parse_args "$@"

# Step 2: Execute feature extraction
process_multiple_csv "$input_path" "$output_dir" "$batch_size" "$max_seq_len" "$model_path" "$ckpt_path" "$output_attentions" "$output_hidden_states"

# Step 3: Distribute tasks according to --task parameter
case "$task" in
    ""|extract_embed)
        # Feature extraction completed
        ;;
    test_growth)
        python scripts/test_growth_rate.py --output_dir=$output_dir $extra_args
        ;;
    test_ph)
        python scripts/test_pH.py --output_dir=$output_dir $extra_args
        ;;
    test_salinity)
        python scripts/test_salinity.py --output_dir=$output_dir $extra_args
        ;;
    test_oxygen)
        python scripts/test_oxygen.py --output_dir=$output_dir $extra_args
        ;;
    test_temperature)
        python scripts/test_temperature.py --output_dir=$output_dir $extra_args
        ;;
    test_probiotic)
        python scripts/test_probiotic.py --output_dir=$output_dir $extra_args
        ;;
    *)
        echo "ERROR: Unsupported task type! --task $task"
        usage
        exit 1
        ;;
esac

if [ $? -ne 0 ]; then
    echo "ERROR: Task --task $task execution failed!"
    exit 1
fi

echo -e "========================================"
echo "✅ Task $task executed successfully!"
echo "========================================"