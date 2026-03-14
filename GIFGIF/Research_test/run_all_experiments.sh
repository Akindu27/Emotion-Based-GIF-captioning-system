#!/bin/bash

# =============================================================================
# Run All Ablation Experiments
# =============================================================================
#
# This script runs all 4 ablation experiments sequentially.
# Each experiment takes ~6-8 hours, so total runtime is ~24-32 hours.
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
#
# Or run experiments individually:
#   python train_ablation.py --experiment exp3_6groups --gif_dir /path/to/gifs --data_dir ablation_csvs
#
# =============================================================================

# ----- CONFIGURATION (UPDATE THESE!) -----
GIF_DIR="/path/to/your/gifs"  # UPDATE THIS!
DATA_DIR="ablation_csvs"
OUTPUT_DIR="ablation_results"
EPOCHS=20
BATCH_SIZE=32
LEARNING_RATE=0.0001

# -----------------------------------------

echo "============================================================================="
echo "🚀 RUNNING ALL ABLATION EXPERIMENTS"
echo "============================================================================="
echo "GIF Directory: $GIF_DIR"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Epochs per experiment: $EPOCHS"
echo ""
echo "⏰ Estimated total time: 24-32 hours"
echo "   (You can run experiments in parallel on multiple GPUs if available)"
echo "============================================================================="
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local exp_number=$2
    
    echo ""
    echo "============================================================================="
    echo "🔬 EXPERIMENT $exp_number: $exp_name"
    echo "============================================================================="
    echo "Start time: $(date)"
    echo ""
    
    python train_ablation.py \
        --experiment $exp_name \
        --gif_dir $GIF_DIR \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Experiment $exp_number complete!"
        echo "End time: $(date)"
    else
        echo ""
        echo "❌ Experiment $exp_number failed!"
        echo "Check logs above for errors."
        exit 1
    fi
}

# ----- RUN EXPERIMENTS -----

# Note: You can comment out experiments you don't want to run
# Minimum requirement: Run Experiment 3 (your main contribution)

# Experiment 1: 17 emotions (if CSVs exist)
if [ -f "$DATA_DIR/train_17emotions.csv" ]; then
    run_experiment "exp1_17emotions" "1"
else
    echo "⚠️  Skipping Experiment 1 (17 emotions CSVs not found)"
fi

# Experiment 2: 11 emotions (if CSVs exist)
if [ -f "$DATA_DIR/train_11emotions.csv" ]; then
    run_experiment "exp2_11emotions" "2"
else
    echo "⚠️  Skipping Experiment 2 (11 emotions CSVs not found)"
fi

# Experiment 3: 6 hierarchical groups (YOUR MAIN CONTRIBUTION)
if [ -f "$DATA_DIR/train_6groups.csv" ]; then
    run_experiment "exp3_6groups" "3"
else
    echo "❌ ERROR: Experiment 3 CSVs not found!"
    echo "   This is your main experiment - it's required!"
    exit 1
fi

# Experiment 4: 3 simple groups
if [ -f "$DATA_DIR/train_3groups.csv" ]; then
    run_experiment "exp4_3groups" "4"
else
    echo "⚠️  Skipping Experiment 4 (3 groups CSVs not found)"
fi

# ----- GENERATE COMPARISON TABLE -----

echo ""
echo "============================================================================="
echo "📊 GENERATING COMPARISON TABLE"
echo "============================================================================="

python generate_comparison.py --results_dir $OUTPUT_DIR

echo ""
echo "============================================================================="
echo "✅ ALL EXPERIMENTS COMPLETE!"
echo "============================================================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review results in: $OUTPUT_DIR/comparison_table.csv"
echo "2. Check confusion matrices and training curves"
echo "3. Use results in your research paper"
echo ""
echo "Finished at: $(date)"
echo "============================================================================="
