#!/bin/bash
# Run all experiments sequentially in separate processes for GPU memory isolation.
set -e

EXPERIMENTS="baseline_r8 uniform_r16 blockwise_no_ccd blockwise_ccd"
GPU=${1:-7}

echo "Running experiments on GPU $GPU"
echo "Experiments: $EXPERIMENTS"
echo "---"

for exp in $EXPERIMENTS; do
    echo ""
    echo "=========================================="
    echo "Starting experiment: $exp"
    echo "=========================================="
    PYTHONPATH="" CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python run_single_experiment.py $exp 2>&1
    echo "Experiment $exp done (exit code: $?)"
    sleep 2
done

echo ""
echo "All experiments complete!"
echo "Results:"
for exp in uniform_r4 $EXPERIMENTS; do
    if [ -f "outputs/experiments/$exp/results.json" ]; then
        echo "  $exp: $(python3 -c "import json; r=json.load(open('outputs/experiments/$exp/results.json')); m=r['metrics']; print(f'DINO={m.get(\"dino_score\",0):.4f} CLIP-T={m.get(\"clip_t_score\",0):.4f} B-norm={r.get(\"lora_b_norm\",0):.4f}')")"
    else
        echo "  $exp: NO RESULTS"
    fi
done
