#!/bin/bash
# Agent / critic-priority ablations for FLUX Kontext agentic editing.
# Usage: bash scripts/AR_image_generation_FluxKontext_ablation.sh [name ...]
# With no names, every ablation below runs. Each writes to its own experiment_datetime folder.
# Format: name|ablate_agents|critic_priority
ABLATIONS=(
    "no_text_refiner|text_refiner|default"
    "no_critic|critic|default"
    "prio_none||none"
    "criteria_no_visual||message,sensation"
    "criteria_no_message||visual,sensation"
    "criteria_no_sensation||visual,message"
)
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

for entry in "${ABLATIONS[@]}"; do
    IFS='|' read -r name agents priority <<< "$entry"
    if [ $# -gt 0 ] && [[ ! " $* " =~ " $name " ]]; then
        continue
    fi
    echo "=== Ablation: $name (ablate_agents='$agents', critic_priority='$priority') ==="
    python inference.py --config_type=DEFAULT \
                        --inference_type=image_generation \
                        --text_input_type=AR \
                        --T2I_model=AgenticEditing \
                        --Editing_model=FluxKontext \
                        --T2I_prompt=AR_with_sensation.jinja \
                        --AD_type=SensoryAdsAll \
                        --with_physical_sensation=True \
                        --test_set_sensation=train/sensation_annotations_parsed.json \
                        --test_set_QA=train/SensoryAd_Action_Reason.json \
                        --ablate_agents="$agents" \
                        --critic_priority="$priority" \
                        --experiment_datetime="ablation_${RUN_ID}_${name}"
done
