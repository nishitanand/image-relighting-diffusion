#!/bin/bash
# Setup accelerate configuration for 8xA100 training

echo "Configuring accelerate for 8xA100 training..."

accelerate config --config_file accelerate_config.yaml <<EOF
0
0
0
0
8
0
fp16
EOF

echo "Accelerate configuration created: accelerate_config.yaml"
echo "You can now run: ./train.sh"

