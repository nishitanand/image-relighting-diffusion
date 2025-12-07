#!/bin/bash
# Quick test script to verify installation

echo "Testing CLIP-Based Image Filter Installation"
echo "============================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Check required packages
echo "Checking required packages..."
python3 -c "
try:
    import torch
    print('✓ torch')
except:
    print('✗ torch - MISSING')

try:
    import torchvision
    print('✓ torchvision')
except:
    print('✗ torchvision - MISSING')

try:
    import transformers
    print('✓ transformers')
except:
    print('✗ transformers - MISSING')

try:
    import PIL
    print('✓ Pillow')
except:
    print('✗ Pillow - MISSING')

try:
    import numpy
    print('✓ numpy')
except:
    print('✗ numpy - MISSING')

try:
    import tqdm
    print('✓ tqdm')
except:
    print('✗ tqdm - MISSING')

try:
    import pandas
    print('✓ pandas')
except:
    print('✗ pandas - MISSING')

try:
    import open_clip
    print('✓ open_clip_torch')
except:
    print('✗ open_clip_torch - MISSING (will use transformers as fallback)')
"
echo ""
echo "============================================"
echo "Installation test complete!"
echo ""
echo "If any packages are missing, run:"
echo "  pip install -r requirements.txt"

