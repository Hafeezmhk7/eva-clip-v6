#!/bin/bash
# setup_workspace.sh - Create shared workspace for BLIP3-o project

echo "🏗️  Setting up BLIP3-o Shared Workspace"
echo "======================================="

USER=$(whoami)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Use SCRATCH_SHARED for workspace
if [ -n "$SCRATCH_SHARED" ] && [ -d "$SCRATCH_SHARED" ]; then
    WORKSPACE_BASE="$SCRATCH_SHARED/${USER}_blip3o_workspace"
    echo "📁 Using SCRATCH_SHARED for workspace: $SCRATCH_SHARED"
else
    echo "❌ SCRATCH_SHARED not available!"
    echo "   SCRATCH_SHARED env var: ${SCRATCH_SHARED:-not set}"
    echo "   This script requires SCRATCH_SHARED to be available"
    exit 1
fi

echo "🎯 Workspace location: $WORKSPACE_BASE"

# Create main workspace directory
mkdir -p "$WORKSPACE_BASE"

# Create structured subdirectories
echo "📂 Creating workspace structure..."

# Main directories
mkdir -p "$WORKSPACE_BASE/embeddings"           # For chunked embeddings
mkdir -p "$WORKSPACE_BASE/training"             # For training outputs/checkpoints
mkdir -p "$WORKSPACE_BASE/models"               # For final models
mkdir -p "$WORKSPACE_BASE/logs"                 # For training logs
mkdir -p "$WORKSPACE_BASE/cache"                # For model caches (HF, torch, etc.)
mkdir -p "$WORKSPACE_BASE/data"                 # For downloaded tar files
mkdir -p "$WORKSPACE_BASE/temp"                 # For temporary files

# Create subdirectories for different experiments
mkdir -p "$WORKSPACE_BASE/embeddings/256_tokens"
mkdir -p "$WORKSPACE_BASE/training/256_tokens"
mkdir -p "$WORKSPACE_BASE/models/256_tokens"

echo "✅ Workspace structure created:"
echo "   📁 $WORKSPACE_BASE/"
echo "   ├── 📂 embeddings/           # Chunked embeddings storage"
echo "   │   └── 📂 256_tokens/       # 256-token specific embeddings"
echo "   ├── 📂 training/             # Training outputs & checkpoints"
echo "   │   └── 📂 256_tokens/       # 256-token training runs"
echo "   ├── 📂 models/               # Final trained models"
echo "   │   └── 📂 256_tokens/       # 256-token models"
echo "   ├── 📂 logs/                 # Training and job logs"
echo "   ├── 📂 cache/                # Model caches (HF, torch, wandb)"
echo "   ├── 📂 data/                 # Downloaded dataset tar files"
echo "   └── 📂 temp/                 # Temporary files"

# Create environment setup script
cat > "$WORKSPACE_BASE/setup_env.sh" << 'EOF'
#!/bin/bash
# Source this script to set up environment variables for BLIP3-o workspace

# Get the workspace directory (directory containing this script)
export BLIP3O_WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set up paths for different components
export BLIP3O_EMBEDDINGS_DIR="$BLIP3O_WORKSPACE/embeddings/256_tokens"
export BLIP3O_TRAINING_DIR="$BLIP3O_WORKSPACE/training/256_tokens"
export BLIP3O_MODELS_DIR="$BLIP3O_WORKSPACE/models/256_tokens"
export BLIP3O_LOGS_DIR="$BLIP3O_WORKSPACE/logs"
export BLIP3O_CACHE_DIR="$BLIP3O_WORKSPACE/cache"
export BLIP3O_DATA_DIR="$BLIP3O_WORKSPACE/data"
export BLIP3O_TEMP_DIR="$BLIP3O_WORKSPACE/temp"

# Set up cache directories for various tools
export TORCH_HOME="$BLIP3O_CACHE_DIR/torch"
export HF_HOME="$BLIP3O_CACHE_DIR/huggingface"
export TRANSFORMERS_CACHE="$BLIP3O_CACHE_DIR/transformers"
export WANDB_DIR="$BLIP3O_CACHE_DIR/wandb"

# Create directories if they don't exist
mkdir -p "$BLIP3O_EMBEDDINGS_DIR"
mkdir -p "$BLIP3O_TRAINING_DIR"
mkdir -p "$BLIP3O_MODELS_DIR"
mkdir -p "$BLIP3O_LOGS_DIR"
mkdir -p "$BLIP3O_CACHE_DIR"
mkdir -p "$BLIP3O_DATA_DIR"
mkdir -p "$BLIP3O_TEMP_DIR"
mkdir -p "$TORCH_HOME"
mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$WANDB_DIR"

echo "✅ BLIP3-o workspace environment set up:"
echo "   🎯 Workspace: $BLIP3O_WORKSPACE"
echo "   📊 Embeddings: $BLIP3O_EMBEDDINGS_DIR"
echo "   🏋️ Training: $BLIP3O_TRAINING_DIR"
echo "   🤖 Models: $BLIP3O_MODELS_DIR"
echo "   📝 Logs: $BLIP3O_LOGS_DIR"
echo "   💾 Cache: $BLIP3O_CACHE_DIR"
EOF

chmod +x "$WORKSPACE_BASE/setup_env.sh"

# Create cleanup script
cat > "$WORKSPACE_BASE/cleanup.sh" << 'EOF'
#!/bin/bash
# Cleanup script for BLIP3-o workspace

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🧹 BLIP3-o Workspace Cleanup"
echo "============================="
echo "Workspace: $WORKSPACE"
echo ""

# Show current usage
echo "📊 Current disk usage:"
du -sh "$WORKSPACE"/* 2>/dev/null || echo "No files to show"
echo ""

echo "What would you like to clean up?"
echo "1) Cache files only (safe)"
echo "2) Training checkpoints (keep final models)"
echo "3) Downloaded data files (keep embeddings)"
echo "4) All temporary files"
echo "5) Everything except final models (DESTRUCTIVE)"
echo "6) Cancel"

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo "🧹 Cleaning cache files..."
        rm -rf "$WORKSPACE/cache"/*
        echo "✅ Cache cleaned"
        ;;
    2)
        echo "🧹 Cleaning training checkpoints..."
        find "$WORKSPACE/training" -name "checkpoint-*" -type d -exec rm -rf {} + 2>/dev/null
        echo "✅ Training checkpoints cleaned"
        ;;
    3)
        echo "🧹 Cleaning downloaded data files..."
        rm -rf "$WORKSPACE/data"/*
        echo "✅ Data files cleaned"
        ;;
    4)
        echo "🧹 Cleaning temporary files..."
        rm -rf "$WORKSPACE/temp"/*
        rm -rf "$WORKSPACE/cache"/*
        echo "✅ Temporary files cleaned"
        ;;
    5)
        echo "⚠️  This will delete everything except final models!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            rm -rf "$WORKSPACE/cache"/*
            rm -rf "$WORKSPACE/data"/*
            rm -rf "$WORKSPACE/temp"/*
            rm -rf "$WORKSPACE/training"/*
            rm -rf "$WORKSPACE/embeddings"/*
            rm -rf "$WORKSPACE/logs"/*
            echo "🗑️ Workspace cleaned (models preserved)"
        else
            echo "Cancelled"
        fi
        ;;
    6)
        echo "Cancelled"
        ;;
    *)
        echo "Invalid choice"
        ;;
esac

echo ""
echo "📊 Current disk usage after cleanup:"
du -sh "$WORKSPACE"/* 2>/dev/null || echo "No files to show"
EOF

chmod +x "$WORKSPACE_BASE/cleanup.sh"

# Create status script
cat > "$WORKSPACE_BASE/status.sh" << 'EOF'
#!/bin/bash
# Status script for BLIP3-o workspace

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📊 BLIP3-o Workspace Status"
echo "==========================="
echo "📁 Workspace: $WORKSPACE"
echo ""

# Check disk usage
echo "💾 Disk Usage:"
if [ -d "$WORKSPACE" ]; then
    du -sh "$WORKSPACE"/* 2>/dev/null | sort -hr || echo "No files found"
    echo ""
    echo "Total: $(du -sh "$WORKSPACE" | cut -f1)"
else
    echo "Workspace not found!"
    exit 1
fi

echo ""

# Check for embeddings
echo "📊 Embeddings Status:"
EMBEDDINGS_DIR="$WORKSPACE/embeddings/256_tokens"
if [ -f "$EMBEDDINGS_DIR/embeddings_manifest.json" ]; then
    echo "✅ Embeddings found"
    python3 -c "
import json
try:
    with open('$EMBEDDINGS_DIR/embeddings_manifest.json', 'r') as f:
        manifest = json.load(f)
    print(f'   Shards: {manifest[\"total_shards\"]}')
    print(f'   Samples: {manifest[\"total_samples\"]:,}')
    print(f'   Format: {manifest[\"format_version\"]}')
except:
    print('   Error reading manifest')
" 2>/dev/null || echo "   Could not read manifest details"
else
    echo "❌ No embeddings found"
    echo "   Run: sbatch job_scripts/extract_emb_256_chunk.job"
fi

echo ""

# Check for models
echo "🤖 Models Status:"
MODELS_DIR="$WORKSPACE/models/256_tokens"
if [ -d "$MODELS_DIR" ] && [ "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
    echo "✅ Models found:"
    ls -la "$MODELS_DIR"/ | grep "^d" | awk '{print "   " $9}' || echo "   No model directories"
else
    echo "❌ No models found"
    echo "   Run training to create models"
fi

echo ""

# Check for active training
echo "🏋️ Training Status:"
TRAINING_DIR="$WORKSPACE/training/256_tokens"
if [ -d "$TRAINING_DIR" ] && [ "$(ls -A "$TRAINING_DIR" 2>/dev/null)" ]; then
    echo "📁 Training directories found:"
    ls -la "$TRAINING_DIR"/ | grep "^d" | awk '{print "   " $9}' || echo "   No training directories"
    
    # Check for recent activity
    RECENT=$(find "$TRAINING_DIR" -name "*.log" -o -name "pytorch_model.bin" -newermt "1 hour ago" 2>/dev/null | wc -l)
    if [ "$RECENT" -gt 0 ]; then
        echo "🔥 Recent training activity detected (last hour)"
    fi
else
    echo "❌ No training outputs found"
fi

echo ""

# Show environment variables if set
echo "🔧 Environment:"
if [ -n "$BLIP3O_WORKSPACE" ]; then
    echo "✅ Environment variables set"
    echo "   BLIP3O_WORKSPACE: $BLIP3O_WORKSPACE"
    echo "   BLIP3O_EMBEDDINGS_DIR: $BLIP3O_EMBEDDINGS_DIR"
else
    echo "⚠️  Environment not set. Run:"
    echo "   source $WORKSPACE/setup_env.sh"
fi
EOF

chmod +x "$WORKSPACE_BASE/status.sh"

# Create project symlink
echo ""
echo "🔗 Creating project symlink..."
mkdir -p "./blip3o_workspace"
rm -f "./blip3o_workspace/shared"
ln -sf "$WORKSPACE_BASE" "./blip3o_workspace/shared"

# Set up initial environment
source "$WORKSPACE_BASE/setup_env.sh"

# Save workspace path for easy access
echo "$WORKSPACE_BASE" > "$HOME/.blip3o_workspace_path"

echo ""
echo "✅ BLIP3-o Shared Workspace Setup Complete!"
echo "==========================================="
echo ""
echo "📁 Workspace location: $WORKSPACE_BASE"
echo "🔗 Project symlink: ./blip3o_workspace/shared"
echo ""
echo "🚀 Next Steps:"
echo "1. Source the environment in your job scripts:"
echo "   source $WORKSPACE_BASE/setup_env.sh"
echo ""
echo "2. Check workspace status anytime:"
echo "   $WORKSPACE_BASE/status.sh"
echo ""
echo "3. Clean up when needed:"
echo "   $WORKSPACE_BASE/cleanup.sh"
echo ""
echo "4. Your environment variables are now set:"
echo "   BLIP3O_EMBEDDINGS_DIR: $BLIP3O_EMBEDDINGS_DIR"
echo "   BLIP3O_TRAINING_DIR: $BLIP3O_TRAINING_DIR"
echo "   BLIP3O_MODELS_DIR: $BLIP3O_MODELS_DIR"
echo ""
echo "🎯 Ready to extract embeddings and train models!"