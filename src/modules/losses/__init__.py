"""
src/modules/losses/__init__.py
UPDATED: Loss modules initialization with all new functions and fixes
"""

import logging

logger = logging.getLogger(__name__)

# Import the COMPLETE FIXED flow matching loss with all new functions
try:
    from .blip3o_flow_matching_loss import (
        # Core classes
        BLIP3oFlowMatchingLoss,
        
        # Factory functions
        create_blip3o_flow_matching_loss,
        create_debug_loss,
        create_production_loss,
        
        # Utility functions
        analyze_loss_scaling,
    )
    
    FLOW_MATCHING_LOSS_AVAILABLE = True
    logger.info("✅ COMPLETE FIXED BLIP3-o flow matching loss loaded successfully")
    logger.info("   Available functions:")
    logger.info("     • BLIP3oFlowMatchingLoss (main class)")
    logger.info("     • create_blip3o_flow_matching_loss (factory)")
    logger.info("     • create_debug_loss (debug configuration)")
    logger.info("     • create_production_loss (production configuration)")
    logger.info("     • analyze_loss_scaling (debugging utility)")
    
except ImportError as e:
    FLOW_MATCHING_LOSS_AVAILABLE = False
    logger.error(f"❌ Failed to load FIXED flow matching loss: {e}")
    logger.error("   Make sure you have replaced the flow matching loss file with the fixed version")
    raise ImportError(f"FIXED BLIP3-o flow matching loss is required but failed to load: {e}")

# Verify that we have the fixed version with scaling parameters
try:
    # Test that we can create a loss with the new scaling parameters
    test_loss = create_blip3o_flow_matching_loss(
        velocity_scale=0.1,
        target_norm_scale=1.0,
        adaptive_scaling=True
    )
    logger.info("✅ Verified FIXED version with scaling parameters")
    del test_loss  # Clean up
    
except Exception as e:
    logger.error(f"❌ Failed to verify FIXED version: {e}")
    logger.error("   The flow matching loss file may not be the complete fixed version")
    raise ImportError(f"FIXED flow matching loss verification failed: {e}")

# Log initialization with fix information
logger.info("BLIP3-o loss modules initialized with COMPLETE FIXES")
logger.info("Key fixes applied:")
logger.info("  ✅ Velocity scaling to address norm mismatch")
logger.info("  ✅ Adaptive scaling mechanism")
logger.info("  ✅ Proper rectified flow implementation")
logger.info("  ✅ Consistent normalization handling")
logger.info("  ✅ Comprehensive evaluation metrics")

# Export all functions and classes
__all__ = [
    # Availability flag
    "FLOW_MATCHING_LOSS_AVAILABLE",
    
    # Core classes
    "BLIP3oFlowMatchingLoss",
    
    # Factory functions  
    "create_blip3o_flow_matching_loss",
    "create_debug_loss",
    "create_production_loss",
    
    # Utility functions
    "analyze_loss_scaling",
]

# Helper functions for easy access
def get_fixed_loss_function(**kwargs):
    """
    Get the fixed flow matching loss with recommended parameters
    
    Returns:
        BLIP3oFlowMatchingLoss with all fixes applied
    """
    if not FLOW_MATCHING_LOSS_AVAILABLE:
        raise RuntimeError("FIXED flow matching loss not available")
    
    # Set recommended defaults with fixes
    defaults = {
        'velocity_scale': 0.1,          # CRITICAL: Fix scale mismatch
        'target_norm_scale': 1.0,       # Keep targets normalized
        'adaptive_scaling': True,        # Enable adaptive scaling
        'prediction_type': 'velocity',   # BLIP3-o standard
        'normalize_targets': True,       # Consistent normalization
        'flow_type': 'rectified',       # BLIP3-o paper alignment
    }
    
    # Override with user parameters
    defaults.update(kwargs)
    
    return create_blip3o_flow_matching_loss(**defaults)

def get_overfitting_loss_function(**kwargs):
    """
    Get loss function optimized for overfitting tests
    """
    if not FLOW_MATCHING_LOSS_AVAILABLE:
        raise RuntimeError("FIXED flow matching loss not available")
    
    # Overfitting-specific parameters
    defaults = {
        'velocity_scale': 0.1,
        'target_norm_scale': 1.0,
        'adaptive_scaling': True,
        'ema_decay': 0.95,  # Faster adaptation for overfitting
    }
    
    defaults.update(kwargs)
    return create_blip3o_flow_matching_loss(**defaults)

def print_loss_fixes():
    """Print information about the fixes applied"""
    print("🔧 BLIP3-o Loss Function Fixes Applied:")
    print("=" * 40)
    print("✅ Scale Mismatch Solution:")
    print("   • velocity_scale=0.1 (scale down velocity targets)")
    print("   • output_scale=0.1 (scale down model outputs)")
    print("   • adaptive_scaling=True (auto-adjust during training)")
    print()
    print("✅ Flow Matching Improvements:")
    print("   • Proper rectified flow implementation")
    print("   • Correct velocity target computation")
    print("   • Consistent normalization handling")
    print()
    print("✅ Evaluation Enhancements:")
    print("   • Comprehensive similarity metrics")
    print("   • Training/evaluation mode compatibility")
    print("   • Detailed progress tracking")
    print()
    print("✅ Debugging Features:")
    print("   • Norm tracking and analysis")
    print("   • Adaptive scaling monitoring")
    print("   • Quality assessment metrics")
    print("=" * 40)

# Add helper functions to exports
__all__.extend([
    "get_fixed_loss_function",
    "get_overfitting_loss_function", 
    "print_loss_fixes",
])

# Validate the module is properly loaded
def validate_loss_module():
    """Validate that the loss module is properly loaded with fixes"""
    if not FLOW_MATCHING_LOSS_AVAILABLE:
        return False, "Flow matching loss not available"
    
    try:
        # Test creating loss with all new parameters
        test_loss = create_blip3o_flow_matching_loss(
            velocity_scale=0.1,
            target_norm_scale=1.0,
            adaptive_scaling=True,
            ema_decay=0.99
        )
        
        # Test that it has the new methods
        if not hasattr(test_loss, 'get_scaling_info'):
            return False, "Missing new scaling methods"
        
        if not hasattr(test_loss, 'update_adaptive_scaling'):
            return False, "Missing adaptive scaling methods"
        
        return True, "All fixes verified"
        
    except Exception as e:
        return False, f"Validation failed: {e}"

# Run validation on import
is_valid, validation_msg = validate_loss_module()
if is_valid:
    logger.info(f"✅ Loss module validation: {validation_msg}")
else:
    logger.error(f"❌ Loss module validation failed: {validation_msg}")
    logger.error("   Please ensure you have the complete fixed version of blip3o_flow_matching_loss.py")