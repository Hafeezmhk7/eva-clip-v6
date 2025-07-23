"""
src/modules/evaluation/__init__.py
Evaluation modules initialization
"""

from .blip3o_detailed_evaluator import BLIP3oDetailedEvaluator, create_detailed_evaluator

# Log initialization
import logging
logger = logging.getLogger(__name__)
logger.info("BLIP3-o evaluation modules initialized")