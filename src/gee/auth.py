import os
import sys
import logging
from typing import Optional
import ee

def initialize_gee(project_id: Optional[str] = None, logger: Optional[logging.Logger] = None, fail_fast: bool = True) -> bool:
    """
    Initializes Google Earth Engine strictly without interactive authentication loops.
    Reuses existing credentials or uses project ID from config/env.
    """
    if logger is None:
        logger = logging.getLogger("ParbhaniDrought")
        
    target_project = project_id or os.environ.get("GEE_PROJECT")
    logger.info("Initializing Google Earth Engine connection...")
    
    try:
        if target_project:
            logger.info(f"Connecting to GEE with Project ID: '{target_project}'...")
            ee.Initialize(project=target_project)
        else:
            logger.info("Connecting to GEE with default persistent credentials...")
            ee.Initialize()
        logger.info("Google Earth Engine initialized successfully.")
        return True
    except Exception as e:
        logger.error("=" * 65)
        logger.error("Google Earth Engine initialization failed!")
        logger.error(f"Error: {e}")
        logger.error("-" * 65)
        logger.error("Authentication must be performed outside the automated pipeline.")
        logger.error("Please run the following in your terminal to authenticate:")
        logger.error("    earthengine authenticate")
        if not target_project:
            logger.error("Set your Google Cloud project via:")
            logger.error("    set GEE_PROJECT=your-project-id")
        logger.error("=" * 65)
        if fail_fast:
            sys.exit(1)
        return False
