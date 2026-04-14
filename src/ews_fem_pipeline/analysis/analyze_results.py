import numpy as np
from pathlib import Path
import logging

from ews_fem_pipeline.analysis import data_analysis_main

def summarize_displacement(vtk_files: list[Path]):
    # placeholder (later uitbreiden met echte VTK reader)
    print(f"Analyzing {len(vtk_files)} files")

    # voorbeeld output
    return {
        "max_displacement": None,
        "mean_displacement": None
    }





logger = logging.getLogger(__name__)

def analyze_results(feb_files):
    logger.info("Starting analysis...")

    for feb in feb_files:
        vtk_dir = feb.parent / "output"

        if not vtk_dir.exists():
            logger.warning(f"No output for {feb}")
            continue

        logger.info(f"Analyzing {feb}")
        data_analysis_main.run(vtk_dir)  # pas aan naar jouw structuur

    logger.info("Analysis complete")