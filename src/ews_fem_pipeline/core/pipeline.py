from pathlib import Path
from ews_fem_pipeline.prepare_simulation import (
    generate_mesh,
    write_to_feb,
    load_settings_from_toml,
    write_settings_to_toml,
)
from ews_fem_pipeline.run_simulation import FEBioRunner
from ews_fem_pipeline.convert_simulation import feb_to_blender

from ews_fem_pipeline.analysis.analyze_results import summarize_displacement

def run_pipeline(input_files: list[Path], jobs: int = 0):
    feb_files = []

    for filepath in input_files:

        # Logging for each file
        logger.info(f"Running pipeline for {filepath}")

        settings = load_settings_from_toml(filepath)

        # Save full settings
        output_dir = filepath.parent / "output"
        output_dir.mkdir(exist_ok=True)

        write_settings_to_toml(
            output_dir / f"{filepath.stem}_all_settings.toml",
            settings
        )

        mesh = generate_mesh(settings)

        write_to_feb(filepath, mesh, settings)

        feb_files.append(filepath.with_suffix(".feb"))

    # Run FEBio
    if jobs == 0:
        jobs = 1 if len(feb_files) == 1 else 4

    output_files = FEBioRunner().run(tuple(feb_files), jobs)

    # Convert
    for f in output_files:
        feb_to_blender(f)

    # summarize_displacement(output_files) # Summary of results

    return output_files