import platform
from pathlib import Path

from loguru import logger


def rmdir(directory):
    """Recursively delete a directory and all its contents.

    Args:
        directory: Path to the directory to remove. Must exist.
    """
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


def init_tmrl_data():
    """Wipe and re-create ~/TmrlData with default resources downloaded from GitHub.

    Downloads ``resources.zip`` from the tmrl GitHub release matching the installed
    package version (falling back to a known-good release), extracts it under
    ``~/TmrlData``, and copies the default config, reward trajectory, and
    pre-trained weights into their respective sub-directories. On Windows, also
    installs the OpenPlanet plugin files into ``~/OpenplanetNext/Plugins`` if that
    directory exists.

    Raises:
        AssertionError: If ``~/TmrlData`` still exists after the wipe step.
        ConnectionError: If ``resources.zip`` could not be downloaded from any URL.
    """
    from shutil import copy2
    from zipfile import ZipFile

    from tmrl.tools.init_package.resources_bundle import download_resources_zip

    home_folder = Path.home()
    tmrl_folder = home_folder / "TmrlData"

    if tmrl_folder.exists():
        rmdir(tmrl_folder)

    assert not tmrl_folder.exists(), f"Failed to delete {tmrl_folder}"

    checkpoints_folder = tmrl_folder / "checkpoints"
    dataset_folder = tmrl_folder / "dataset"
    reward_folder = tmrl_folder / "reward"
    weights_folder = tmrl_folder / "weights"
    config_folder = tmrl_folder / "config"
    checkpoints_folder.mkdir(parents=True, exist_ok=True)
    dataset_folder.mkdir(parents=True, exist_ok=True)
    reward_folder.mkdir(parents=True, exist_ok=True)
    weights_folder.mkdir(parents=True, exist_ok=True)
    config_folder.mkdir(parents=True, exist_ok=True)

    resources_target = tmrl_folder / "resources.zip"
    resources_url_used = download_resources_zip(resources_target)
    logger.info("Downloaded TMRL resources from {}", resources_url_used)

    with ZipFile(resources_target, "r") as zip_ref:
        zip_ref.extractall(tmrl_folder)

    resources_target.unlink()

    resources_folder = tmrl_folder / "resources"
    copy2(resources_folder / "config.json", config_folder)
    copy2(resources_folder / "reward.pkl", reward_folder)
    copy2(resources_folder / "SAC_4_LIDAR_pretrained.tmod", weights_folder)
    copy2(resources_folder / "SAC_4_imgs_pretrained.tmod", weights_folder)

    if platform.system() == "Windows":
        openplanet_folder = home_folder / "OpenplanetNext"

        if openplanet_folder.exists():
            try:
                # Remove legacy script-based plugin files from earlier tmrl versions.
                op_scripts_folder = openplanet_folder / "Scripts"
                if op_scripts_folder.exists():
                    to_remove = [
                        op_scripts_folder / "Plugin_GrabData_0_1.as",
                        op_scripts_folder / "Plugin_GrabData_0_1.as.sig",
                        op_scripts_folder / "Plugin_GrabData_0_2.as",
                        op_scripts_folder / "Plugin_GrabData_0_2.as.sig",
                    ]
                    for old_file in to_remove:
                        if old_file.exists():
                            old_file.unlink()
                op_plugins_folder = openplanet_folder / "Plugins"
                op_plugins_folder.mkdir(parents=True, exist_ok=True)
                tm20_plugin_1 = resources_folder / "Plugins" / "TMRL_GrabData.op"
                tm20_plugin_2 = resources_folder / "Plugins" / "TMRL_SaveGhost.op"
                copy2(tm20_plugin_1, op_plugins_folder)
                copy2(tm20_plugin_2, op_plugins_folder)
            except Exception as e:
                print(
                    "Exception while copying the OpenPlanet plugin automatically. "
                    "Please copy the plugin manually for TrackMania 2020 support. "
                    f"The caught exception was: {e!s}.",
                )
        else:
            print(
                f"The OpenPlanet folder was not found at {openplanet_folder}. \
            Please copy the OpenPlanet script and signature manually for TrackMania 2020 support."
            )


# Auto-initialize on first import so callers never need to run the setup step manually.
TMRL_FOLDER = Path.home() / "TmrlData"

if not TMRL_FOLDER.exists():
    logger.warning("The TMRL folder was not found on your machine. Attempting download...")
    init_tmrl_data()
    logger.info(
        "TMRL folder successfully downloaded, please wait for initialization to complete..."
    )
