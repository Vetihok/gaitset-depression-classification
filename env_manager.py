from dataclasses import dataclass
import logging
import os
from typing import ClassVar, List, Optional

import yaml

from opts import get_opts

log = logging.getLogger(__name__)

@dataclass
class EnvManager:
    """Stores execution metadata for managing multiple config runs."""

    # --- Singleton instance ---
    _instance: ClassVar[Optional['EnvManager']] = None

    launch_datetime: str

    config_list: List
    current_config_id: str | None
    current_run_id: int | None
    current_exp_folder: str | None

    work_path: str = "output"
    subfolder: str = ""
    cuda_visible_devices: str = "0,1"
    use_cpu: bool = False
    num_runs: int = 1
    overwrite_test_results: bool = False
    overwrite_average_results: bool = False
    restore_mode: str = "last"

    @classmethod
    def from_yaml(cls, yaml_file, config_list, launch_datetime):
        """Initialize environment metadata for execution.
        
        Args:
            launch_datetime: ISO format datetime string for this execution session
            config_list: List of config IDs being executed
            work_path: Base work path from config
            subfolder: Optional subfolder from config

        Returns:
            EnvManager: a new instance of EnvManager
        """
        if cls._instance is not None:
            return cls._instance

        with open(yaml_file) as f:
            self_config = yaml.safe_load(f)

        cls._instance = cls(
            launch_datetime=launch_datetime,
            config_list=config_list,
            current_config_id = None,
            current_run_id = None,
            current_exp_folder = None,
            work_path=self_config.get('WORK_PATH', "output"),
            subfolder=self_config.get('SUBFOLDER', ""),
            cuda_visible_devices=self_config.get('CUDA_VISIBLE_DEVICES', '0,1'),
            use_cpu=self_config.get('USE_CPU', False),
            num_runs=self_config.get('NUM_RUNS', 1),
            overwrite_test_results=self_config.get('OVERWRITE_TEST_RESULTS', False),
            overwrite_average_results=self_config.get('OVERWRITE_AVERAGE_RESULTS', False),
            restore_mode=self_config.get('RESTORE_MODE', "last"),
        )

        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance. Raises an error if not initialized."""
        if cls._instance is None:
            raise RuntimeError(
                "EnvManager not initialized. Call `EnvManager.from_yaml()` first."
            )
        return cls._instance

    def has_multiple_configs(self):
        """Return True if executing multiple configurations."""
        return len(self.config_list) > 1
    
    def has_multiple_runs(self):
        """Return True if executing multiple runs per configuration."""
        return self.num_runs > 1
    
    def set_current_config_id(self, config_id):
        """Set the current config ID being processed."""
        self.current_config_id = config_id
    
    def set_current_run_id(self, run_id):
        """Set the current run ID being processed."""
        self.current_run_id = run_id

    # Utilities

    def get_current_exp_folder(self):
        """Get or create the parent folder name for the whole experiment.
        
        Returns:
            Name of parent folder
        """
        opt = get_opts()
        if self.current_exp_folder is None and self.has_multiple_configs():
            def first_char(s):
                return str(s)[0]
            config_names = "_".join(map(first_char, self.config_list))
            self.current_exp_folder = f"exp_{config_names}_{self.launch_datetime}"
        elif self.current_exp_folder is None:
            self.current_exp_folder = f"exp_{self.config_list[0]}_{self.launch_datetime}"
        
        return self.current_exp_folder

    def get_configs_dir(self):
        return os.path.join(self.work_path, "configs")

    def get_dir(self, checkpoint=False, common=False, results=False, mkdir=True):
        """
        Return the current working directory depending on the current config and the current run being processed.

        Each parameter is used to add a subfolder in the current directory.
        """
        opt = get_opts()
        path = self.work_path

        # --- debug option ---
        if opt.debug:
            path = os.path.join(path, "debug")
        else:
            path = os.path.join(path, "exp")
        
        # --- subfolder in env_config file ---
        if self.subfolder and not opt.debug:
            path = os.path.join(path, self.subfolder)

        # --- experiment folder ---
        restore_dir = opt.restore_dir
        if restore_dir is not None:
            if os.path.isabs(restore_dir):
                path = restore_dir
            else:
                path = os.path.join(path, restore_dir)
        else:
            path = os.path.join(path, self.get_current_exp_folder())
        
        # --- only for multiple configurations ---
        if self.has_multiple_configs():
            if self.current_config_id is not None and not common:
                path = os.path.join(path, f"config_{self.current_config_id}")
            elif not results:
                # During initialization, use a common folder for shared files
                path = os.path.join(path, "common")

        # --- run subfolder for multiple runs per configuration ---
        if self.has_multiple_runs() and self.current_run_id is not None:
            path = os.path.join(path, f"run_{self.current_run_id}")

        # --- results dir ---
        if results:
            path = os.path.join(path, "results")
        
        # --- checkpoint for model saving ---
        if checkpoint:
            path = os.path.join(path, "checkpoint")

        if mkdir:
            os.makedirs(path, exist_ok=True)
        return path

    def get_general_save_path(self, name, format, checkpoint=False, common=False, results=False):
        return os.path.join(
            self.get_dir(checkpoint=checkpoint, common=common, results=results),
            name + "." + format)

    def get_tested_runs_for_config(self, config_id):
        """
        Return a set of run ids that are tested, e. g. a folder named results exists and contains predictions.npz and results.json.

        It works with one or more config and one ore more run.
        """
        temp_config_id = self.current_config_id
        temp_run_id = self.current_run_id
        self.set_current_config_id(config_id)

        tested_runs = set()
        for run in range(self.num_runs):
            self.set_current_run_id(run)
            run_dir = self.get_dir(mkdir=False)
            if not os.path.isdir(run_dir):
                continue
            
            for file in os.listdir(run_dir):

                results_dir = os.path.join(run_dir, file)

                if 'results' in file and os.path.isdir(results_dir):

                    dir_list = os.listdir(results_dir)

                    if 'predictions.npz' in dir_list and 'results.json' in dir_list:

                        tested_runs.add(run)

        log.debug(f'Tested runs: {str(tested_runs)=}')
        self.set_current_config_id(temp_config_id)
        self.set_current_run_id(temp_run_id)
        return tested_runs

    def get_completed_configs(self):
        """
        Return a list of config ids that are completed, e. g. the results folder of the config contains results.json and checkpoint folder is not empty.
        
        It works with one or more config.
        """
        temp_config_id = self.current_config_id

        completed_configs = []
        for config_id in self.config_list:
            self.set_current_config_id(config_id)

            results_dir = self.get_dir(results=True, mkdir=False)
            checkpoint_dir = self.get_dir(results=True, mkdir=False)
            if os.path.exists(results_dir) and os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
                dir_list = os.listdir(results_dir)
                if dir_list and 'results.json' in dir_list:
                    completed_configs.append(config_id)

        log.debug(f'Completed configs: {str(completed_configs)=}')
        self.set_current_config_id(temp_config_id)
        return completed_configs
                