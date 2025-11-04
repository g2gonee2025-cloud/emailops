"""
Configuration Service Module

Handles all configuration management operations, abstracting config logic from the GUI.
"""

import json
import logging
from pathlib import Path
from typing import Any

from emailops.core_config import (
    EmailOpsConfig,
    get_default_config,
    reset_config,
)

logger = logging.getLogger(__name__)


class ConfigService:
    """Service for handling configuration management."""

    def __init__(self, settings_file: Path | None = None):
        """
        Initialize the configuration service.

        Args:
            settings_file: Optional path to settings file
        """
        self.settings_file = settings_file or (Path.home() / ".emailops_gui.json")
        self.current_config = None
        self.load_configuration()

    def load_configuration(self) -> EmailOpsConfig:
        """
        Load configuration from file.

        Returns:
            Loaded configuration object
        """
        try:
            self.current_config = EmailOpsConfig.load(self.settings_file)
            logger.info(f"Loaded configuration from {self.settings_file}")
            return self.current_config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)
            # Fall back to default config
            self.current_config = get_default_config()
            return self.current_config

    def save_configuration(self, config: EmailOpsConfig | None = None) -> bool:
        """
        Save configuration to file.

        Args:
            config: Configuration to save (uses current if None)

        Returns:
            True if successful, False otherwise
        """
        try:
            if config is None:
                config = self.current_config

            if config is None:
                logger.error("No configuration to save")
                return False

            config.save(self.settings_file)
            logger.info(f"Saved configuration to {self.settings_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}", exc_info=True)
            return False

    def apply_configuration(
        self, config_dict: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """
        Apply configuration changes from a dictionary.

        Args:
            config_dict: Dictionary of configuration values

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Load current config or get default
            new_config = self.current_config or get_default_config()

            # Apply each configuration value
            errors = []
            for key, value_str in config_dict.items():
                if not hasattr(new_config, key):
                    logger.warning(f"Unknown configuration key: {key}")
                    continue

                try:
                    # Convert value to appropriate type
                    converted_value = self._convert_config_value(
                        value_str, type(getattr(new_config, key))
                    )
                    setattr(new_config, key, converted_value)

                except Exception as e:
                    error_msg = f"Failed to set {key}: {e}"
                    errors.append(error_msg)
                    logger.warning(error_msg)

            # Validate the configuration
            is_valid, validation_errors = self.validate_configuration(new_config)
            if not is_valid:
                errors.extend(validation_errors)

            if errors:
                return False, "; ".join(errors)

            # Save and apply the configuration
            self.current_config = new_config
            self.save_configuration(new_config)

            # Update global configuration
            reset_config()
            new_config.update_environment()

            logger.info("Configuration applied successfully")
            return True, None

        except Exception as e:
            error_msg = f"Failed to apply configuration: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def validate_configuration(
        self, config: EmailOpsConfig | None = None
    ) -> tuple[bool, list[str]]:
        """
        Validate configuration values.

        Args:
            config: Configuration to validate (uses current if None)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if config is None:
            config = self.current_config

        if config is None:
            return False, ["No configuration to validate"]

        errors = []

        # Validate export root
        if config.core.export_root:
            export_path = Path(config.core.export_root)
            if not export_path.exists():
                errors.append(f"Export root does not exist: {config.core.export_root}")
            elif not export_path.is_dir():
                errors.append(f"Export root is not a directory: {config.core.export_root}")

        # Validate numeric ranges
        if config.unified.temperature < 0 or config.unified.temperature > 2:
            errors.append(f"Temperature must be between 0 and 2: {config.unified.temperature}")

        if config.search.k < 1 or config.search.k > 1000:
            errors.append(f"k must be between 1 and 1000: {config.search.k}")

        if config.search.sim_threshold < 0 or config.search.sim_threshold > 1:
            errors.append(
                f"Similarity threshold must be between 0 and 1: {config.search.sim_threshold}"
            )

        if config.processing.chunk_size < 100:
            errors.append(f"Chunk size must be at least 100: {config.processing.chunk_size}")

        if config.processing.chunk_overlap < 0 or config.processing.chunk_overlap >= config.processing.chunk_size:
            errors.append(
                f"Chunk overlap must be between 0 and chunk_size: {config.processing.chunk_overlap}"
            )

        if config.processing.num_workers < 1 or config.processing.num_workers > 32:
            errors.append(
                f"Number of workers must be between 1 and 32: {config.processing.num_workers}"
            )

        # Validate provider
        valid_providers = ["vertex", "openai", "anthropic"]
        if config.core.provider not in valid_providers:
            errors.append(
                f"Invalid provider: {config.core.provider}. Must be one of {valid_providers}"
            )

        # Validate reply policy
        valid_policies = ["reply_all", "smart", "sender_only"]
        if config.email.reply_policy not in valid_policies:
            errors.append(
                f"Invalid reply policy: {config.email.reply_policy}. Must be one of {valid_policies}"
            )

        return len(errors) == 0, errors

    def reset_to_defaults(self) -> EmailOpsConfig:
        """
        Reset configuration to defaults.

        Returns:
            Default configuration object
        """
        try:
            default_config = get_default_config()
            self.current_config = default_config
            self.save_configuration(default_config)

            # Update global configuration
            reset_config()
            default_config.update_environment()

            logger.info("Configuration reset to defaults")
            return default_config

        except Exception as e:
            logger.error(f"Failed to reset configuration: {e}", exc_info=True)
            return get_default_config()

    def get_configuration_dict(
        self, config: EmailOpsConfig | None = None
    ) -> dict[str, Any]:
        """
        Get configuration as a dictionary.

        Args:
            config: Configuration to convert (uses current if None)

        Returns:
            Configuration dictionary
        """
        if config is None:
            config = self.current_config

        if config is None:
            config = get_default_config()

        return config.to_dict()

    def get_configuration_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if self.current_config and hasattr(self.current_config, key):
            return getattr(self.current_config, key)
        return default

    def set_configuration_value(self, key: str, value: Any) -> bool:
        """
        Set a specific configuration value.

        Args:
            key: Configuration key
            value: New value

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.current_config:
                self.current_config = get_default_config()

            if not hasattr(self.current_config, key):
                logger.warning(f"Unknown configuration key: {key}")
                return False

            # Convert value to appropriate type
            target_type = type(getattr(self.current_config, key))
            converted_value = self._convert_config_value(value, target_type)

            setattr(self.current_config, key, converted_value)
            return True

        except Exception as e:
            logger.error(f"Failed to set configuration value: {e}", exc_info=True)
            return False

    def _convert_config_value(self, value: Any, target_type: type) -> Any:
        """
        Convert a value to the target configuration type.

        Args:
            value: Value to convert
            target_type: Target type

        Returns:
            Converted value

        Raises:
            ValueError: If conversion fails
        """
        if isinstance(value, target_type):
            return value

        value_str = str(value)

        if target_type is bool:
            return value_str.lower() in ("true", "1", "yes", "on")
        elif target_type is int:
            return int(value_str)
        elif target_type is float:
            return float(value_str)
        elif target_type is set:
            # Handle set by splitting the string
            if isinstance(value, str):
                return {item.strip() for item in value_str.split(",") if item.strip()}
            elif isinstance(value, (list, tuple)):
                return set(value)
            else:
                return {value_str}
        elif target_type is Path:
            return Path(value_str)
        else:
            return value_str

    def export_configuration(self, output_path: Path) -> bool:
        """
        Export configuration to a file.

        Args:
            output_path: Path to export configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            config_dict = self.get_configuration_dict()

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, default=str)

            logger.info(f"Exported configuration to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export configuration: {e}", exc_info=True)
            return False

    def import_configuration(self, input_path: Path) -> tuple[bool, str | None]:
        """
        Import configuration from a file.

        Args:
            input_path: Path to import configuration from

        Returns:
            Tuple of (success, error_message)
        """
        try:
            if not input_path.exists():
                return False, f"Configuration file not found: {input_path}"

            with input_path.open("r", encoding="utf-8") as f:
                config_dict = json.load(f)

            # Apply the imported configuration
            success, error = self.apply_configuration(config_dict)

            if success:
                logger.info(f"Imported configuration from {input_path}")

            return success, error

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in configuration file: {e}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Failed to import configuration: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def get_config_differences(
        self,
        config1: EmailOpsConfig | None = None,
        config2: EmailOpsConfig | None = None,
    ) -> dict[str, tuple[Any, Any]]:
        """
        Get differences between two configurations.

        Args:
            config1: First configuration (uses current if None)
            config2: Second configuration (uses default if None)

        Returns:
            Dictionary of differences {key: (value1, value2)}
        """
        if config1 is None:
            config1 = self.current_config or get_default_config()
        if config2 is None:
            config2 = get_default_config()

        dict1 = config1.to_dict()
        dict2 = config2.to_dict()

        differences = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())

        for key in all_keys:
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            if val1 != val2:
                differences[key] = (val1, val2)

        return differences

    def update_environment(self) -> bool:
        """
        Update environment variables from current configuration.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.current_config:
                self.current_config.update_environment()
                logger.info("Updated environment variables from configuration")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update environment: {e}", exc_info=True)
            return False
