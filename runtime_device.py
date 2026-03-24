"""Runtime device helpers that must stay torch-free."""

import os


def apply_visible_devices(config):
    """Apply CUDA_VISIBLE_DEVICES from config before importing torch."""
    device_cfg = (config or {}).get('device', {})
    visible_devices = device_cfg.get('visible_devices')
    if visible_devices is None:
        return None

    if isinstance(visible_devices, (list, tuple)):
        visible_str = ','.join(str(device_id) for device_id in visible_devices)
    else:
        visible_str = str(visible_devices)

    os.environ['CUDA_VISIBLE_DEVICES'] = visible_str
    return visible_str


def get_configured_visible_device_count(config):
    """Return the number of configured visible CUDA devices, if specified."""
    device_cfg = (config or {}).get('device', {})
    visible_devices = device_cfg.get('visible_devices')
    if visible_devices is None:
        return None

    if isinstance(visible_devices, (list, tuple)):
        return len(visible_devices)

    visible_str = str(visible_devices).strip()
    if not visible_str:
        return 0
    return len([item for item in visible_str.split(',') if item.strip()])


def resolve_device_type(config, cli_device=None):
    """Resolve the requested runtime device."""
    if cli_device:
        return cli_device
    return (config or {}).get('device', {}).get('type', 'cuda')
