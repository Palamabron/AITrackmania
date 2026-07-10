"""Rt-gym interface tokens for discrete **boundary lidar** (track-relative rays).

TMRL treats suffix-style vectors ending in ``TRACKMAP`` / ``LIDAR`` and the fused variants that
contain ``TRACKMAPIMAGES`` / ``LIDARIMAGES`` as one family. Legacy **screen-ray** layouts that used
``*LIDARPROGRESS*`` are rejected at validate time (see ``EnvironmentConfig``).
"""


def rtgym_discrete_boundary_lidar_vec(rt: str) -> bool:
    """Return ``True`` when the interface token selects discrete boundary vector observations.

    Tokens ending in ``TRACKMAP`` or ``LIDAR`` (case-insensitive) activate the
    pre-recorded track-boundary polyline layout without a mandatory camera stack.
    Legacy ``*LIDARPROGRESS*`` screen-ray tokens are explicitly excluded here
    (they are rejected at validation time by ``EnvironmentConfig``).

    Args:
        rt: Raw ``rtgym_interface`` string from config.

    Returns:
        ``True`` for boundary-vector layouts, ``False`` otherwise.
    """
    u = str(rt).upper()
    if "LIDARPROGRESS" in u:
        return False
    return u.endswith("TRACKMAP") or u.endswith("LIDAR")


def rtgym_discrete_boundary_lidar_images(rt: str) -> bool:
    """Return ``True`` when the token selects boundary geometry fused with an image history.

    Tokens containing ``TRACKMAPIMAGES`` or ``LIDARIMAGES`` (case-insensitive) activate
    the fused boundary-lidar + camera-stack pipeline.

    Args:
        rt: Raw ``rtgym_interface`` string from config.

    Returns:
        ``True`` for boundary+image fusion layouts, ``False`` otherwise.
    """
    u = str(rt).upper()
    return "TRACKMAPIMAGES" in u or "LIDARIMAGES" in u


def rtgym_discrete_boundary_lidar_family(rt: str) -> bool:
    """Return ``True`` for any discrete boundary lidar layout (vector-only or image-fused).

    Args:
        rt: Raw ``rtgym_interface`` string from config.

    Returns:
        ``True`` when either :func:`rtgym_discrete_boundary_lidar_vec` or
        :func:`rtgym_discrete_boundary_lidar_images` is ``True``.
    """
    return rtgym_discrete_boundary_lidar_vec(rt) or rtgym_discrete_boundary_lidar_images(rt)
