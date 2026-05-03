"""Rt-gym interface tokens for discrete **boundary lidar** (track-relative rays).

TMRL treats suffix-style vectors ending in ``TRACKMAP`` / ``LIDAR`` and the fused variants that
contain ``TRACKMAPIMAGES`` / ``LIDARIMAGES`` as one family. Legacy **screen-ray** layouts that used
``*LIDARPROGRESS*`` are rejected at validate time (see ``EnvironmentConfig``).
"""


def rtgym_discrete_boundary_lidar_vec(rt: str) -> bool:
    """True for discrete boundary vector obs (no mandatory camera stack)."""
    u = str(rt).upper()
    if "LIDARPROGRESS" in u:
        return False
    return u.endswith("TRACKMAP") or u.endswith("LIDAR")


def rtgym_discrete_boundary_lidar_images(rt: str) -> bool:
    """True when the token selects boundary geometry fused with an image history."""
    u = str(rt).upper()
    return "TRACKMAPIMAGES" in u or "LIDARIMAGES" in u


def rtgym_discrete_boundary_lidar_family(rt: str) -> bool:
    """Either vector-only or images+fusion boundary lidar."""
    return rtgym_discrete_boundary_lidar_vec(rt) or rtgym_discrete_boundary_lidar_images(rt)
