"""Central configuration: camera, uncertainty model and radio-link parameters."""
from dataclasses import dataclass, replace
import numpy as np

# World frame (landing site = origin): X = East, Y = North, Z = Up. Ground plane is Z = 0.

@dataclass(frozen=True)
class Camera:
    width: int = 640
    height: int = 480
    hfov_deg: float = 70.0
    height_m: float = 3.0      # camera height above the ground plane
    pitch_deg: float = 35.0    # tilt BELOW the horizon
    yaw_deg: float = 0.0       # 0 = looking North (+Y); positive = clockwise (towards +X)
    x: float = 0.0             # camera position relative to landing site (m)
    y: float = 0.0
    base_z: float = 0.0        # terrain elevation under the camera (m, landing-site datum)

    @property
    def z(self) -> float:      # camera height in world coordinates
        return self.base_z + self.height_m

    @property
    def f(self) -> float:      # focal length in pixels (square pixels)
        return (self.width / 2) / np.tan(np.radians(self.hfov_deg) / 2)

    def at(self, x: float, y: float, yaw_deg: float, base_z: float = 0.0) -> "Camera":
        return replace(self, x=x, y=y, yaw_deg=yaw_deg, base_z=base_z)


@dataclass(frozen=True)
class UncertaintyParams:
    sigma_px: float = 1.5      # pixel localisation noise (1 sigma) at confidence = 1
    d0_m: float = 15.0         # distance scale of the trust decay
    tau_s: float = 30.0        # data-age scale of the trust decay
    drift_mps: float = 0.02    # growth of positional doubt per second of age (m/s)
    dem_sigma_m: float = 0.15  # elevation error of the terrain model (1 sigma, m)


@dataclass(frozen=True)
class LinkParams:
    """Gilbert-Elliott burst-loss channel + latency + residual bit errors."""
    p_gb: float = 0.10         # P(Good -> Bad) per packet
    p_bg: float = 0.30         # P(Bad -> Good) per packet
    loss_good: float = 0.02    # packet-loss probability while in Good state
    loss_bad: float = 0.80     # packet-loss probability while in Bad state
    latency_s: float = 1.3
    jitter_s: float = 0.4
    ber: float = 1e-4          # bit-error rate of packets that do get through
