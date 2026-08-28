"""Scene generator.

Two backends:

  * render_synthetic()  -- pure Pillow. No GPU, no physics engine, always works.
                           Draws colored blocks + a glass so the VLM has a real
                           image to read. Used for local dev and CI.

  * render_pybullet()   -- documented below. On your A40 node, swap this in for a
                           genuine robotics simulator: it loads URDFs, renders from
                           a virtual camera, and lets skills physically move objects
                           so re-perception after execution reflects real change.

Both return (png_bytes, seed) where `seed` is the ground-truth world used to
initialise the edge skill-service.
"""
from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw


def render_synthetic() -> tuple[bytes, dict]:
    img = Image.new("RGB", (512, 384), (235, 232, 225))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 300, 512, 384], fill=(200, 190, 170))            # table edge
    d.rectangle([60, 200, 120, 260], fill=(200, 40, 40))             # red cube
    d.rectangle([200, 210, 260, 270], fill=(40, 80, 200))            # blue cube
    d.ellipse([360, 190, 410, 280], outline=(150, 200, 230), width=4)  # glass
    d.rectangle([20, 320, 90, 370], outline=(120, 90, 60), width=3)   # left bin
    d.rectangle([420, 320, 490, 370], outline=(120, 90, 60), width=3)  # right bin

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    seed = {
        "locations": {"red_cube_1": "table", "blue_cube_1": "table", "glass_1": "table"},
        "fragile": {"red_cube_1": False, "blue_cube_1": False, "glass_1": True},
    }
    return buf.getvalue(), seed


def to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode()


# --------------------------------------------------------------------------- #
# PyBullet upgrade (run on the A40 node). Kept as a reference implementation.
# --------------------------------------------------------------------------- #
def render_pybullet() -> tuple[bytes, dict]:  # pragma: no cover
    """Genuine robotics sim. Requires `pip install pybullet`.

    import pybullet as p, pybullet_data, numpy as np
    p.connect(p.DIRECT)                       # headless on the cluster
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.loadURDF("cube_small.urdf", [0.1, 0.0, 0.02])   # ... place objects
    view = p.computeViewMatrix([0.5, 0.5, 0.6], [0.1, 0.0, 0.0], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(60, 512/384, 0.1, 3.0)
    _, _, rgba, _, _ = p.getCameraImage(512, 384, view, proj)
    # rgba -> PNG bytes, build `seed` from the loaded object poses, return both.

    The key win: after a skill physically moves a body, re-render + re-perceive
    and the verifier sees the actual new world -- the cognitive loop closes on
    real sensor data, not on a bookkeeping variable.
    """
    raise NotImplementedError("Fill in on the A40 node; see docstring.")
