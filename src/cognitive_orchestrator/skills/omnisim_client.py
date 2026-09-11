"""OmniSim skill client — the adapter between the orchestrator and a real OmniSim arm.

This is a THIRD implementation of the SkillClient protocol (see skills/client.py),
alongside LocalSkillClient and HTTPSkillClient. The orchestration graph is unchanged;
only the skill-execution seam is swapped.

WHAT THIS BRIDGES
-----------------
The orchestrator speaks in SYMBOLIC terms:
    skill = place, args = {"object_id": "block", "location": "left_bin"}
OmniSim speaks in PHYSICAL terms:
    POST /place  {"xyz": [0.24, -0.42, 0.20]}
So this adapter's core job is translating symbolic <-> physical:
  - outbound: symbolic location name  -> world xyz coordinates (LOCATIONS table)
  - inbound:  object world xyz         -> nearest symbolic location (for state())

VERIFIED ENDPOINT BEHAVIOUR (from the working bash probe against the 2f140 world)
--------------------------------------------------------------------------------
  GET  /protocol            -> {"ok": true, "robot_id": "omniarm6", ...}
  GET  /capabilities        -> [ {"id":"omniarm6","capabilities":{...,"has_gripper":true,...}} ]
  POST /get_robot_state {}  -> {"gripper":{"holding":bool,...}, "last_command":{...}, ...}
  POST /node_pose {"def":"GRASP_BLOCK"}
                            -> {"def":"GRASP_BLOCK","pos":[x,y,z]}   (object world position)
  POST /pick  {"object":"block"}
                            -> {"accepted":true,"holding":"GRASP_BLOCK","settled":true, ...}
                               (BLOCKS until motion settles; wait is built in, wait_max_s=50)
  POST /place {"xyz":[x,y,z]}
                            -> {"accepted":true,"released":"GRASP_BLOCK","settled":true, ...}

  Object identity: the graspable block has DEF "GRASP_BLOCK" (used by /node_pose),
  but /pick takes the friendlier {"object": "block"}. Keep both constants below.

SCOPE FOR THE FIRST PASS (agreed with OmniLink)
-----------------------------------------------
  * World: projects/samples/demos/worlds/chat/omnilink_omniarm6_2f140.omniworld
  * ONE object ("block" / DEF "GRASP_BLOCK"). Do NOT try to spawn 3 cubes yet.
  * Orchestrator stays in mocked mode (deterministic). This adapter is the only
    "real" component. Runs entirely on the laptop against localhost:8765.
"""
from __future__ import annotations

import math
from typing import Any

import httpx

from ..schemas import SkillName, SkillResult

# --- OmniSim object identity (single graspable block in the 2f140 world) ------
OBJECT_NAME = "block"          # what /pick expects
OBJECT_DEF = "GRASP_BLOCK"     # what /node_pose and state reports use

# --- Symbolic-location -> world-xyz table (THE core of the adapter) -----------
# TODO(verify): these coordinates are placeholders except 'left_bin', which the
# bash probe confirmed reachable at [0.24, -0.42, 0.20]. Pick the others inside
# the workspace envelope (min_radius 0.15, max_radius 0.95, min_z -0.1) and
# confirm each is reachable with a manual /place before trusting them.
LOCATIONS: dict[str, list[float]] = {
    "left_bin":  [0.24, -0.42, 0.20],   # verified reachable in the probe
    "right_bin": [0.46,  0.00, 0.20],   # TODO verify
    "safe_zone": [0.40,  0.00, 0.20],   # TODO verify
    "table":     [0.46,  0.00, 0.20],   # TODO verify (near the block's start pose)
}

# Max distance (metres) between an object's actual xyz and a LOCATIONS entry for
# state() to report the object as "in" that location. Tune against real runs.
LOCATION_SNAP_RADIUS_M = 0.12


class OmniSimSkillClient:
    """Implements the SkillClient protocol against a live OmniSim robot_bridge."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765", timeout: float = 60.0) -> None:
        # timeout is generous: /pick and /place BLOCK until motion settles (up to 50s).
        self.base_url = base_url.rstrip("/")
        self._c = httpx.Client(timeout=timeout)

    # -- low-level HTTP helpers -------------------------------------------------
    def _get(self, path: str) -> Any:
        r = self._c.get(f"{self.base_url}/{path}")
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> Any:
        r = self._c.post(f"{self.base_url}/{path}", json=body)
        r.raise_for_status()
        return r.json()

    # -- coordinate translation (the heart of the adapter) ---------------------
    def _location_to_xyz(self, location: str) -> list[float]:
        """Symbolic name -> world xyz. Raise on unknown location (fail loud)."""
        if location not in LOCATIONS:
            raise ValueError(
                f"Unknown location {location!r}; valid: {list(LOCATIONS)}"
            )
        return LOCATIONS[location]

    def _xyz_to_location(self, xyz: list[float]) -> str:
        """World xyz -> nearest symbolic location within LOCATION_SNAP_RADIUS_M.
                Returns 'table' if nothing is close enough.
        """
        nearest_name = "table"
        nearest_distance = float("inf")
        for name, loc_xyz in LOCATIONS.items():
            distance = math.dist(xyz, loc_xyz)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_name = name
        return nearest_name if nearest_distance <= LOCATION_SNAP_RADIUS_M else "table"

    # -- SkillClient protocol: seed / state / manifest / execute ---------------
    def seed(self, locations: dict[str, str], fragile: dict[str, bool]) -> dict:
        """Drive the real world back to a known starting state before each run.

        OmniSim has no object-reset / set-object-pose / spawn endpoint (confirmed
        via GET /capabilities: the only motion verbs are pick, place, set_tcp_target,
        set_tcp_pose, set_joint_positions, reset_to_home, wave, run_learned_skill --
        and GET /protocol's "extensions" list is empty, so there is nothing hidden
        beyond this). So "resetting" the block is done the same way any other move
        is done: a real /pick + /place back to its configured starting location.
        That also means seeding IS a genuine cross-bin pick-and-place, not a no-op,
        which is what makes repeated scored runs independent of each other.
        """
        self._post("reset_to_home", {})  # known arm posture before touching anything

        target = locations.get(OBJECT_NAME)
        if target:
            current = self.state()["locations"].get(OBJECT_NAME)
            if current != target:
                pick_resp = self._post("pick", {"object": OBJECT_NAME})
                if bool(pick_resp.get("accepted")) and bool(pick_resp.get("holding")):
                    xyz = self._location_to_xyz(target)
                    self._post("place", {"xyz": xyz})
                # If pick was rejected, fall through -- state() below will report
                # the block's real (unmoved) position rather than lying about it.

        self._post("reset_to_home", {})  # leave the arm clear for perceive/plan
        return self.state()

    def state(self) -> dict:
        """Return world state in the SAME shape the verifier expects from the
        local client: {"locations": {object_id: location}, "held": obj|None,
        "fragile": {object_id: bool}}.

        Keep object_id consistent: use OBJECT_NAME everywhere so the verifier and
        the mocked planner agree on the id.
        """
        robot = self._post("get_robot_state", {})
        pose = self._post("node_pose", {"def": OBJECT_DEF})
        pos = pose.get("pos")

        location = "table"
        if isinstance(pos, list) and len(pos) == 3:
            location = self._xyz_to_location(pos)

        holding = bool(robot.get("gripper", {}).get("holding"))
        held = OBJECT_NAME if holding else None

        return {
            "locations": {OBJECT_NAME: location},
            "held": held,
            "fragile": {OBJECT_NAME: False},
        }

    def manifest(self) -> dict[str, str]:
        """Return {skill_name: description} for the planner. Derive from
        GET /capabilities. At minimum advertise the three skills the orchestrator
                uses, confirming the arm supports them.
        """
        capabilities = self._get("capabilities")
        if not capabilities:
            raise ValueError("OmniSim returned empty capabilities list")

        caps = capabilities[0].get("capabilities", {})
        waitable_actions = set(caps.get("waitable_actions", []))
        if "pick" not in waitable_actions or "place" not in waitable_actions:
            raise ValueError(
                f"Robot missing required waitable actions; got {sorted(waitable_actions)}"
            )

        return {
            SkillName.PICK.value: "pick(object_id) -> grasp the configured block",
            SkillName.PLACE.value: "place(object_id, location) -> move held block to a symbolic location",
            SkillName.INSPECT.value: "inspect(object_id) -> report the block world position",
        }

    def execute(self, skill: SkillName, args: dict[str, str]) -> SkillResult:
        """Dispatch one skill to OmniSim and return a SkillResult.

        Map exactly like the working bash probe:

          PICK:    POST /pick {"object": OBJECT_NAME}
                   ok = response.get("accepted") and response.get("holding")
          PLACE:   xyz = self._location_to_xyz(args["location"])
                   POST /place {"xyz": xyz}
                   ok = response.get("accepted") and response.get("released") == OBJECT_DEF
          INSPECT: POST /node_pose {"def": OBJECT_DEF}
                   ok = "pos" in response

        For each, build SkillResult(ok=..., skill=skill, args=args, message=...)
        with a short human-readable message (e.g. f"picked {OBJECT_NAME}",
        f"placed at {args['location']} -> {xyz}", f"{OBJECT_DEF} at {pos}").

        Because /pick and /place BLOCK until settled, no polling loop is needed —
        the HTTP call returns only once motion is done. Trust 'settled'/'accepted'
        in the response for success rather than re-reading state.
        """
        if skill == SkillName.PICK:
            response = self._post("pick", {"object": OBJECT_NAME})
            ok = bool(response.get("accepted")) and bool(response.get("holding"))
            message = f"picked {OBJECT_NAME}" if ok else f"pick failed: {response}"
            return SkillResult(ok=ok, skill=skill, args=args, message=message)

        if skill == SkillName.PLACE:
            location = args.get("location", "")
            xyz = self._location_to_xyz(location)
            response = self._post("place", {"xyz": xyz})
            ok = bool(response.get("accepted")) and response.get("released") == OBJECT_DEF
            message = (
                f"placed at {location} -> {xyz}"
                if ok
                else f"place failed at {location} -> {xyz}: {response}"
            )
            return SkillResult(ok=ok, skill=skill, args=args, message=message)

        if skill == SkillName.INSPECT:
            response = self._post("node_pose", {"def": OBJECT_DEF})
            pos = response.get("pos")
            ok = isinstance(pos, list) and len(pos) == 3
            message = f"{OBJECT_DEF} at {pos}" if ok else f"inspect failed: {response}"
            return SkillResult(ok=ok, skill=skill, args=args, message=message)

        raise ValueError(f"Unsupported skill {skill!r}")


# --- standalone smoke test (run this BEFORE wiring into the graph) ------------
# Lets you verify the adapter against a live OmniSim exactly like the bash probe,
# with no orchestrator involved. Run: python -m cognitive_orchestrator.skills.omnisim_client
if __name__ == "__main__":
    client = OmniSimSkillClient()
    print("manifest:", client.manifest())
    print("state before:", client.state())
    print("pick:", client.execute(SkillName.PICK, {"object_id": OBJECT_NAME}))
    print("place:", client.execute(SkillName.PLACE, {"object_id": OBJECT_NAME, "location": "left_bin"}))
    print("state after:", client.state())

    print("state before:", client.state())
    print("pick:", client.execute(SkillName.PICK, {"object_id": OBJECT_NAME}))
    print("place:", client.execute(SkillName.PLACE, {"object_id": OBJECT_NAME, "location": "right_bin"}))
    print("state after:", client.state())