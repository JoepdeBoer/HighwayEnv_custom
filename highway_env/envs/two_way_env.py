from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork


class TwoWayEnv(AbstractEnv):
    """
    A risk management task: the agent is driving on a two-way lane with icoming traffic.

    It must balance making progress by overtaking and ensuring safety.

    These conflicting objectives are implemented by a reward signal and a constraint signal,
    in the CMDP/BMDP framework.
    """


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "TimeToCollision", "horizon": 5},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "crash_reward": -10,
                "right_lane_reward": 0.2,
                "high_speed_reward": 0.8,
            }
        )
        return config

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        return sum(
            self.config.get(name, 0) * reward
            for name, reward in self._rewards(action).items()
        )

    def _rewards(self, action: int) -> dict[str, float]:
        "all rewards positive weigths can be negative"
        nr_lanes = len(self.road.network.all_side_lanes(self.vehicle.lane_index))+1
        if self._is_terminated() :
            return {"high_speed_reward": 0, "right_lane_reward": 0, "crash_reward":1}

        return {
            "high_speed_reward": self.vehicle.speed
            / (self.vehicle.MAX_SPEED),
            "right_lane_reward": (self.vehicle.lane_index[2] + 1)/(nr_lanes), # Index 2 is current lane index
            "crash_reward": 0
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        crashed = self.vehicle.crashed
        offroad = not self.vehicle.on_road

        return crashed or offroad
        return self.vehicle.crashed or not self.vehicle.on_road

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> np.ndarray:
        self._make_road()
        self._make_vehicles()

    def _make_road(self, length=800):
        """
        Make a road composed of a two-way road.

        :return: the road
        """
        net = RoadNetwork()

        # Lanes
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, 0],
                [length, 0],
                line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
            ),
        )
        net.add_lane(
            "a",
            "b",
            StraightLane(
                [0, StraightLane.DEFAULT_WIDTH],
                [length, StraightLane.DEFAULT_WIDTH],
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
            ),
        )
        net.add_lane(
            "b",
            "a",
            StraightLane(
                [length, 0], [0, 0], line_types=(LineType.NONE, LineType.NONE)
            ),
        )

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the road

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=30
        )
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(3):
            self.road.vehicles.append(
                vehicles_type(
                    road,
                    position=road.network.get_lane(("a", "b", 1)).position(
                        70 + 40 * i + 10 * self.np_random.normal(), 0
                    ),
                    heading=road.network.get_lane(("a", "b", 1)).heading_at(
                        70 + 40 * i
                    ),
                    speed=24 + 2 * self.np_random.normal(),
                    enable_lane_change=False,
                )
            )
        for i in range(2):
            v = vehicles_type(
                road,
                position=road.network.get_lane(("b", "a", 0)).position(
                    200 + 100 * i + 10 * self.np_random.normal(), 0
                ),
                heading=road.network.get_lane(("b", "a", 0)).heading_at(200 + 100 * i),
                speed=20 + 5 * self.np_random.normal(),
                enable_lane_change=False,
            )
            v.target_lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)
