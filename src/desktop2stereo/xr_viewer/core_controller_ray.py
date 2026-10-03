from __future__ import annotations

import math

import numpy as np


class CoreControllerRayMixin:
    """Legacy controller-ray smoothing shared by the Vulkan presenter."""

    @staticmethod
    def _mat3_to_quat(matrix):
        trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 2] - matrix[2, 0]) / scale
            z = (matrix[1, 0] - matrix[0, 1]) / scale
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif matrix[1, 1] > matrix[2, 2]:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
        quaternion = np.array([x, y, z, w], dtype=np.float64)
        return quaternion / max(float(np.linalg.norm(quaternion)), 1e-10)

    @staticmethod
    def _quat_to_mat3(quaternion):
        x, y, z, w = quaternion
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _slerp_quat(first, second, amount):
        dot = float(np.dot(first, second))
        if dot < 0.0:
            second = -second
            dot = -dot
        if dot > 0.9995:
            result = first + amount * (second - first)
            return result / max(float(np.linalg.norm(result)), 1e-10)
        theta_zero = math.acos(min(dot, 1.0))
        theta = theta_zero * amount
        sin_theta = math.sin(theta)
        sin_theta_zero = math.sin(theta_zero)
        first_scale = math.cos(theta) - dot * sin_theta / sin_theta_zero
        second_scale = sin_theta / sin_theta_zero
        return first_scale * first + second_scale * second

    def _smooth_controller_poses(self) -> None:
        for hand, aim_matrix, grip_matrix in (
            (0, self._aim_mat_l, self._grip_mat_l),
            (1, self._aim_mat_r, self._grip_mat_r),
        ):
            if aim_matrix is None:
                continue
            if grip_matrix is not None:
                raw_position = (
                    grip_matrix[:3, 3] + grip_matrix[:3, 1] * 0.020
                ).astype(np.float64)
            else:
                raw_position = aim_matrix[:3, 3].astype(np.float64)
            self._apply_ray_smoothing(hand, raw_position, aim_matrix)

    def _apply_ray_smoothing(self, hand, raw_position, aim_matrix) -> None:
        position_attr = "_smooth_ray_origin_l" if hand == 0 else "_smooth_ray_origin_r"
        quaternion_attr = "_smooth_ray_quat_l" if hand == 0 else "_smooth_ray_quat_r"
        forward_attr = "_smooth_ray_fwd_l" if hand == 0 else "_smooth_ray_fwd_r"
        position_filter = self._ray_filter_l if hand == 0 else self._ray_filter_r
        previous_position = getattr(self, position_attr)
        previous_quaternion = getattr(self, quaternion_attr)
        if previous_position is None:
            position_filter.reset()
        smoothed_position = position_filter.filter(raw_position, self._last_frame_dt)
        raw_quaternion = self._mat3_to_quat(aim_matrix[:3, :3].astype(np.float64))
        if previous_quaternion is None:
            smoothed_quaternion = raw_quaternion
        else:
            dot = min(abs(float(np.dot(raw_quaternion, previous_quaternion))), 1.0)
            angle = 2.0 * math.acos(dot) if dot < 1.0 else 0.0
            if angle < self._ray_deadzone_rad:
                smoothed_quaternion = previous_quaternion
            else:
                adaptive = min(
                    self._rot_smooth * (1.0 + min(angle * 30.0, 2.0)),
                    0.30,
                )
                smoothed_quaternion = self._slerp_quat(
                    previous_quaternion, raw_quaternion, adaptive
                )
        x, y, z, w = smoothed_quaternion
        forward = np.array(
            [
                -(2.0 * x * z + 2.0 * w * y),
                -(2.0 * y * z - 2.0 * w * x),
                -(1.0 - 2.0 * x * x - 2.0 * y * y),
            ],
            dtype=np.float64,
        )
        setattr(self, position_attr, smoothed_position.copy())
        setattr(self, quaternion_attr, smoothed_quaternion.copy())
        setattr(self, forward_attr, forward)

    def _get_smoothed_ray(self, hand):
        position = self._smooth_ray_origin_l if hand == 0 else self._smooth_ray_origin_r
        forward = self._smooth_ray_fwd_l if hand == 0 else self._smooth_ray_fwd_r
        if position is None or forward is None:
            return None, None
        return position.copy(), forward.copy()

    def _reset_smoothed_ray(self, hand) -> None:
        suffix = "l" if hand == 0 else "r"
        setattr(self, f"_smooth_ray_origin_{suffix}", None)
        setattr(self, f"_smooth_ray_quat_{suffix}", None)
        setattr(self, f"_smooth_ray_fwd_{suffix}", None)
        (self._ray_filter_l if hand == 0 else self._ray_filter_r).reset()
