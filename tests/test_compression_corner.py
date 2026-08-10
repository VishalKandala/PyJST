import unittest

import numpy as np

from pyjst import compression_corner_case, freestream_initial_state, weak_oblique_shock
from pyjst.boundary import apply_boundary_conditions
from pyjst.flux import central_residual
from pyjst.state import conservative_to_primitive


class CompressionCornerTests(unittest.TestCase):
    def test_mach_2_ten_degree_oblique_shock_reference(self) -> None:
        solution = weak_oblique_shock(2.0, 10.0)
        self.assertAlmostEqual(solution.shock_angle_degrees, 39.3139, places=3)
        self.assertAlmostEqual(solution.pressure_ratio, 1.7066, places=3)
        self.assertAlmostEqual(solution.downstream_mach, 1.6405, places=3)

    def test_baseline_case_generates_positive_body_fitted_cells(self) -> None:
        case = compression_corner_case()
        vertices = case.vertices()
        self.assertEqual(vertices.shape, (161, 81, 2))
        self.assertEqual(vertices[0, 0, 1], 0.0)
        self.assertGreater(vertices[-1, 0, 1], 0.0)
        lower_left = vertices[:-1, :-1, :]
        lower_right = vertices[1:, :-1, :]
        upper_left = vertices[:-1, 1:, :]
        signed_area_twice = (
            (lower_right[..., 0] - lower_left[..., 0]) * (upper_left[..., 1] - lower_left[..., 1])
            - (lower_right[..., 1] - lower_left[..., 1]) * (upper_left[..., 0] - lower_left[..., 0])
        )
        self.assertTrue(np.all(signed_area_twice > 0.0))

    def test_case_builds_a_solver_grid_and_state(self) -> None:
        case_definition = compression_corner_case()
        grid = case_definition.grid()
        solver_case = case_definition.solver_case()
        state = freestream_initial_state(grid, solver_case)
        self.assertEqual(state.shape, (164, 84, 4))

    def test_body_fitted_geometry_preserves_a_uniform_inviscid_state(self) -> None:
        case_definition = compression_corner_case()
        grid, solver_case = case_definition.grid(), case_definition.solver_case()
        residual = central_residual(freestream_initial_state(grid, solver_case), grid, solver_case.gas)
        sx, sy = grid.interior
        self.assertLess(np.max(np.abs(residual[sx, sy, :])), 2.0e-8)

    def test_lower_slip_wall_reflects_normal_velocity(self) -> None:
        case_definition = compression_corner_case()
        grid, solver_case = case_definition.grid(), case_definition.solver_case()
        state = freestream_initial_state(grid, solver_case)
        apply_boundary_conditions(state, grid, solver_case)
        sx, sy = grid.interior
        primitive = conservative_to_primitive(state, solver_case.gas)
        normal = grid.y_face_vectors[:, 0, :]
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        physical_normal_velocity = primitive[sx, sy.start, 1] * normal[:, 0] + primitive[sx, sy.start, 2] * normal[:, 1]
        ghost_normal_velocity = primitive[sx, sy.start - 1, 1] * normal[:, 0] + primitive[sx, sy.start - 1, 2] * normal[:, 1]
        np.testing.assert_allclose(ghost_normal_velocity, -physical_normal_velocity)

    def test_detached_case_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            weak_oblique_shock(2.0, 40.0)


if __name__ == "__main__":
    unittest.main()
