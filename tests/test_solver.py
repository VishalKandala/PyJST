import unittest
from dataclasses import replace

import numpy as np

from pyjst import CartesianGrid, freestream_initial_state, iteration_cfl, local_time_steps, solve, uniform_supersonic_case
from pyjst.config import JSTParameters


class SolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = uniform_supersonic_case()
        self.grid = CartesianGrid(self.case.grid)

    def test_local_time_steps_are_positive_and_uniform_for_uniform_flow(self) -> None:
        state = freestream_initial_state(self.grid, self.case)
        time_step = local_time_steps(state, self.grid, self.case)
        self.assertEqual(time_step.shape, (self.case.grid.nx, self.case.grid.ny))
        self.assertTrue(np.all(time_step > 0.0))
        np.testing.assert_array_equal(time_step, time_step[0, 0])

    def test_uniform_supersonic_flow_is_preserved_exactly(self) -> None:
        initial = freestream_initial_state(self.grid, self.case)
        result = solve(initial, self.grid, self.case)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        np.testing.assert_array_equal(result.residual_history, [0.0])
        np.testing.assert_array_equal(result.absolute_residual_history, [0.0])
        np.testing.assert_array_equal(result.conservative, initial)

    def test_cfl_ramp_reaches_the_configured_limit(self) -> None:
        case = replace(self.case, numerics=JSTParameters(cfl=0.5, cfl_initial=0.1, cfl_ramp_iterations=4))
        self.assertEqual(iteration_cfl(case, 1), 0.1)
        self.assertAlmostEqual(iteration_cfl(case, 3), 0.3)
        self.assertEqual(iteration_cfl(case, 5), 0.5)
        self.assertEqual(iteration_cfl(case, 100), 0.5)


if __name__ == "__main__":
    unittest.main()
