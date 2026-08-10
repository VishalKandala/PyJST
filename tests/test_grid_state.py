import unittest

import numpy as np

from pyjst import CartesianGrid, GasModel, GridSpec
from pyjst.state import conservative_to_primitive, primitive_to_conservative, uniform_conservative_state


class GridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = CartesianGrid(GridSpec(4, 3, 0.0, 2.0, -1.0, 2.0))

    def test_shape_interior_and_volume(self) -> None:
        self.assertEqual(self.grid.shape, (8, 7))
        self.assertEqual(self.grid.interior, (slice(2, 6), slice(2, 5)))
        self.assertEqual(self.grid.cell_volume, 0.5)

    def test_physical_and_ghost_centres(self) -> None:
        x, y = self.grid.cell_centres()
        self.assertEqual(x.shape, (4, 3))
        self.assertAlmostEqual(x[0, 0], 0.25)
        self.assertAlmostEqual(y[0, 0], -0.5)
        x_all, y_all = self.grid.cell_centres(include_ghosts=True)
        self.assertEqual(x_all.shape, self.grid.shape)
        self.assertAlmostEqual(x_all[0, 0], -0.75)
        self.assertAlmostEqual(y_all[0, 0], -2.5)


class StateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gas = GasModel()

    def test_round_trip_preserves_multiple_states(self) -> None:
        primitive = np.array([[[1.225, 50.0, -2.0, 101325.0], [0.8, -10.0, 20.0, 50000.0]]])
        conservative = primitive_to_conservative(primitive, self.gas)
        np.testing.assert_allclose(conservative_to_primitive(conservative, self.gas), primitive)

    def test_uniform_state_energy_is_correct(self) -> None:
        state = uniform_conservative_state((2, 3), self.gas, 1.0, 3.0, 4.0, 10.0)
        self.assertEqual(state.shape, (2, 3, 4))
        self.assertAlmostEqual(state[0, 0, 3], 10.0 / 0.4 + 12.5)

    def test_rejects_nonphysical_conservative_state(self) -> None:
        invalid = np.array([1.0, 0.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            conservative_to_primitive(invalid, self.gas)


if __name__ == "__main__":
    unittest.main()
