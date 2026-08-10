import unittest

import numpy as np

from pyjst import CartesianGrid, GasModel, GridSpec, JSTParameters
from pyjst.jst import jst_dissipation, pressure_sensors
from pyjst.state import primitive_to_conservative, uniform_conservative_state


class JSTTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gas = GasModel()
        self.grid = CartesianGrid(GridSpec(4, 4, 0.0, 1.0, 0.0, 1.0))
        self.parameters = JSTParameters()

    def test_uniform_state_has_zero_sensors_and_dissipation(self) -> None:
        state = uniform_conservative_state(self.grid.shape, self.gas, 1.0, 200.0, 0.0, 100000.0)
        sensor_x, sensor_y = pressure_sensors(state, self.grid, self.gas)
        np.testing.assert_array_equal(sensor_x, 0.0)
        np.testing.assert_array_equal(sensor_y, 0.0)
        np.testing.assert_array_equal(jst_dissipation(state, self.grid, self.gas, self.parameters), 0.0)

    def test_pressure_sensor_detects_a_pressure_peak(self) -> None:
        primitive = np.empty((*self.grid.shape, 4))
        primitive[..., 0] = 1.0
        primitive[..., 1] = 10.0
        primitive[..., 2] = 0.0
        primitive[..., 3] = 1.0
        primitive[3, :, 3] = 2.0
        state = primitive_to_conservative(primitive, self.gas)
        sensor_x, sensor_y = pressure_sensors(state, self.grid, self.gas)
        self.assertAlmostEqual(sensor_x[3, 3], 1.0 / 3.0)
        self.assertEqual(sensor_y[3, 3], 0.0)

    def test_nonuniform_state_produces_finite_dissipation(self) -> None:
        state = uniform_conservative_state(self.grid.shape, self.gas, 1.0, 100.0, 0.0, 100000.0)
        state[3, 3, 1] *= 1.05
        dissipation = jst_dissipation(state, self.grid, self.gas, self.parameters)
        sx, sy = self.grid.interior
        self.assertTrue(np.all(np.isfinite(dissipation)))
        self.assertGreater(np.max(np.abs(dissipation[sx, sy, :])), 0.0)


if __name__ == "__main__":
    unittest.main()
