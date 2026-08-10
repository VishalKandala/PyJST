import unittest

import numpy as np

from pyjst import CartesianGrid, freestream_initial_state, uniform_supersonic_case
from pyjst.postprocess import mach_number, physical_primitive, pressure_ratio


class PostprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = uniform_supersonic_case()
        self.grid = CartesianGrid(self.case.grid)
        self.state = freestream_initial_state(self.grid, self.case)

    def test_uniform_solution_fields_match_the_freestream(self) -> None:
        primitive = physical_primitive(self.state, self.grid, self.case)

        self.assertEqual(primitive.shape, (self.case.grid.nx, self.case.grid.ny, 4))
        np.testing.assert_allclose(pressure_ratio(self.state, self.grid, self.case), 1.0)
        np.testing.assert_allclose(mach_number(self.state, self.grid, self.case), self.case.freestream.mach)

    def test_physical_primitive_rejects_an_incorrect_state_shape(self) -> None:
        with self.assertRaises(ValueError):
            physical_primitive(self.state[1:], self.grid, self.case)
