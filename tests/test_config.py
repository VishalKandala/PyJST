import unittest

from pyjst.config import GasModel, GridSpec, uniform_supersonic_case


class ConfigTests(unittest.TestCase):
    def test_freestream_converts_mach_to_velocity(self) -> None:
        gas = GasModel()
        rho, u, v, pressure = uniform_supersonic_case().freestream.primitive(gas)
        self.assertEqual(rho, 1.225)
        self.assertEqual(pressure, 101_325.0)
        self.assertGreater(u, 0.0)
        self.assertAlmostEqual(v, 0.0)

    def test_grid_requires_ghost_cells_for_jst_stencil(self) -> None:
        with self.assertRaises(ValueError):
            GridSpec(8, 8, 0.0, 1.0, 0.0, 1.0, ghost_cells=1)

    def test_case_has_expected_storage_shape(self) -> None:
        case = uniform_supersonic_case()
        self.assertEqual(case.grid.cell_shape, (68, 36))


if __name__ == "__main__":
    unittest.main()
