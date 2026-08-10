import unittest

from pyjst.performance import benchmark_case, benchmark_suite


class PerformanceTests(unittest.TestCase):
    def test_benchmark_case_reports_positive_metrics(self) -> None:
        sample = benchmark_case("straight-channel", 8, repeats=1, warmup_iterations=0)

        self.assertEqual((sample.case, sample.nx, sample.ny), ("straight-channel", 8, 8))
        self.assertGreater(sample.seconds_per_iteration, 0.0)
        self.assertGreater(sample.cell_updates_per_second, 0.0)

    def test_benchmark_suite_includes_both_cases(self) -> None:
        samples = benchmark_suite((8,), repeats=1, warmup_iterations=0)

        self.assertEqual([sample.case for sample in samples], ["straight-channel", "compression-corner"])
