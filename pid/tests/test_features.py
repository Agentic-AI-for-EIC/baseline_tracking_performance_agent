"""Unit tests for the per-file feature builders (mocked ROOT reads)."""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from pid import features


class TestDrichTrackLink(unittest.TestCase):
    def test_has_flag_follows_length_not_link_row(self):
        # A track with a link row but no radiator segment has no pathlength:
        # has_drich must be 0, not 1.
        tr = pd.DataFrame({"event": [0, 0], "idx": [0, 1], "drich_gas_track": [3, 4]})
        seg = pd.DataFrame({"event": [0], "idx": [0], "drich_gas_length": [2.5]})

        def fake_flat(tree, campaign, keys):
            key = keys[0]
            if key == "drich_gas_track":
                return tr
            if key == "drich_gas_length":
                return seg
            return pd.DataFrame(columns=["event", "idx", key])

        with mock.patch.object(features, "_flat", side_effect=fake_flat):
            out = features.drich_track_link("P", "T", "C")
        by_track = out.set_index("track_idx")
        row = by_track.loc[4]
        self.assertEqual(int(row["has_drich_gas"]), 0)
        self.assertTrue(pd.isna(row["drich_gas_pathlength"]))
        row0 = by_track.loc[3]
        self.assertEqual(int(row0["has_drich_gas"]), 1)
        self.assertAlmostEqual(float(row0["drich_gas_pathlength"]), 2.5)


if __name__ == "__main__":
    unittest.main()
