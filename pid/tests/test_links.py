"""Tests for the podio offset/relation arithmetic in :mod:`pid.links`.

This is the riskiest code in the pipeline: getting an event boundary wrong does
not raise, it silently returns another event's hits, and the shower shapes still
look plausible. So it is tested against a hand-built example whose correct answer
is known by construction, and against the slow reference implementation
(:func:`trkperf.io.expand_by_offsets`).
"""

from __future__ import annotations

import unittest

import awkward as ak
import numpy as np
import pandas as pd

from trkperf import io

from pid import links


def _two_event_fixture():
    """Two events; event 0 has 2 parents / 5 values, event 1 has 1 parent / 4 values.

    Offsets restart per event (that is the podio convention), so the flattened
    value array is [e0v0..e0v4, e1v0..e1v3] while the per-parent ranges are
    [[0,2),[2,5)] in event 0 and [[0,4)] in event 1.
    """
    values = ak.Array([[10.0, 11.0, 12.0, 13.0, 14.0], [20.0, 21.0, 22.0, 23.0]])
    begin = ak.Array([[0, 2], [0]])
    end = ak.Array([[2, 5], [4]])
    parent_event = np.array([0, 0, 1], dtype=np.int64)
    return values, begin, end, parent_event


class TestOffsetExpansion(unittest.TestCase):
    def test_inline_vector_values_are_gathered_per_object(self):
        values, begin, end, pev = _two_event_fixture()
        flat, counts = links.per_event_flat(values)
        row, got = links.inline_vector(flat, counts, pev,
                                       ak.to_numpy(ak.flatten(begin)),
                                       ak.to_numpy(ak.flatten(end)))
        self.assertEqual(np.asarray(row).tolist(), [0, 0, 1, 1, 1, 2, 2, 2, 2])
        # event 0: parent 0 -> [10,11], parent 1 -> [12,13,14]; event 1: parent -> [20..23]
        np.testing.assert_allclose(got, [10, 11, 12, 13, 14, 20, 21, 22, 23])

    def test_matches_the_reference_loop_implementation(self):
        values, begin, end, pev = _two_event_fixture()
        ref = io.expand_by_offsets(values, begin, end).to_list()
        flat, counts = links.per_event_flat(values)
        row, got = links.inline_vector(flat, counts, pev,
                                       ak.to_numpy(ak.flatten(begin)),
                                       ak.to_numpy(ak.flatten(end)))
        rebuilt = []
        for parent in range(int(row.max()) + 1 if row.size else 0):
            rebuilt.append(got[row == parent].tolist())
        flat_ref = [obj for ev in ref for obj in ev]
        self.assertEqual(rebuilt, flat_ref)

    def test_per_event_bases_are_not_confused(self):
        """The classic bug: forgetting that offsets restart every event."""
        values, begin, end, pev = _two_event_fixture()
        flat, counts = links.per_event_flat(values)
        row, got = links.inline_vector(flat, counts, pev,
                                       ak.to_numpy(ak.flatten(begin)),
                                       ak.to_numpy(ak.flatten(end)))
        second_event_values = got[row == 2]
        self.assertEqual(list(second_event_values), [20.0, 21.0, 22.0, 23.0])
        self.assertNotIn(10.0, second_event_values)

    def test_as_matrix_pads_missing_entries_with_nan_not_zero(self):
        row = np.array([0, 0, 1, 2], dtype=np.int64)
        vals = np.array([5.0, 6.0, 7.0, 8.0])
        M = links.as_matrix(row, vals, 3, 3)
        self.assertTrue(np.isnan(M[1, 2]))   # parent 1 had one value only
        self.assertTrue(np.isnan(M[2, 1:]).all())
        self.assertEqual(M[0, 0], 5.0)
        self.assertFalse(np.any(M == 0.0))   # never a fabricated zero

    def test_empty_ranges_stay_empty(self):
        values = ak.Array([[1.0, 2.0], []])
        begin = ak.Array([[0, 2], []])
        end = ak.Array([[2, 2], []])
        flat, counts = links.per_event_flat(values)
        row, got = links.inline_vector(flat, counts, np.array([0, 0], dtype=np.int64),
                                       ak.to_numpy(ak.flatten(begin)),
                                       ak.to_numpy(ak.flatten(end)))
        # parent 1 has an empty range -> contributes no values
        self.assertEqual(np.asarray(row).tolist(), [0, 0])
        self.assertEqual(np.asarray(got).tolist(), [1.0, 2.0])


class TestRelationGathering(unittest.TestCase):
    def setUp(self):
        # clusters -> hit indices, two events, per-event hit collections of
        # different length (the case that breaks naive arithmetic).
        self.clusters = pd.DataFrame({
            "file_id": [0, 0, 0],
            "event": [0, 0, 1],
            "idx": [0, 1, 0],
            "hb": [0, 3, 0],
            "he": [3, 5, 2],
        })
        self.hits = pd.DataFrame({
            "file_id": [0] * 7,
            "event": [0, 0, 0, 0, 0, 1, 1],
            "idx": [0, 1, 2, 3, 4, 0, 1],
            "energy": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 200.0],
        })
        # Relation contents are per-event LOCAL indices into the hit collection,
        # so event 1 refers to its own hits 0 and 1, not 5 and 6.
        self.rel = ak.Array([[0, 1, 2, 3, 4], [0, 1]])

    def _gather(self):
        flat, counts = links.per_event_flat(self.rel)
        row, local = links.relation_targets(
            flat, counts, self.clusters["event"].to_numpy(),
            self.clusters["hb"].to_numpy(), self.clusters["he"].to_numpy())
        tab = links.gather_target(row, local,
                                  self.clusters["event"].to_numpy()[row],
                                  self.hits, ["energy"])
        return row, tab

    def test_hit_energies_map_to_the_right_cluster_and_event(self):
        row, tab = self._gather()
        sums = np.bincount(row, weights=tab["energy"].to_numpy(), minlength=len(self.clusters))
        np.testing.assert_allclose(sums, [1 + 2 + 3, 4 + 5, 100 + 200])

    def test_out_of_range_relation_index_raises_instead_of_clipping(self):
        # A hit index beyond its own event's collection size must stop the run:
        # silently reading another event's hit is exactly how a shower shape
        # becomes plausible-but-wrong.
        rel = ak.Array([[0, 1, 2, 3, 4], [0, 9]])
        flat, counts = links.per_event_flat(rel)
        row, local = links.relation_targets(flat, counts,
                                           self.clusters["event"].to_numpy(),
                                           self.clusters["hb"].to_numpy(),
                                           self.clusters["he"].to_numpy())
        with self.assertRaises(ValueError):
            links.gather_target(row, local, self.clusters["event"].to_numpy()[row],
                                self.hits, ["energy"])

    def test_inconsistent_offsets_raise_instead_of_reading_past_the_array(self):
        bad = self.clusters.copy()
        bad.loc[2, "he"] = 9  # claims 9 hits for event 1, which has 2
        flat, counts = links.per_event_flat(self.rel)
        with self.assertRaises(ValueError):
            links.relation_targets(flat, counts, bad["event"].to_numpy(),
                                   bad["hb"].to_numpy(), bad["he"].to_numpy())


class TestGeometry(unittest.TestCase):
    def test_delta_r_is_zero_for_identical_directions(self):
        self.assertAlmostEqual(float(links.delta_r(1.0, 0.5, 1.0, 0.5)), 0.0)

    def test_delta_r_wraps_phi_across_the_branch_cut(self):
        near_pi = links.delta_r(1.0, np.pi - 0.01, 1.0, -np.pi + 0.01)
        self.assertLess(float(near_pi), 0.03)

    def test_theta_phi_from_position_roundtrip(self):
        theta, phi = links.theta_phi_from_position(np.array([1.0]), np.array([0.0]),
                                                  np.array([1.0]))
        self.assertAlmostEqual(float(theta[0]), np.pi / 4, places=6)
        self.assertAlmostEqual(float(phi[0]), 0.0, places=6)
        backward, _ = links.theta_phi_from_position(np.array([0.0]), np.array([0.0]),
                                                   np.array([-2.0]))
        self.assertGreater(float(backward[0]), np.pi / 2)  # z<0 -> theta>90 deg

    def test_velocity_is_bounded(self):
        b = links.velocity(np.array([0.0, 1.0, 1000.0]))
        self.assertTrue(np.all(b <= 1.0) and np.all(b >= 0.0))
        self.assertTrue(np.isnan(b[0]) or b[0] == 0.0)


class TestMatchTracksToClusters(unittest.TestCase):
    def _fixture(self):
        tracks = pd.DataFrame({"event": [0, 0, 0], "idx": [0, 1, 2],
                               "theta": [0.5, 2.6, 1.0], "phi": [0.1, -3.0, 2.0]})
        clusters = pd.DataFrame({"event": [0, 0], "idx": [0, 1], "E": [3.0, 7.0],
                                 "theta": [0.52, 2.61], "phi": [0.11, -3.02]})
        return tracks, clusters

    def test_geometric_fallback_matches_within_the_window(self):
        tracks, clusters = self._fixture()
        out = links.match_tracks_to_clusters(
            tracks, clusters, pd.DataFrame(columns=["event", "track_idx", "cluster_idx"]),
            detector="ecal", track_cols=("theta", "phi"), energy_col="E")
        self.assertEqual(int(out["ecal_matched"].sum()), 2)
        self.assertTrue((out.loc[out["ecal_matched"], "ecal_source"] == "geometric").all())
        self.assertAlmostEqual(float(out.loc[0, "ecal_E"]), 3.0)

    def test_tracks_outside_the_window_are_left_unmatched_not_forced(self):
        tracks, clusters = self._fixture()
        out = links.match_tracks_to_clusters(
            tracks, clusters, None, detector="ecal", delta_r_max=0.01,
            track_cols=("theta", "phi"), energy_col="E")
        self.assertEqual(int(out["ecal_matched"].sum()), 0)
        self.assertTrue(out["ecal_E"].isna().all())

    def test_one_cluster_per_track_and_never_two_tracks_on_one_cluster_by_default(self):
        # Two tracks 0.001 apart facing a single cluster: each takes the nearest,
        # and no track may receive two clusters.
        tracks, clusters = self._fixture()
        out = links.match_tracks_to_clusters(
            tracks, clusters, None, detector="ecal", track_cols=("theta", "phi"),
            energy_col="E")
        self.assertLessEqual(int(out["ecal_matched"].sum()), len(tracks))
        self.assertFalse(out.duplicated(subset=["event", "idx"]).any())

    def test_builtin_pairs_are_preferred_and_labelled(self):
        tracks, clusters = self._fixture()
        pairs = pd.DataFrame({"event": [0], "track_idx": [0], "cluster_idx": [1]})
        out = links.match_tracks_to_clusters(tracks, clusters, pairs, detector="ecal",
                                            track_cols=("theta", "phi"), energy_col="E")
        # pair (0 -> cluster 1) is far outside the window, so geometry wins
        self.assertEqual(out.loc[0, "ecal_source"], "geometric")
        near = pd.DataFrame({"event": [0], "track_idx": [1], "cluster_idx": [1]})
        out2 = links.match_tracks_to_clusters(tracks, clusters, near, detector="ecal",
                                             track_cols=("theta", "phi"), energy_col="E")
        self.assertEqual(out2.loc[1, "ecal_source"], "built_in")
        self.assertAlmostEqual(float(out2.loc[1, "ecal_E"]), 7.0)


    def test_winner_cluster_index_is_reported(self):
        tracks, clusters = self._fixture()
        out = links.match_tracks_to_clusters(
            tracks, clusters, pd.DataFrame(columns=["event", "track_idx", "cluster_idx"]),
            detector="ecal", track_cols=("theta", "phi"), energy_col="E")
        # track 0 takes cluster 0 (E = 3.0); track 2 is outside the window.
        self.assertEqual(int(out.loc[0, "ecal_cluster_idx"]), 0)
        self.assertTrue(pd.isna(out.loc[2, "ecal_cluster_idx"]))
        self.assertTrue(out.loc[out["ecal_matched"], "ecal_cluster_idx"].notna().all())


if __name__ == "__main__":
    unittest.main()
