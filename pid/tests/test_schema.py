"""Schema and environment tests - network-free, run on any machine."""

from __future__ import annotations

import unittest

from pid import config, schema


class TestMlEnvironment(unittest.TestCase):
    def test_ml_stack_is_importable_and_reported(self):
        env = schema.assert_ml_env(strict=True)
        for pkg in schema.ML_REQUIREMENTS:
            self.assertIn(pkg, env)
            self.assertNotEqual(env[pkg], "missing")

    def test_numpy_shadow_is_visible_in_provenance(self):
        # The ML package directory ships its own numpy; if that ever changes the
        # fits change with it, so the resolved path must be reported.
        env = schema.assert_ml_env(strict=False)
        self.assertTrue(env.get("numpy_path"))

    def test_coexists_with_root_and_trkperf(self):
        import ROOT  # noqa: F401
        from trkperf import binning, io, matching, reco, report, truth  # noqa: F401


class TestBranchTable(unittest.TestCase):
    def test_every_branch_has_a_campaign(self):
        for logical, branch in schema.BRANCHES.items():
            self.assertTrue(branch.campaigns, f"{logical} maps to no campaign")
            for campaign in branch.campaigns:
                self.assertIn(campaign, set(config.CAMPAIGN_BY_DATASET_TAG.values()))
                self.assertTrue(branch.campaigns[campaign])

    def test_logical_name_matches_the_registry_key(self):
        for logical, branch in schema.BRANCHES.items():
            self.assertEqual(branch.logical or logical, logical)

    def test_dataset_tag_maps_to_a_known_campaign(self):
        self.assertEqual(schema.campaign_of("clean"), "26.02.0")
        self.assertEqual(schema.campaign_of("bkg_mixed"), "26.07.1")
        with self.assertRaises(ValueError):
            schema.campaign_of("nonsense")

    def test_campaign_specific_branches_are_marked_not_assumed(self):
        # Fields that only exist in one production must be listed with one key,
        # so the feature builder can never quietly differ between the two runs.
        self.assertEqual(set(schema.BRANCHES["ecal_P_radius"].campaigns), {"26.07.1"})
        self.assertEqual(set(schema.BRANCHES["tof_endcap_time"].campaigns), {"26.07.1"})
        self.assertIn("26.02.0", schema.BRANCHES["tof_barrel_time"].campaigns)
        self.assertIn("26.07.1", schema.BRANCHES["tof_barrel_time"].campaigns)


class TestDeadLinks(unittest.TestCase):
    def test_dead_links_are_documented_with_evidence(self):
        self.assertGreaterEqual(len(schema.DEAD_LINKS), 8)
        for link in schema.DEAD_LINKS:
            self.assertTrue(link.reason.strip(), link.name)
            self.assertTrue(link.evidence.strip(),
                            f"{link.name}: a dead end must record what was measured")

    def test_producer_pid_likelihoods_are_declared_unusable(self):
        names = {d.name for d in schema.DEAD_LINKS}
        for expected in ("DIRCParticleIDs", "CombinedTOFParticleIDs",
                         "DRICHParticleIDs", "RICHEndcapNParticleIDs"):
            self.assertIn(expected, names)

    def test_dead_link_report_renders(self):
        text = schema.dead_link_report()
        self.assertIn("dangling", text.lower())


class TestConfigConsistency(unittest.TestCase):
    def test_tasks_reference_known_legs_and_classes(self):
        for name, spec in config.TASKS.items():
            self.assertIn(spec["leg"], config.LEGS, name)
            for cls in spec["classes"]:
                self.assertTrue(cls == "hadron" or cls in config.CLASS_LABELS,
                                f"{name}: unknown class {cls}")

    def test_legs_partition_eta(self):
        self.assertLess(config.BACKWARD_ETA_MAX, config.FORWARD_ETA_MIN)
        for leg in ("backward", "forward"):
            spec = config.LEGS[leg]
            self.assertLess(spec["eta_min"], spec["eta_max"], leg)

    def test_pid_reuses_trkperf_conventions(self):
        from trkperf import config as tk

        self.assertEqual(config.MATCH_WEIGHT_THRESHOLD, tk.MATCH_WEIGHT_THRESHOLD)
        self.assertEqual(config.MIN_ENTRIES_PER_BIN, tk.MIN_ENTRIES_PER_BIN)
        self.assertIs(config.PT_BIN_EDGES, tk.PT_BIN_EDGES)

    def test_leakage_blocked_families_cover_the_truth_side_dedx(self):
        # The truth-hit eDep/pathLength proxy is available in the files; it is
        # blocked because it is derived from the label's own particle.
        self.assertIn("truth_dedx", config.LEAKAGE_BLOCKED_FAMILIES)


if __name__ == "__main__":
    unittest.main()
