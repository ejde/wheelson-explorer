"""Tests for the house map artifact (middleware/map/house_map.json)."""

import unittest

from house_map import load_map


class TestHouseMap(unittest.TestCase):
    def test_loads_and_validates(self):
        data = load_map()
        self.assertEqual(data["version"], 1)

    def test_seven_rooms(self):
        data = load_map()
        names = sorted(r["name"] for r in data["rooms"])
        self.assertEqual(names, sorted([
            "Kitchen", "Leo's Room", "Dining Area", "Main Bath",
            "Simon's Room", "Living Room", "Room",
        ]))

    def test_grid_shape_and_codes(self):
        data = load_map()
        grid = data["grid"]
        self.assertEqual(len(grid["cells"]), grid["rows"])
        self.assertTrue(all(len(row) == grid["cols"]
                            for row in grid["cells"]))
        flat = "".join(grid["cells"])
        for room in data["rooms"]:
            self.assertIn(str(room["id"]), flat)

    def test_scale_and_pose_pending(self):
        # Phase 1: scale awaits one real-world measurement, and the pose
        # awaits Phase 2 localization. Neither is guessed.
        data = load_map()
        self.assertIsNone(data["grid"]["cell_meters"])
        self.assertIsNone(data["pose"])


if __name__ == "__main__":
    unittest.main()
