"""Tests for the middleware --host bind flag (main.build_parser)."""

import unittest

from main import build_parser


class TestBindMode(unittest.TestCase):
    def test_default_host_is_all_interfaces(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.host, "0.0.0.0")

    def test_explicit_localhost(self):
        args = build_parser().parse_args(["--host", "127.0.0.1"])
        self.assertEqual(args.host, "127.0.0.1")

    def test_explicit_all_interfaces(self):
        args = build_parser().parse_args(["--host", "0.0.0.0"])
        self.assertEqual(args.host, "0.0.0.0")

    def test_port_default_unchanged(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.port, 8000)

    def test_personality_default_unchanged(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.personality, "benson")


if __name__ == "__main__":
    unittest.main()
