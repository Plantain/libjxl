"""Shared-library integration tests. Run with PYTHONPATH=tools/hybrid."""
import gc
import os
from pathlib import Path
import unittest

from jxl_hybrid import Encoder, StageOverrides, encode, encode_batch, record, scoreboard_encode


class WrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("JXL_HYBRID_TEST_TILE")
        cls.rgb = Path(path).read_bytes() if path else bytes((i * 13 + i // 1536) % 256 for i in range(512 * 512 * 3))
        cls.enc = Encoder(threads=4)
        cls.dump = cls.enc.record(cls.rgb)

    @classmethod
    def tearDownClass(cls):
        cls.dump.close()
        cls.enc.close()

    def test_record_and_replay(self):
        self.assertEqual(self.enc.encode(self.rgb, self.dump), self.dump.codestream)
        self.assertEqual(self.enc.encode(self.rgb, self.dump.decisions, opsin=self.dump.opsin), self.dump.codestream)

    def test_stages(self):
        for stage in range(1, 7):
            with self.subTest(stage=stage):
                self.assertEqual(self.enc.encode(self.rgb, overrides=self.dump.overrides([stage])), self.dump.codestream)

    def test_batch(self):
        self.assertEqual(self.enc.encode_batch([self.rgb] * 19, [self.dump] * 19), [self.dump.codestream] * 19)
        self.assertEqual(self.enc.encode_batch([]), [])
        self.assertEqual(self.enc.encode_batch([self.rgb], [self.dump], opsin=[self.dump.opsin]), [self.dump.codestream])

    def test_batch_errors_and_reuse(self):
        bad = StageOverrides(stages=128)
        with self.assertRaises(ValueError):
            self.enc.encode_batch([self.rgb] * 2, overrides=[None, bad])
        self.assertEqual(self.enc.encode(self.rgb, self.dump), self.dump.codestream)
        with self.assertRaises(ValueError):
            self.enc.encode_batch([self.rgb], [])

    def test_borrowed_lifetime(self):
        dump = self.enc.record(self.rgb)
        decisions = dump.decisions
        del dump
        gc.collect()
        self.assertEqual(self.enc.encode(self.rgb, decisions), self.dump.codestream)
        decisions._owner.close()
        with self.assertRaises(ValueError):
            self.enc.encode(self.rgb, decisions)

    def test_invalid_inputs(self):
        for rgb in (b"short", memoryview(self.rgb)[::2]):
            with self.assertRaises(ValueError):
                self.enc.encode(rgb)
        for options in ({"threads": 0}, {"threads": 17}, {"distance": float("nan")}, {"effort": 6}):
            with self.assertRaises(ValueError):
                Encoder(**options)
        with self.assertRaises(ValueError):
            self.dump.overrides([0])
        with self.assertRaises(ValueError):
            self.enc.encode(self.rgb, self.dump, self.dump.overrides([1]))

    def test_closed_encoder(self):
        enc = Encoder(threads=1)
        enc.close()
        enc.close()
        with self.assertRaises(ValueError):
            enc.encode(self.rgb)

    def test_module_api(self):
        with record(self.rgb) as dump:
            self.assertEqual(encode(self.rgb, dump), dump.codestream)
            self.assertEqual(encode_batch([self.rgb], [dump]), [dump.codestream])
            self.assertEqual(scoreboard_encode(self.rgb, 1.2), dump.codestream)


if __name__ == "__main__":
    unittest.main()
