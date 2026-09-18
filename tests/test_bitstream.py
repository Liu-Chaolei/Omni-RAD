"""Container regressions require only Python's standard library."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bitstream import read_bitstream, write_bitstream


class BitstreamTests(unittest.TestCase):
    def test_roundtrip_preserves_decode_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.ord'
            output = {'shape': (3, 4), 'lmbda_val': [2.0], 'strings': [[b'latent'], [b'hyper']]}
            write_bitstream(path, output, 321, 511)
            payload, shape, original = read_bitstream(path)
            self.assertEqual(payload['strings'], output['strings'])
            self.assertEqual(payload['lmbda_val'], [2.0])
            self.assertEqual(shape, (3, 4))
            self.assertEqual(original, (321, 511))
            self.assertNotIn('t_star_val', payload)

    def test_corrupt_streams_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.ord'
            write_bitstream(path, {'shape': (1, 1), 'lmbda_val': [0.5], 'strings': [[b'y'], [b'z']]}, 1, 2)
            valid = path.read_bytes()
            for invalid in [b'', valid[:-1], b'BAD!' + valid[4:], valid + b'junk']:
                with self.subTest(data=invalid):
                    path.write_bytes(invalid)
                    with self.assertRaises(ValueError):
                        read_bitstream(path)

    def test_batch_encoding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_bitstream(Path(directory) / 'bad.ord', {'shape': (1, 1), 'lmbda_val': [1], 'strings': [[b'a', b'b'], [b'c', b'd']]}, 10, 10)


if __name__ == '__main__':
    unittest.main()
