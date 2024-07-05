import unittest
import numpy as np
from src.data.preprocess import preprocess_images

class TestPreprocess(unittest.TestCase):
    def test_preprocess_images(self):
        data_dir = 'test_data'
        images, labels = preprocess_images(data_dir)
        self.assertEqual(images.shape[1:], (224, 224, 3))
        self.assertTrue(len(labels) > 0)
        self.assertTrue(np.all((labels == 0) | (labels == 1)))

if __name__ == '__main__':
    unittest.main()
