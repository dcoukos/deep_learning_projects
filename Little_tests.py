import unittest
from shared_arch import *
import torch


class MyTestCase(unittest.TestCase):
    def test_hotstuff(self):
        A = torch.tensor([[1, 2], [4, 5], [8,9]]) #vector Nx2with a number for each class ^
        B = torch.tensor([[0,1,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,0,0,1]])
        C = convert_to_hot(A)

        self.assertEqual(True, torch.equal(B.long(), C.long()))


if __name__ == '__main__':
    unittest.main()
