import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from top_n_box import *
import unittest
    
class Test(unittest.TestCase):
        
    def test_tansyo_ret(self):
        # 予測が当たっている場合
        target_ranks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        predicted_ranks = np.array([1, 3, 4, 5, 6, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        tansyo_prize = np.array([300 for _ in range(len(predicted_ranks))])
        num_hit, ret = tansyo_ret(target_ranks, predicted_ranks, tansyo_prize)
        self.assertEqual(num_hit, 1)
        self.assertEqual(ret, 200)
        
        # 予測が当たっていない場合
        target_ranks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        predicted_ranks = np.array([2, 3, 4, 5, 6, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        tansyo_prize = np.array([300 for _ in range(len(predicted_ranks))])
        num_hit, ret = tansyo_ret(target_ranks, predicted_ranks, tansyo_prize)
        self.assertEqual(num_hit, 0)
        self.assertEqual(ret, 0)
        
    def test_hukusyo_ret(self):
        # 予測的中
        # 現実では1着の場合
        target_ranks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        predicted_ranks = np.array([1, 3, 4, 5, 6, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        hukusyo_prize = np.array([300 for _ in range(len(predicted_ranks))])
        num_hit, ret = hukusyo_ret(target_ranks, predicted_ranks, hukusyo_prize)
        self.assertEqual(num_hit, 1)
        self.assertEqual(ret, 200)
        # 現実では2着の場合
        target_ranks = np.array([2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        predicted_ranks = np.array([1, 3, 4, 5, 6, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        hukusyo_prize = np.array([300 for _ in range(len(predicted_ranks))])
        num_hit, ret = hukusyo_ret(target_ranks, predicted_ranks, hukusyo_prize)
        self.assertEqual(num_hit, 1)
        self.assertEqual(ret, 200)
        # 現実では3着の場合
        target_ranks = np.array([3, 2, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        predicted_ranks = np.array([1, 3, 4, 5, 6, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        hukusyo_prize = np.array([300 for _ in range(len(predicted_ranks))])
        num_hit, ret = hukusyo_ret(target_ranks, predicted_ranks, hukusyo_prize)
        self.assertEqual(num_hit, 1)
        self.assertEqual(ret, 200)
        
        # 予測が当たっていない場合
        target_ranks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        predicted_ranks = np.array([2, 3, 4, 5, 6, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        hukusyo_prize = np.array([300 for _ in range(len(predicted_ranks))])
        num_hit, ret = hukusyo_ret(target_ranks, predicted_ranks, hukusyo_prize)
        self.assertEqual(num_hit, 0)
        self.assertEqual(ret, 0)
        
        # 7頭以下の場合 (2着までが的中)
        # 予測が当たっている場合
        target_ranks = np.array([1, 2, 3, 4, 5, 6, 7])
        predicted_ranks = np.array([2, 1, 4, 5, 6, 3, 7])
        hukusyo_prize = np.array([300 for _ in range(len(predicted_ranks))])
        num_hit, ret = hukusyo_ret(target_ranks, predicted_ranks, hukusyo_prize)
        self.assertEqual(num_hit, 1)
        self.assertEqual(ret, 200)
        # 予測が当たっていない場合
        target_ranks = np.array([1, 2, 3, 4, 5, 6, 7])
        predicted_ranks = np.array([3, 2, 1, 5, 6, 3, 7])
        hukusyo_prize = np.array([300 for _ in range(len(predicted_ranks))])
        num_hit, ret = hukusyo_ret(target_ranks, predicted_ranks, hukusyo_prize)
        self.assertEqual(num_hit, 0)
        self.assertEqual(ret, 0)
            
    
if __name__ == "__main__":
    unittest.main()