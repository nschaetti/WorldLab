
import unittest
from worldlab.board.games import CheckersBoard, CheckersGame

class CheckersTest(unittest.TestCase):

    def test_manoury_to_coords(self):
        board = CheckersBoard()
        self.assertTrue(board.manoury_to_coords(1) == (0, 0))
        self.assertTrue(board.manoury_to_coords(5) == (0, 4))
        self.assertTrue(board.manoury_to_coords(6) == (1, 0))
        self.assertTrue(board.manoury_to_coords(16) == (3, 0))
        self.assertTrue(board.manoury_to_coords(26) == (5, 0))
        self.assertTrue(board.manoury_to_coords(36) == (7, 0))
        self.assertTrue(board.manoury_to_coords(45) == (8, 4))
        self.assertTrue(board.manoury_to_coords(50) == (9, 4))

        with self.assertRaises(AssertionError):
            board.manoury_to_coords(0)
        # end with

        with self.assertRaises(AssertionError):
            board.manoury_to_coords(51)
        # end with
    # end def manoury_coords

    def test_coords_to_manoury(self):
        board = CheckersBoard()
        self.assertTrue(board.coords_to_manoury(row=0, col=0) == 1)
        self.assertTrue(board.coords_to_manoury(row=0, col=4) == 5)
        self.assertTrue(board.coords_to_manoury(row=1, col=0) == 6)
        self.assertTrue(board.coords_to_manoury(row=3, col=0) == 16)
        self.assertTrue(board.coords_to_manoury(row=5, col=0) == 26)
        self.assertTrue(board.coords_to_manoury(row=7, col=0) == 36)
        self.assertTrue(board.coords_to_manoury(row=8, col=4) == 45)
        self.assertTrue(board.coords_to_manoury(row=9, col=4) == 50)

        with self.assertRaises(AssertionError):
            board.coords_to_manoury(row=-1, col=0)
        # end with

        with self.assertRaises(AssertionError):
            board.coords_to_manoury(row=0, col=5)
        # end with

        with self.assertRaises(AssertionError):
            board.coords_to_manoury(row=0, col=-1)
        # end with

        with self.assertRaises(AssertionError):
            board.coords_to_manoury(row=10, col=0)
        # end with
    # end def test_coords_to_manoury

    def test_manoury_to_board_coords(self):
        board = CheckersBoard()
        self.assertTrue(board.manoury_to_board_coords(1) == (0, 1))
        self.assertTrue(board.manoury_to_board_coords(5) == (0, 9))
        self.assertTrue(board.manoury_to_board_coords(6) == (1, 0))
        self.assertTrue(board.manoury_to_board_coords(10) == (1, 8))
        self.assertTrue(board.manoury_to_board_coords(21) == (4, 1))
        self.assertTrue(board.manoury_to_board_coords(50) == (9, 8))

        with self.assertRaises(AssertionError):
            board.manoury_to_coords(0)
        # end with

        with self.assertRaises(AssertionError):
            board.manoury_to_coords(51)
        # end with
    # end def test_manoury_to_board_coords

    def test_board_coords_to_manoury(self):
        board = CheckersBoard()
        self.assertTrue(board.board_coords_to_manoury(0, 0) is None)
        self.assertTrue(board.board_coords_to_manoury(0, 1) == 1)
        self.assertTrue(board.board_coords_to_manoury(0, 2) is None)
        self.assertTrue(board.board_coords_to_manoury(0, 3) == 2)
        self.assertTrue(board.board_coords_to_manoury(0, 4) is None)
        self.assertTrue(board.board_coords_to_manoury(0, 5) == 3)
        self.assertTrue(board.board_coords_to_manoury(0, 6) is None)
        self.assertTrue(board.board_coords_to_manoury(0, 7) == 4)
        self.assertTrue(board.board_coords_to_manoury(0, 8) is None)
        self.assertTrue(board.board_coords_to_manoury(0, 9) == 5)


        self.assertTrue(board.board_coords_to_manoury(1, 0) == 6)
        self.assertTrue(board.board_coords_to_manoury(2, 0) is None)
        self.assertTrue(board.board_coords_to_manoury(3, 0) == 16)
        self.assertTrue(board.board_coords_to_manoury(4, 0) is None)
        self.assertTrue(board.board_coords_to_manoury(5, 0) == 26)
        self.assertTrue(board.board_coords_to_manoury(6, 0) is None)
        self.assertTrue(board.board_coords_to_manoury(7, 0) == 36)
        self.assertTrue(board.board_coords_to_manoury(8, 0) is None)
        self.assertTrue(board.board_coords_to_manoury(9, 0) == 46)

        self.assertTrue(board.board_coords_to_manoury(9, 8) == 50)
        self.assertTrue(board.board_coords_to_manoury(9, 9) is None)

        with self.assertRaises(AssertionError):
            board.board_coords_to_manoury(0, -1)
        # end with

        with self.assertRaises(AssertionError):
            board.board_coords_to_manoury(0, 10)
        # end with

        with self.assertRaises(AssertionError):
            board.board_coords_to_manoury(-1, 0)
        # end with

        with self.assertRaises(AssertionError):
            board.board_coords_to_manoury(10, 0)
        # end with
    # end test_manoury_to_board_coords

# end class CheckersTest

if __name__ == '__main__':
    unittest.main()
# end if


