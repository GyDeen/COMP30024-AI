import unittest

from agent.board import BoardState
from agent.action import place, move, eat, cascade

# To run the test use command python -m unittest tests/test_actions.py
class TestCascadeActions(unittest.TestCase):

    def setUp(self):
        """This runs automatically before EVERY individual test."""
        self.board = BoardState()
        self.board.red_tokens = 0
        self.board.blue_tokens = 0

    def test_place_spawns_token_correctly(self):
        # Execute
        new_board = place(self.board, idx=27, is_red=True)
        
        # Assert Grid Physics
        self.assertTrue(new_board.is_red(27))
        self.assertEqual(new_board.get_height(27), 3)
        
        # Assert Game Rules (Tokens)
        self.assertEqual(new_board.red_tokens, 3)
        self.assertEqual(new_board.blue_tokens, 0)

    def test_move_merges_friendly_stacks(self):
        # Setup: Red stack of 1 at pos 10, Red stack of 2 at pos 11
        self.board.set_cell(10, is_red=True, height=1)
        self.board.set_cell(11, is_red=True, height=2)
        self.board.red_tokens = 3

        # Move Right
        new_board = move(self.board, start_idx=10, offset=1)

        # Assert: Cell 10 empty, Cell 11 is height 3
        self.assertTrue(new_board.is_empty(10))
        self.assertEqual(new_board.get_height(11), 3)
        self.assertEqual(new_board.red_tokens, 3)

    def test_eat_deducts_enemy_material(self):
        # Setup: Red height 3 at pos 10, Blue height 2 at pos 11
        self.board.set_cell(10, is_red=True, height=3)
        self.board.set_cell(11, is_red=False, height=2)
        self.board.red_tokens = 3
        self.board.blue_tokens = 2

        # Execute: Red eats Blue
        new_board = eat(self.board, start_idx=10, offset=1)

        # Assert: Cell 10 empty, Cell 11 is Red height 3
        self.assertTrue(new_board.is_empty(10))
        self.assertTrue(new_board.is_red(11))
        
        # Assert: Blue lost 2 tokens!
        self.assertEqual(new_board.red_tokens, 3)
        self.assertEqual(new_board.blue_tokens, 0)

    def test_cascade_pushes_off_edge(self):
        # Setup: Red height 2 at pos 6 (near right edge). Blue height 1 at pos 7 (right edge).
        self.board.set_cell(6, is_red=True, height=2)
        self.board.set_cell(7, is_red=False, height=1)
        self.board.red_tokens = 2
        self.board.blue_tokens = 1
        self.board.render()

        # Execute: Red cascades RIGHT (+1)
        new_board = cascade(self.board, start_idx=6, offset=1)

        # Assert: Origin empty, Cell 7 has 1 Red token.
        self.assertTrue(new_board.is_empty(6))
        self.assertTrue(new_board.is_red(7))
        self.assertEqual(new_board.get_height(7), 1)
        new_board.render()

        # Assert Material: Red lost 1 token (fell off). Blue lost 1 token (pushed off).
        self.assertEqual(new_board.red_tokens, 1)
        self.assertEqual(new_board.blue_tokens, 0)

if __name__ == '__main__':
    unittest.main()