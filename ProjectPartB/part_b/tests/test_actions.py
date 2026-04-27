import unittest

from agent_alphabeta.board import BoardState
from agent_alphabeta.action import place, move, eat, cascade, UP, DOWN, LEFT, RIGHT

# To run all the test use command python -m unittest tests/test_actions.py
# To run specific test use command python -m unittest tests.test_actions.TestCascadeActions.test_name
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
        new_board = cascade(self.board, start_idx=6, offset=RIGHT)

        # Assert: Origin empty, Cell 7 has 1 Red token.
        self.assertTrue(new_board.is_empty(6))
        self.assertTrue(new_board.is_red(7))
        self.assertEqual(new_board.get_height(7), 1)
        new_board.render()

        # Assert Material: Red lost 1 token (fell off). Blue lost 1 token (pushed off).
        self.assertEqual(new_board.red_tokens, 1)
        self.assertEqual(new_board.blue_tokens, 0)
        
    def test_place_spawns_token_both_color(self):
        # set up proper placing phase for both color. Red placed at row 2 col 7, Blue placed at row 0 col 7
        new_board = place(self.board, idx=7, is_red=False)
        new_board = place(new_board, idx=23, is_red=True)

        self.assertTrue(new_board.is_red(23))
        self.assertEqual(new_board.get_height(23), 3)
        self.assertTrue(new_board.is_blue(7))
        self.assertEqual(new_board.get_height(7), 3)

        cascade_board = cascade(new_board, start_idx=23, offset=UP)
        cascade_board.render()
        
        # Assert cell 15 and cell 7 both red with height 1
        self.assertTrue(cascade_board.is_red(15))
        self.assertTrue(cascade_board.is_red(7))
        self.assertEqual(cascade_board.get_height(15), 1)
        self.assertEqual(cascade_board.get_height(7), 1)
        
        # Assert blue stack being eliminated
        self.assertEqual(cascade_board.red_tokens, 2)
        self.assertEqual(cascade_board.blue_tokens, 0)
        
    def test_red_merge_and_cascade_push_off(self):
        # --- Setup ---
        # 1. Red stack of 1 at pos 4 (Row 0, Col 4)
        # 2. Red stack of 1 at pos 5 (Row 0, Col 5)
        # 3. Red stack of 1 at pos 6 (Row 0, Col 6)
        # 4. Blue stack of 1 at pos 7 (Row 0, Col 7 - The Edge)
        self.board.set_cell(4, is_red=True, height=1)
        self.board.set_cell(5, is_red=True, height=1)
        self.board.set_cell(6, is_red=True, height=1)
        self.board.set_cell(7, is_red=False, height=1)
        self.board.red_tokens = 3
        self.board.blue_tokens = 1

        print("--- Initial Setup ---")
        self.board.render()

        # --- Action 1: First Move (Merge) ---
        # Red at 4 moves Right (+1) to merge with Red at 5
        board_v1 = move(self.board, start_idx=4, offset=1)
        print("--- After First Move (Merge 4 -> 5) ---")
        board_v1.render()

        # --- Action 2: Second Move (Merge) ---
        # Red (now height 2) at 5 moves Right (+1) to merge with Red at 6
        board_v2 = move(board_v1, start_idx=5, offset=1)
        print("--- After Second Move (Merge 5 -> 6) ---")
        board_v2.render()

        # --- Action 3: Cascade ---
        # Red (now height 3) at 6 cascades Right (+1)
        # This should: 
        # - Push the Blue stack at 7 off the board (Blue tokens: 1 -> 0)
        # - Place a Red token at 7 (Height 1)
        # - Two Red tokens fall off the edge (Red tokens: 3 -> 1)
        final_board = cascade(board_v2, start_idx=6, offset=1)
        print("--- Final Board after Cascade ---")
        final_board.render()

        # --- Assertions ---
        # 1. Grid State
        self.assertTrue(final_board.is_empty(4))
        self.assertTrue(final_board.is_empty(5))
        self.assertTrue(final_board.is_empty(6))
        self.assertTrue(final_board.is_red(7))
        self.assertEqual(final_board.get_height(7), 1)

        # 2. Token Counts
        # Red started with 3. 1 is on board, 2 fell off during cascade.
        self.assertEqual(final_board.red_tokens, 1)
        # Blue started with 1. It was pushed off the board.
        self.assertEqual(final_board.blue_tokens, 0)

if __name__ == '__main__':
    unittest.main()