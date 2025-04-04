import random
import pygame
import numpy as np
from time import sleep

# Tetris game class
class Tetris:

    '''Tetris game class'''

    # BOARD
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: { # I
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: { # T
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: { # L
            0: [(1,0), (1,1), (1,2), (2,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: { # J
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # Z
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: { # S
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: { # O
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }

    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }

    def __init__(self, tick_rate=3):
        pygame.init()
        self.screen = pygame.display.set_mode((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 25)
        self.tick_rate = tick_rate
        self.reset()

    def reset(self):
        '''Resets the game, returning the current state'''
        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        self.score = 0
        return self._get_board_props(self.board)

    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]

    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = Tetris.MAP_PLAYER
        return board

    def get_game_score(self):
        '''Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        '''
        return self.score

    def _new_round(self):
        '''Starts a new round (new piece)'''
        # Generate new bag with the pieces
        if len(self.bag) == 0:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)

        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True

    def _check_collision(self, piece, pos):
        for x, y in piece:
            new_x = x + pos[0]
            new_y = y + pos[1]
            # 确保坐标在合法范围内
            if new_x < 0 or new_x >= Tetris.BOARD_WIDTH or new_y >= Tetris.BOARD_HEIGHT:
                return True
            if new_y >= 0 and self.board[new_y][new_x] == Tetris.MAP_BLOCK:
                return True
        return False

    def _rotate(self, angle):
        '''Change the current rotation with wall kick'''
        original_rotation = self.current_rotation
        original_pos = self.current_pos.copy()

        # 计算新角度
        r = self.current_rotation + angle
        r %= 360  # 规范化角度到[0, 360)

        # 获取新旧旋转的形状
        old_piece = self._get_rotated_piece()
        self.current_rotation = r
        new_piece = self._get_rotated_piece()

        # 计算边界越界量
        min_x = min(p[0] + self.current_pos[0] for p in new_piece)
        max_x = max(p[0] + self.current_pos[0] for p in new_piece)
        offset = 0

        if max_x >= Tetris.BOARD_WIDTH:
            offset = Tetris.BOARD_WIDTH - 1 - max_x
        elif min_x < 0:
            offset = -min_x

        # 尝试修正位置
        self.current_pos[0] += offset

        # 如果修正后仍碰撞，则回滚旋转
        if self._check_collision(new_piece, self.current_pos):
            self.current_rotation = original_rotation
            self.current_pos = original_pos

        self.current_rotation = r

    def _add_piece_to_board(self, piece, pos):
        board = [x[:] for x in self.board]
        for x, y in piece:
            bx = x + pos[0]
            by = y + pos[1]
            if 0 <= by < Tetris.BOARD_HEIGHT and 0 <= bx < Tetris.BOARD_WIDTH:
                board[by][bx] = Tetris.MAP_BLOCK
        return board

    def _clear_lines(self, board):
        '''Clears completed lines in a board'''
        # 修改判断条件，确保每个格子都是MAP_BLOCK
        lines_to_clear = [index for index, row in enumerate(board) if all(cell == Tetris.MAP_BLOCK for cell in row)]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
        return len(lines_to_clear), board


    def _number_of_holes(self, board):
        '''Number of holes in the board (empty sqquare with at least one block above it)'''
        holes = 0

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            holes += len([x for x in col[i+1:] if x == Tetris.MAP_EMPTY])

        return holes

    def _bumpiness(self, board):
        '''Sum of the differences of heights between pair of columns'''
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] != Tetris.MAP_BLOCK:
                i += 1
            min_ys.append(i)

        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += abs(min_ys[i] - min_ys[i+1])

        return total_bumpiness, max_bumpiness

    def _height(self, board):
        '''Sum and maximum height of the board'''
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i] == Tetris.MAP_EMPTY:
                i += 1
            height = Tetris.BOARD_HEIGHT - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height

    def _get_board_props(self, board):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]

    def get_state(self):
        '''Returns the current state of the game'''
        return self._get_board_props(self.board)

    def get_action_size(self):
        '''Returns the size of the action space'''
        return len(self.TETROMINOS)

    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            if 0 <= x < Tetris.BOARD_WIDTH and 0 <= y < Tetris.BOARD_HEIGHT:  # 新增边界检查
                board[y][x] = Tetris.MAP_PLAYER
        return board

    def get_next_states(self):
        '''Get all possible next states'''
        states = {}
        piece_id = self.current_piece

        if piece_id == 6:
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states

    def get_state_size(self):
        '''Size of the state'''
        return 4

    def play(self, x, rotation, render=False, render_delay=None):
        '''Makes a play given a position and a rotation, returning the reward and if the game is over'''
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        # Drop piece
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1

        # Update board and calculate score
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        score = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
        self.score += score

        # Start new round
        self._new_round()
        if self.game_over:
            score -= 2

        return score, self.game_over

    def render(self):
        '''Renders the current board'''
        self.screen.fill((0, 0, 0))
        for y, row in enumerate(self._get_complete_board()):
            for x, cell in enumerate(row):
                if cell != Tetris.MAP_EMPTY:
                    pygame.draw.rect(self.screen, Tetris.COLORS[cell], pygame.Rect(x * 25, y * 25, 25, 25))

        # 绘制分数
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.tick_rate)

    def handle_keys(self):
        '''Handles key presses for manual mode'''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self._move(-1)
                elif event.key == pygame.K_RIGHT:
                    self._move(1)
                elif event.key == pygame.K_DOWN:
                    self._drop()
                elif event.key == pygame.K_UP:
                    self._rotate(90)

    def _move(self, direction):
        '''Move the current piece left or right'''
        new_pos = [self.current_pos[0] + direction, self.current_pos[1]]
        if not self._check_collision(self._get_rotated_piece(), new_pos):
            self.current_pos = new_pos

    def _drop(self):
        '''Drop the current piece by one row'''
        new_pos = [self.current_pos[0], self.current_pos[1] + 1]
        if not self._check_collision(self._get_rotated_piece(), new_pos):
            self.current_pos = new_pos
        else:
            self._lock_piece()

    def _lock_piece(self):
        '''Lock the current piece in place and start a new round'''
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        self.score += 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
        self._new_round()

    def play_manual(self, render=False, render_delay=None):
        '''Play the game in manual mode'''
        while not self.game_over:
            self.handle_keys()
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self._drop()
        return self.score

if __name__ == '__main__':
    tetris = Tetris()
    tetris.play_manual(render=True)