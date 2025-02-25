import numpy as np
import copy
import re
import random
import math
from stratego import Stratego
from utils import _create_log

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.move = move

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.legal_moves())

    def best_child(self, exploration_weight=1.41):
        if not self.children:
            return None
        best_value = -float('inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                continue
            uct_value = (child.wins / child.visits) + exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        return best_child

class MCTSPlayer:
    def __init__(self, simulations=1000, exploration_weight=1.41):
        self.simulations = simulations
        self.exploration_weight = exploration_weight
        
    def choose_best_move_based_on_probabilities(self, game, legal_moves):
        prob_board = game.get_prob_board()
        best_move = None
        best_score = -float("inf")
        
        for move in legal_moves:
            from_pos, to_pos = move
            prob_distribution = prob_board.get(to_pos, {})
            
            attacker_piece = game.board[from_pos[0]][from_pos[1]]
            if attacker_piece == "EMPTY" or "_" not in attacker_piece:
                continue
            
            attacker_piece = attacker_piece.split("_")[0]
            if attacker_piece not in game.soldiers[game.turn]:
                continue
            
            attacker_rank = game.soldiers[game.turn][attacker_piece]["Rank"]
            attack_score = 0

            for p in prob_distribution:
                if p not in game.soldiers[game.turn]:
                    continue
                defender_rank = game.soldiers[game.turn][p]["Rank"]
                if attacker_rank >= defender_rank:
                    attack_score += prob_distribution[p] * defender_rank
            attack_score += np.random.uniform(0, 0.1)
            if attack_score > best_score:
                best_score = attack_score
                best_move = move
        return best_move

    def choose_move(self, game):
        legal_moves = game.legal_moves()
        best_prob_move = self.choose_best_move_based_on_probabilities(game, legal_moves)
        if best_prob_move:
            return best_prob_move
        root = MCTSNode(game.clone())
        for _ in range(self.simulations):
            if _ % 100 == 0:
                log_msg = f"🔄 Running MCTS Simulation {_}/{self.simulations}..."
                _create_log(log_msg, "Info","game_generation_log.txt")
                print(log_msg)
            node = self._select(root)
            if not node:
                continue
            result = self._simulate(node.state)
            self._backpropagate(node, result)
        best_child = root.best_child(exploration_weight=0)
        if best_child:
            return best_child.move
        return None

    def _select(self, node):
        while node.children and node.is_fully_expanded():
            best_child = node.best_child(self.exploration_weight)
            if best_child is None:
                return node
            node = best_child
        return self._expand(node)

    def _expand(self, node):
        legal_moves = node.state.legal_moves()
        tried_moves = {child.move for child in node.children}
        for move in legal_moves:
            if move not in tried_moves:
                new_state = node.state.clone()
                new_state.make_move(*move)
                child = MCTSNode(new_state, parent=node, move=move)
                node.add_child(child)
                return child
        return None

    def _simulate(self, game, max_depth = 70):
        depth = 0
        while game.status() == "ongoing" and depth < max_depth:
            legal_moves = game.legal_moves()
            if not legal_moves:
                # print(f"⚠️ No valid move found for {game.turn}. Ending game as a draw.")
                # game.game_over = True
                break
            move = random.choice(legal_moves)
            game.make_move(*move)
            depth += 1
        return game.status()

    def _backpropagate(self, node, result):
        while node:
            node.visits += 1
            if result == node.state.turn:
                node.wins += 1
            elif result == "draw":
                node.wins += 0.5
            node = node.parent
