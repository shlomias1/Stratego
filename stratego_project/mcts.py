import numpy as np
import copy
import re
import random
import math
from stratego import Stratego

class MCTSPlayer:
    def __init__(self, simulations=1000, exploration_weight=2):
        self.simulations = simulations
        self.exploration_weight = exploration_weight

    def choose_move(self, game):
        self.simulating_player = game.turn  
        root = MCTSNode(game.clone())
        for _ in range(self.simulations):
            node = self._select(root)
            if node is None:
                break
            result = self._simulate(node.state)
            self._backpropagate(node, result)   
        best_child = root.get_best_child()
        if best_child is None or not best_child.state.history:
            raise ValueError("No valid moves found.")
        move = best_child.state.history[-1] 
        from_pos, to_pos = move
        if game.turn != self.simulating_player:
            raise ValueError("Turn mismatch after MCTS.")
        if not game.is_legal_move(from_pos, to_pos):
            raise ValueError(f"Illegal move chosen by MCTS: {move}")
        return move

    def _select(self, node):
        while not node.state.status() in ["red wins", "blue wins", "draw"]:
            if len(node.children) < len(node.state.legal_moves()):
                return self._expand(node)
            else:
                node = self._uct_select(node)
        return node

    def _uct_select(self, node):
        best_value = -float('inf')
        best_child = None
        for child in node.children:
            uct_value = (
                child.value / (child.visits + 1) +
                self.exploration_weight * math.sqrt(math.log(node.visits + 1) / (child.visits + 1))
            )
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        return best_child or node  # Return current node if no valid child exists

    def _expand(self, node):
        legal_moves = node.state.legal_moves()
        for move in legal_moves:
            if move not in [child.state.history[-1] for child in node.children]:
                child = node.clone()
                child.state.make_move(*move)
                node.add_child(child)
                return child
        return None  # Return None if no new child could be created

    def _simulate(self, game, max_depth=50):
        depth = 0
        if game.turn != self.simulating_player:
            game.switch_turn()
        while game.status() == "ongoing" and depth < max_depth:
            legal_moves = game.legal_moves()
            if not legal_moves: 
                print(f"No legal moves available for {game.turn} at depth {depth}.")
                break
            move = random.choice(legal_moves)
            print(f"Simulating move: {move} for {game.turn} at depth {depth}.")  
            try:
                game.make_move(*move)
            except ValueError as e:
                print(f"Simulation Error at depth {depth}: {e}") 
                print(f"Board state:\n{game}")
                break
            depth += 1
        return game.status()

    def _backpropagate(self, node, result):
        while node:
            node.visits += 1
            if result == node.state.turn:
                print(f"node.state.turn: {node.state.turn}")
                print(f"node.value: {node.value}")
                node.value += 1
            elif result == "draw":
                node.value += 0.5
            node = node.parent

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_best_child(self):
        if not self.children:
            return None  # Return None if no children exist
        best_child = None
        best_value = -float('inf')
        for child in self.children:
            win_rate = child.value / (child.visits + 1)
            if win_rate > best_value:
                best_value = win_rate
                best_child = child
        return best_child

    def clone(self):
        return MCTSNode(self.state.clone(), parent=self)
