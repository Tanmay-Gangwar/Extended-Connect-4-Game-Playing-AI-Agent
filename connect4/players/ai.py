import random
from time import time as currTime
import math
import numpy as np
from typing import List, Tuple, Dict
from connect4.utils import get_pts, get_valid_actions, Integer


class AIPlayer:
    def __init__(self, player_number: int, time: int):
        """
        :param player_number: Current player number
        :param time: Time per move (seconds)
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.time = time
        self.n = None
        self.m = None
        self.expectiMxDepth = None
        self.mxDepth = None
        self.start_time = currTime()
        self.totalPieces = 0
        # Do the rest of your implementation here

    def get_intelligent_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move
        This will play against either itself or a human player
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """
        # Do the rest of your implementation here
        self.start_time = currTime()
        currState = state[0]
        self.n = len(currState)
        self.m = len(currState[0])
        popsLeft = np.array([state[1][1]._i, state[1][2]._i])
        topLoc = np.array([self.n for _ in range(self.m)])
        self.totalPieces = 0
        for i in range(self.n - 1, -1, -1):
            for j in range(self.m):
                if currState[i][j] != 0:
                    topLoc[j] = i
                    self.totalPieces += 1
        
        bestMove = None
        for self.mxDepth in range(1, 30):
            print("Trying mxDepth {}".format(self.mxDepth))
            score, move = self.alphaBeta(currState, popsLeft, topLoc, self.player_number, self.mxDepth, -float('inf'), float('inf'))
            if score is None: break
            print("Best move for depth {} is {}".format(self.mxDepth, move))
            bestMove = move
        return bestMove
    
    def alphaBeta(self, state: np.array, popsLeft: np.array, topLoc: np.array, player: int, depth: int, alpha: int, beta: int):
        if currTime() - self.start_time > self.time - 1: return None, None
        availableMoves = self.getAvailableMoves(topLoc, player, popsLeft)
        if depth == 0 or len(availableMoves) == 0: 
            return self.heuristic(state)

        if player == 1: scores = [(- abs(self.m // 2 - j) - abs(self.n // 2 - topLoc[j]) - (4 if pop else 0), (j, pop)) for j, pop in availableMoves]
        else: scores = [(abs(self.m // 2 - j) + abs(self.n // 2 - topLoc[j]) + (4 if pop else 0), (j, pop)) for j, pop in availableMoves]
        scores = sorted(scores, reverse = (player == 1))

        if depth > 3:
            tempAlpha = alpha
            tempBeta = beta
            for i, (_, (j, pop)) in enumerate(scores):
                scores[i] = (None, (j, pop))

            for i, (_, (j, pop)) in enumerate(scores):
                popColor = self.playMove(state, topLoc, j, player, pop, popsLeft)
                score, move = self.alphaBeta(state, popsLeft, topLoc, 3 - player, depth - 3, tempAlpha, tempBeta)
                if score is None: return None, None
                if pop and state[self.n - 1][j] == player:
                    if player == 1: score -= 10
                    else: score += 10
                scores[i] = (score, (j, pop))
                if player == 1: tempAlpha = max(tempAlpha, scores[i][0])
                else: tempBeta = min(tempBeta, scores[i][0])
                self.undoMove(state, topLoc, j, player, pop, popColor, popsLeft)
                if tempAlpha >= tempBeta: break
            while scores[-1][0] is None: scores.pop()
            scores = sorted(scores, reverse = (player == 1))
        
        for i, (_, (j, pop)) in enumerate(scores):
            scores[i] = (None, (j, pop))

        for i in range(len(scores)):
            j, pop = scores[i][1]
            popColor = self.playMove(state, topLoc, j, player, pop, popsLeft)
            score, move = self.alphaBeta(state, popsLeft, topLoc, 3 - player, depth - 1, alpha, beta)
            if score is None: return None, None
            if pop and state[self.n - 1][j] == player:
                if player == 1: score -= 10
                else: score += 10
            # if depth == self.mxDepth: print(player, j, pop, score)
            if player == 1: scores[i] = (score - abs(self.m // 2 - j) - abs(self.n // 2 - topLoc[j]), (j, pop))
            else: scores[i] = (score + abs(self.m // 2 - j) + abs(self.n // 2 - topLoc[j]), (j, pop))
            if player == 1: alpha = max(alpha, scores[i][0])
            else: beta = min(beta, scores[i][0])
            self.undoMove(state, topLoc, j, player, pop, popColor, popsLeft)
            if alpha >= beta: break
        while scores[-1][0] is None: scores.pop()
        
        best = max(scores) if player == 1 else min(scores)
        return best
    
    def get_expectimax_move(self, state: Tuple[np.array, Dict[int, Integer]]) -> Tuple[int, bool]:
        """
        Given the current state of the board, return the next move based on
        the Expecti max algorithm.
        This will play against the random player, who chooses any valid move
        with equal probability
        :param state: Contains:
                        1. board
                            - a numpy array containing the state of the board using the following encoding:
                            - the board maintains its same two dimensions
                                - row 0 is the top of the board and so is the last row filled
                            - spaces that are unoccupied are marked as 0
                            - spaces that are occupied by player 1 have a 1 in them
                            - spaces that are occupied by player 2 have a 2 in them
                        2. Dictionary of int to Integer. It will tell the remaining popout moves given a player
        :return: action (0 based index of the column and if it is a popout move)
        """
        # Do the rest of your implementation here
        self.start_time = currTime()
        currState = state[0]
        self.n = len(currState)
        self.m = len(currState[0])
        popsLeft = np.array([state[1][1]._i, state[1][2]._i])
        topLoc = np.array([self.n for _ in range(self.m)])
        self.totalPieces = 0
        for i in range(self.n - 1, -1, -1):
            for j in range(self.m):
                if currState[i][j] != 0:
                    topLoc[j] = i
                    self.totalPieces += 1
        
        bestMove = None
        for self.expectiMxDepth in range(1, 30):
            print("Trying mxDepth {}".format(self.expectiMxDepth))
            score, move = self.expectimax(currState, popsLeft, topLoc, self.player_number, self.expectiMxDepth)
            if score is None: break
            print("Best move for depth {} is {}".format(self.expectiMxDepth, move))
            bestMove = move
        return bestMove

    def heuristic(self, state: np.array):
        score = get_pts(1, state) - get_pts(2, state)
        for i in range(self.n):
            for j in range(self.m):
                dr = (self.n - 1) / 2 - abs((self.n - 1) / 2 - i)
                dc = (self.m - 1) / 2 - abs((self.m - 1) / 2 - j)
                if state[i][j] == 1: score += dr * dr + dc * dc
                else: score -= dr * dr + dc * dc
        return score, None
    
    def playPopMove(self, state: np.array, topLoc: np.array, loc: int, player: int, popsLeft: np.array):
        popColor = state[self.n - 1][loc]
        for i in range(self.n - 1, topLoc[loc], -1):
            state[i][loc] = state[i - 1][loc]
        state[topLoc[loc]][loc] = 0
        topLoc[loc] += 1
        popsLeft[player - 1] -= 1
        self.totalPieces -= 1
        return popColor

    
    def undoPopMove(self, state: np.array, topLoc: np.array, loc: int, popColor: int, player: int, popsLeft: np.array):
        topLoc[loc] -= 1
        for i in range(topLoc[loc], self.n - 1):
            state[i][loc] = state[i + 1][loc]
        state[self.n - 1][loc] = popColor
        popsLeft[player - 1] += 1
        self.totalPieces += 1
    

    def playInsertMove(self, state: np.array, topLoc: np.array, loc: int, player: int):
        topLoc[loc] -= 1
        state[topLoc[loc]][loc] = player
        self.totalPieces += 1
    

    def undoInsertMove(self, state: np.array, topLoc: np.array, loc: int, player: int):
        state[topLoc[loc]][loc] = 0
        topLoc[loc] += 1
        self.totalPieces -= 1
    

    def playMove(self, state: np.array, topLoc: np.array, loc: int, player: int, popMove: int, popsLeft: np.array):
        if popMove: return self.playPopMove(state, topLoc, loc, player, popsLeft)
        self.playInsertMove(state, topLoc, loc, player)
    

    def undoMove(self, state: np.array, topLoc: np.array, loc: int, player: int, popMove: int, popColor: int, popsLeft: np.array):
        if popMove: self.undoPopMove(state, topLoc, loc, popColor, player, popsLeft)
        else: self.undoInsertMove(state, topLoc, loc, player)
    

    def getAvailableMoves(self, topLoc: np.array, player: int, popsLeft: np.array):
        moves = []
        for j in range(self.m):
            if topLoc[j] > 0:
                moves.append((j, False))
        for j in range(self.m):
            if topLoc[j] < self.n and popsLeft[player - 1] > 0 and j % 2 == player - 1 and self.totalPieces * 2 >= self.n * self.m:
                moves.append((j, True))
        return moves
    
    
    def expectimax(self, state: np.array, popsLeft: np.array, topLoc: np.array, player: int, depth: int):
        if currTime() - self.start_time > self.time - 1: return None, None
        availableMoves = self.getAvailableMoves(topLoc, player, popsLeft)
        if depth == 0 or len(availableMoves) == 0: 
            return self.heuristic(state)

        scores = [None for _ in range(len(availableMoves))]
        for i in range(len(availableMoves)):
            j, pop = availableMoves[i]
            popColor = self.playMove(state, topLoc, j, player, pop, popsLeft)
            score, move = self.expectimax(state, popsLeft, topLoc, 3 - player, depth - 1)
            if score is None: return None, None
            if pop and state[self.n - 1][j] == player:
                if player == 1: score -= 10
                else: score += 10
            if player == self.player_number: 
                if player == 1: scores[i] = (score - abs(self.m // 2 - j) - abs(self.n // 2 - topLoc[j]), (j, pop))
                else: scores[i] = (score + abs(self.m // 2 - j) + abs(self.n // 2 - topLoc[j]), (j, pop))
            else: scores[i] = score
            self.undoMove(state, topLoc, j, player, pop, popColor, popsLeft)
        
        if self.player_number == player: 
            best = max(scores) if player == 1 else min(scores)
            return best
        return sum(scores) / len(scores), None