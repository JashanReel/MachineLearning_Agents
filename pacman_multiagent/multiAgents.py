# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math
import random, util
from collections import deque

from game import Agent, Directions
from util import manhattan_distance


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        def bfs_distance(start, goals):
            queue = deque([(start, 0)])
            visited = set()
            visited.add(start)

            while queue:
                pos, dist = queue.popleft()
                if pos in goals:
                    return dist
                # Parcours des directions valides
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    next_pos = (pos[0] + dx, pos[1] + dy)
                    if next_pos not in visited and not current_game_state.hasWall(next_pos[0], next_pos[1]):
                        visited.add(next_pos)
                        queue.append((next_pos, dist + 1))

            return 1_000_000

        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood()
        new_ghost_states = successor_game_state.getGhostStates()
        new_scared_times = [ghostState.scaredTimer for ghostState in new_ghost_states]

        "*** YOUR CODE HERE ***"
        #faire un système de malus/bonus de points, si je vais dans une direction avec de la nourriture --> bonus
        # si je vais dans une direction où y a des fantomes --> malus
        #si en mode invincible, je mets des bonus vers la nourriture la plus proche et
        # capsules = grosse bouffe qui rend invincible

        if successor_game_state.isWin():
            return math.inf

        if successor_game_state.isLose():
            return - math.inf

        score = 0

        foodCoords = new_food.asList()

        #nearestFood = bfs_distance(new_pos, foodCoords)
        nearestFood = math.inf
        for coords in foodCoords:
            distToFood = manhattan_distance(coords, new_pos)
            if distToFood < nearestFood:
                nearestFood = distToFood

        score -= nearestFood

        ghost_distances = []
        scared_ghost_bonus = 0
        danger_penalty = 0

        for ghost in new_ghost_states:
            ghost_pos = ghost.getPosition()
            ghost_dist = manhattan_distance(new_pos, ghost_pos)
            ghost_distances.append(ghost_dist)

            if ghost.scaredTimer > 0:
                # Fantôme effrayé -> on veut s'en approcher pour le manger, +1 pour éviter les ZeroDivisionError
                if ghost_dist <= ghost.scaredTimer:
                    scared_ghost_bonus += 200 / (ghost_dist + 1)
            else:
                # Fantôme dangereux -> malus si trop proche, +1 pour éviter les ZeroDivisionError
                if ghost_dist <= 2:
                    danger_penalty -= 300 / (ghost_dist + 1)

        score += scared_ghost_bonus
        score += danger_penalty

        if action == Directions.STOP:
            score -= 100

        successorNbFood = successor_game_state.getNumFood()
        currentNbFood = current_game_state.getNumFood()
        if successorNbFood < currentNbFood:
            score += 100

        for posCapsule in current_game_state.getCapsules():
            pacPos = successor_game_state.getPacmanPosition()
            if pacPos == posCapsule:
                score += 300

        return score + successor_game_state.getScore()

def score_evaluation_function(current_game_state):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search game
      (not reflex game).
    """
    return current_game_state.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search game.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'better', depth = '2'):
        #evalFn = 'better'
        #evalFn = 'score_evaluation_function'
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of game in the game
        """
        "*** YOUR CODE HERE ***"
        #util.raise_not_defined()
        best_action = None
        best_score = -math.inf

        for action in game_state.getLegalActions(0):
            successor_state = game_state.generateSuccessor(0, action)
            score = self.minimax(successor_state, 0, 1)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def minimax(self, game, depth, agent):
        if (self.depth == depth) or game.isWin() or game.isLose():
                return self.evaluationFunction(game)

        num_agents = game.getNumAgents()

        if agent == 0:
            maxCurr = - math.inf
            list_actions = game.getLegalActions(0)

            if not list_actions:
                return self.evaluationFunction(game)

            for turn in list_actions:
                newGame = game.generateSuccessor(agent, turn)
                newMax = self.minimax(newGame, depth, agent + 1)

                if newMax > maxCurr:
                    maxCurr = newMax

            return maxCurr

        else:
            minCurr = math.inf
            list_actions = game.getLegalActions(agent)

            if not list_actions:
                return self.evaluationFunction(game)

            for turn in list_actions:
                newGame = game.generateSuccessor(agent, turn)
                #newMin = self.minimax(newGame, depth + 1, agent - 1)

                next_agent = (agent + 1) % num_agents
                next_depth = depth + 1 if next_agent == 0 else depth
                newMin = self.minimax(newGame, next_depth, next_agent)

                if newMin < minCurr:
                    minCurr = newMin

            return minCurr

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        best_score = -math.inf

        for action in game_state.getLegalActions(0):
            successor_state = game_state.generateSuccessor(0, action)
            score = self.minimax(successor_state, 0, 1, -math.inf, math.inf)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def minimax(self, game, depth, agent, alpha, beta):
        if (self.depth == depth) or game.isWin() or game.isLose():
                return self.evaluationFunction(game)

        num_agents = game.getNumAgents()

        if agent == 0:
            maxCurr = - math.inf
            list_actions = game.getLegalActions(0)

            if not list_actions:
                return self.evaluationFunction(game)

            for turn in list_actions:
                newGame = game.generateSuccessor(agent, turn)
                newMax = self.minimax(newGame, depth, 1, alpha, beta)

                if newMax > maxCurr:
                    maxCurr = newMax

                if maxCurr > beta:
                    return maxCurr

                alpha = max(alpha, maxCurr)

            return maxCurr

        else:
            minCurr = math.inf
            list_actions = game.getLegalActions(agent)

            if not list_actions:
                return self.evaluationFunction(game)

            for turn in list_actions:
                newGame = game.generateSuccessor(agent, turn)
                #newMin = self.minimax(newGame, depth + 1, agent - 1, alpha, beta)

                next_agent = (agent + 1) % num_agents
                next_depth = depth + 1 if next_agent == 0 else depth
                newMin = self.minimax(newGame, next_depth, next_agent, alpha, beta)

                if newMin < minCurr:
                    minCurr = newMin

                if minCurr < alpha:
                    return minCurr

                beta = min(beta, minCurr)

            return minCurr

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        best_score = -math.inf

        for action in gameState.getLegalActions(0):
            successor_state = gameState.generateSuccessor(0, action)
            score = self.expectimax(successor_state, 0, 1)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def expectimax(self, game, depth, agent):
        if (self.depth == depth) or game.isWin() or game.isLose():
                return self.evaluationFunction(game)

        num_agents = game.getNumAgents()

        if agent == 0:
            maxCurr = - math.inf
            list_actions = game.getLegalActions(0)

            if not list_actions:
                return self.evaluationFunction(game)

            for turn in list_actions:
                newGame = game.generateSuccessor(agent, turn)
                newMax = self.expectimax(newGame, depth, agent + 1)

                if newMax > maxCurr:
                    maxCurr = newMax

            return maxCurr

        else:
            list_actions = game.getLegalActions(agent)

            if not list_actions:
                return self.evaluationFunction(game)

            expected_value = 0
            probability = 1.0/ len(list_actions)

            for turn in list_actions:
                newGame = game.generateSuccessor(agent, turn)
                next_agent = (agent + 1) % num_agents
                next_depth = depth + 1 if next_agent == 0 else depth
                expected_value += probability * self.expectimax(newGame, next_depth, next_agent)

            return expected_value

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

      voir bfs au lieu de manhattan distance pour prendre les murs en compte

      pacman pos
      distance vers ghosts malus
      food proche
      ghost appeuré, bonus si pacman s'en approche
      capsule invincible --> scaredTimer
    """
    "*** YOUR CODE HERE ***"

    def bfs_distance(start, goals):
        queue = deque([(start, 0)])
        visited = set()
        visited.add(start)

        while queue:
            pos, dist = queue.popleft()
            if pos in goals:
                return dist
            # Parcours des directions valides
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = (pos[0] + dx, pos[1] + dy)
                if next_pos not in visited and not currentGameState.hasWall(next_pos[0], next_pos[1]):
                    visited.add(next_pos)
                    queue.append((next_pos, dist + 1))

        return 1_000_000

    #if currentGameState.isLose():
    #    return - 9_000_000
    #elif currentGameState.isWin():
    #    return 9_000_000

    pos = currentGameState.getPacmanPosition()
    currentScore = currentGameState.getScore()

    foodlist = currentGameState.getFood().asList()

    if foodlist:
        #manhattanDistanceToClosestFood = min(map(lambda x: manhattan_distance(pos, x), foodlist))
        manhattanDistanceToClosestFood = bfs_distance(pos, set(foodlist))
    else:
        manhattanDistanceToClosestFood = 0   # à ignorer dans le cas où il n'y aurait plus aucune gomme

    numberOfCapsulesLeft = len(currentGameState.getCapsules())

    numberOfFoodsLeft = len(foodlist)

    scaredGhosts, activeGhosts = [], []
    for ghost in currentGameState.getGhostStates():
        if not ghost.scaredTimer:
            activeGhosts.append(ghost)
        else:
            scaredGhosts.append(ghost)

    if activeGhosts:
        distanceToClosestActiveGhost = min(map(lambda g: manhattan_distance(pos, g.getPosition()), activeGhosts))
        #distanceToClosestActiveGhost = bfs_distance(pos, {g.getPosition() for g in activeGhosts})
    else:
        distanceToClosestActiveGhost = math.inf  # à ignorer si aucun fantome actif

    distanceToClosestActiveGhost = max(distanceToClosestActiveGhost, 5)
    # utile pour éviter d'excessivement pénaliser pacman si un ghost est trop proche de lui
    # lui permet d'éviter de trop fuir et permet aussi de prendre plus de risques

    if scaredGhosts:
        #distanceToClosestScaredGhost = min(map(lambda g: manhattan_distance(pos, g.getPosition()), scaredGhosts))
        distanceToClosestScaredGhost = bfs_distance(pos, {g.getPosition() for g in scaredGhosts})
    else:
        distanceToClosestScaredGhost = 0  # à ignorer si aucun fantome effrayé

    score = 1 * currentScore + \
            -1.5 * manhattanDistanceToClosestFood + \
            -2 * (1. / distanceToClosestActiveGhost) + \
            -2 * distanceToClosestScaredGhost + \
            -20 * numberOfCapsulesLeft + \
            -4 * numberOfFoodsLeft

    """
    Distance à la nourriture (-1.5 * distanceToClosestFood)
        ➝ Pacman préfère les chemins courts vers la nourriture.
    
    Distance au fantôme dangereux (2 * (1. / distanceToClosestActiveGhost))
        ➝ Plus le fantôme est proche, plus on veut l'éviter.
    
    Distance au fantôme effrayé (2 * distanceToClosestScaredGhost)
        ➝ Si un fantôme est effrayé, on veut s'en approcher.
    
    Nombre de capsules restantes (-3.5 * numberOfCapsulesLeft)
        ➝ Moins il reste de capsules, plus le jeu est proche de la fin.
    
    Nombre de nourritures restantes (2 * (1. / (numberOfFoodsLeft + 1)))
        ➝ Pacman est encouragé à finir le jeu rapidement.
    """

    return score

# Abbreviation
better = betterEvaluationFunction

