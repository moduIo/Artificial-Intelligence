# Tim Zhang
# 110746199
# CSE537 HW 1
#---------------------------------------------------
import sys
import queue
import time

#---------------------------------------------------
# Node class
#---------------------------------------------------
class node:
	#
	# Constructor
	#
	def __init__(self, state, parent, action):
		global nid
		nid += 1
		self.state = state
		self.parent = parent
		self.action = action
		self.nid = nid

		if parent != None:
			self.pathCost = parent.pathCost + 1
			self.depth = parent.depth + 1

		else:
			self.pathCost = 0
			self.depth = 0

		# Heuristic switch
		#if int(sys.argv[5])	== 1:
		#	self.heuristicCost = self.manhattanDistanceHeuristic()
		#else:
		#	self.heuristicCost = self.misplacedTilesHeuristic()

		self.heuristicCost = self.manhattanDistanceHeuristic()
		self.fCost = self.pathCost + self.heuristicCost

	#
	# Less than definition for priority queue
	#
	def __lt__(self, other):
		return self.nid < other.nid

	#
	# Returns the state reached when the blank is moved left
	#
	def moveLeft(self, emptySpace):
		state = list(self.state)
		state[emptySpace], state[emptySpace - 1] = state[emptySpace - 1], state[emptySpace]

		return state

	#
	# Returns the state reached when the blank is moved right
	#
	def moveRight(self, emptySpace):
		state = list(self.state)
		state[emptySpace], state[emptySpace + 1] = state[emptySpace + 1], state[emptySpace]

		return state

	#
	# Returns the state reached when the blank is moved up
	#
	def moveUp(self, emptySpace):
		state = list(self.state)
		state[emptySpace], state[emptySpace - instance] = state[emptySpace - instance], state[emptySpace]

		return state

	#
	# Returns the state reached when the blank is moved down
	#
	def moveDown(self, emptySpace):
		state = list(self.state)
		state[emptySpace], state[emptySpace + instance] = state[emptySpace + instance], state[emptySpace]

		return state

	#
	# Calculates the successors of the state
	#
	def successor(self):
		global instance
		successors = [None] * 0
		rightEdge = False
		leftEdge = False
		topEdge = False
		bottomEdge = False

		emptySpace = self.state.index(0)

		# Calculate positional constraints
		if (emptySpace + 1) % instance == 0:  # The space is on a right-most edge
			rightEdge = True

		elif (emptySpace + 1) % instance == 1:  # The space if on a left-most edge
			leftEdge = True

		if (emptySpace + 1) <= instance:  # The space is in the top row
			topEdge = True

		elif (emptySpace + 1) > (instance * instance) - instance:  # The space in on the bottom row
			bottomEdge = True

		# Calculate the successors
		if not topEdge:
			# Move Up action
			successors.append(['U', self.moveUp(emptySpace)])

		if not bottomEdge:
			# Move Down action
			successors.append(['D', self.moveDown(emptySpace)])

		if not leftEdge:
			# Move Left action
			successors.append(['L', self.moveLeft(emptySpace)])

		if not rightEdge:
			# Move Right action
			successors.append(['R', self.moveRight(emptySpace)])

		return successors

	#
	# Calculates the "Misplaced Tiles" heuristic
	#
	def misplacedTilesHeuristic(self):
		global instance
		misplacedTiles = 0

		for n in range(1, instance * instance):
			if self.state.index(n) + 1 != n:
				misplacedTiles += 1

		return misplacedTiles

	#
	# Calculates the "Manhattan Distance" heuristic
	#
	def manhattanDistanceHeuristic(self):
		global instance
		manhattanDistance = 0

		for pos in range(1, instance * instance):
			i = self.state.index(pos) + 1

			while i != pos:

				if i < pos:
					# Try moving the tile down
					if i + instance <= pos:
						i += instance
						manhattanDistance += 1

					# If the tile is within a right movable distance away
					# AND it is not on a right edge
					elif (pos - i <= instance - (i % instance)) and (i % instance != 0):
						move = pos - i
						i += move
						manhattanDistance += move

					# Else move the tile down
					else:
						i += instance
						manhattanDistance += 1

				else:
					# Try moving the tile up
					if i - instance >= pos:
						i -= instance
						manhattanDistance += 1

					# If the tile is within a left movable distance away
					# OR it is a on a right edge and the movable distance is within n - 1 moves away
					elif (i - pos <= i % instance - 1) or (i % instance == 0 and i - pos <= instance - 1):
						move = i - pos
						i -= move
						manhattanDistance += move

					# Else move the tile up
					else:
						i -= instance
						manhattanDistance += 1

		return manhattanDistance

	#
	# Checks if the node is a goal state by a simple matching
	#
	def goalTest(self):
		global instance

		if (self.state == [1, 2, 3, 4, 5, 6, 7, 8, 0] and instance == 3) or (self.state == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0] and instance == 4):
			return True

		return False

#---------------------------------------------------
# Implementation of A* Tree Search
#---------------------------------------------------
def aStarSearch(intialState):
	global explored
	frontier = queue.PriorityQueue()

	frontier.put((initialState.fCost, initialState))

	while not frontier.empty():
		n = frontier.get()
		current = n[1]
		explored += 1

		if current.goalTest():
			return current

		for successor in current.successor():
			# Do not put reversible moves onto the queue
			if current.parent == None or successor[1] != current.parent.state:
				s = node(successor[1], current, successor[0])
				frontier.put((s.fCost, s))			

	return False

#---------------------------------------------------
# Implementation of Iterative Deepening A* Search
#---------------------------------------------------
def iterativeDeepeningAStarSearch(initialState):
	limit = initialState.fCost

	while True:
		result = depthLimitedAStarSearch(initialState, limit)
		
		# result will be the new depth limit when depth limit is reached
		if isinstance(result, int):
			limit = result
			continue

		# If no node was reached then return False
		elif not result:
			return False

		# Otherwise return the node
		return result

#---------------------------------------------------
# Implementation of recursive DLA* Search
#---------------------------------------------------
def depthLimitedAStarSearch(n, limit):
	global explored
	explored += 1

	if n.goalTest():
		return n

	elif n.fCost > limit:
		return n.fCost

	else:
		cutoff = float("inf")
		cutoffOccured = False

		for successor in n.successor():
			# Do not expand reversible moves
			if n.parent == None or successor[1] != n.parent.state:
				s = node(successor[1], n, successor[0])

				result = depthLimitedAStarSearch(s, limit)

				# If the result is an f-cost
				if isinstance(result, int):
					cutoffOccured = True

					# Keep track of the minimum cutoff
					if result < cutoff:
						cutoff = result

				elif result:
					return result

		if cutoffOccured:
			return cutoff

		else:
			return False

#---------------------------------------------------
# Returns the path of actions that lead to the
# solution into the given file.
#---------------------------------------------------
def solutionPath(node):
	fout = open(sys.argv[4], "w")
	path = []
	solution = ''

	while node.parent != None:
		path.append(node.action)
		node = node.parent

	while path:
		if len(path) == 1:
			solution += path.pop()
		else:
			solution += path.pop() + ','

	fout.write(solution)

#---------------------------------------------------
# Transforms input puzzle into a list
# The "Blank" tile will be represented as a 0
#---------------------------------------------------
def generatePuzzle(puzzle):
	fin = open(sys.argv[3])
	lines = fin.readlines()

	for l in lines:
		row = l.split(',')
		
		for n in row:
			if n.rstrip() == '':
				puzzle.append(0)
			else: 
				puzzle.append(int(n.rstrip()))

#---------------------------------------------------
# Main
#---------------------------------------------------
instance = int(sys.argv[2])		# Instance of the puzzle, (3x3) or (4x4)
puzzle = [None] * 0				# Empty initial puzzle
nid = 0							# Counter for number of nodes, used for comparison
explored = 0					# Counter for the number of states explored

generatePuzzle(puzzle)

initialState = node(puzzle, None, None)  # Initialize the start state with the input puzzle and no parent

start = time.clock() * 1000.0

if int(sys.argv[1]) == 1:
	solved = aStarSearch(initialState)
else:
	solved = iterativeDeepeningAStarSearch(initialState)

end = time.clock() * 1000.0

s = ""
for n in initialState.state:
	s += str(n) + " "

print("Initial State: " + s)
print("States Explored: " + str(explored))
print("Solution Time: " + str(end - start) + "ms")
print("Solution Depth: " + str(solved.depth))

solutionPath(solved)