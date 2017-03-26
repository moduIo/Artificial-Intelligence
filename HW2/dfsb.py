# Tim Zhang
# 110746199
# CSE537 HW 2
#---------------------------------------------------
import sys
import time
import re

#---------------------------------------------------
# Transforms input file into graph representation
#---------------------------------------------------
def generateCSP():
	global N, M, K, constraints, assignment, domains, arcs

	fin = open(sys.argv[1])
	lines = fin.readlines()

	parameters = re.findall(r'\d+', str(lines[0].split(' ')))

	N = int(parameters[0])
	M = int(parameters[1])
	K = int(parameters[2])
	constraints = [[] for x in range(N)]
	assignment = [None] * N
	domains = [[0] * K for x in range(N)]

	# Initialize domains to all be [0..K-1]
	for d in domains:
		for i in range(0, K):
			d[i] = i

	# Remove CSP parameters
	lines.pop(0)

	# Set constraints
	for l in lines:
		constraint = re.findall(r'\d+', str(l.split(' ')))
		constraints[int(constraint[0])].append(int(constraint[1]))
		constraints[int(constraint[1])].append(int(constraint[0]))

	# Explicitly transform constraints into arcs
	for i in range(0, N):
		for constraint in constraints[i]:
			arcs.append([i, constraint])

#---------------------------------------------------
# Outputs assignment to file.
#---------------------------------------------------
def writeAssignment(solution):
	fout = open(sys.argv[2], "w")

	if solution == "failure":
		fout.write("No Answer")

	else:
		newline = ""

		for s in solution:
			fout.write(newline + str(s))
			newline = "\n"

#---------------------------------------------------
# DFSB implementation
#---------------------------------------------------
def dfsb(end):
	global K, constraints, assignment, searches

	# Timer for comparison
	if time.time() >= end:
		return 'failure'

	if isComplete(assignment):
		return assignment

	var = selectUnassignedVariable(assignment)
	searches += 1

	for color in range(0, K):
		assignment[var] = color

		if isConsistent(assignment, constraints):
			result = dfsb(end)

			if result != 'failure':
				return result

			# Remove {var = value} from assignment
			assignment[var] = None

		else:
			assignment[var] = None

	return 'failure'

#---------------------------------------------------
# DFSB++ implementation
#---------------------------------------------------
def improved_dfsb():
	global K, constraints, assignment, searches, arcs

	if isComplete(assignment):
		return assignment

	a = arcs[:]

	AC3(a)  # Run AC3 in each iteration
	var = MRV_selectUnassignedVariable(assignment)

	if var == 'failure':
		return 'failure'

	colors = LCV(var)
	searches += 1

	for color in colors:
		assignment[var] = color

		if isConsistent(assignment, constraints):
			result = improved_dfsb()

			if result != 'failure':
				return result

			# Remove {var = value} from assignment
			assignment[var] = None

		else:
			assignment[var] = None

	return 'failure'

#---------------------------------------------------
# Checks if an assignment is complete
#---------------------------------------------------
def isComplete(assignment):
	for x in assignment:
		if x is None:
			return False

	return True

#---------------------------------------------------
# Checks if an assignment is consistent
#---------------------------------------------------
def isConsistent(assignment, constraints):
	global N

	for var in range(0, N):
		for constraint in constraints[var]:

			if constraints[var] is None:
				continue

			elif assignment[var] == assignment[constraint] and assignment[constraint] is not None:
				return False

	return True

#---------------------------------------------------
# Returns the first unassigned variable
#---------------------------------------------------
def selectUnassignedVariable(assignment):
	for position, x in enumerate(assignment):
		if x is None:
			return position

#---------------------------------------------------
# Returns the first unassigned variable using the 
# MRV heuristic
#---------------------------------------------------
def MRV_selectUnassignedVariable(assignment):
	global K
	mrv = K	   # Current MRV
	var = 0    # Index of MRV variable

	for position, x in enumerate(assignment):
		# If the variable is unassigned
		if x is None:
			remaining = K - calculateIllegalValues(position)

			if remaining == 0:
				return 'failure'

			elif remaining <= mrv:
				mrv = remaining
				var = position

	return var

#---------------------------------------------------
# Returns the amount of illegal values remaining on 
# the variable indexed by the argument
#---------------------------------------------------
def calculateIllegalValues(index):
	global constraints, assignment, domains
	illegalValues = set()

	for var in constraints[index]:
		if assignment[var] is not None:
			illegalValues.add(assignment[var])
			if int(assignment[var]) in domains[index]:
				domains[index].remove(int(assignment[var]))

	return len(illegalValues)

#---------------------------------------------------
# Returns the values for the variable ordered by
# LCV heuristic by ordering the assignments by the
# number of remaining colors that the neighbors
# can still take on.
#---------------------------------------------------
def LCV(var):
	global K, assignment, constraints
	colors = [0] * K
	lcv = [0] * K
	illegalValues = set()

	# If there are no constraints on the variable the order doesn't matter
	if constraints[var] is None:
		for i in range(0, K):
			lcv[i] = i
		return lcv

	for color in range(0, K):
		illegalValues.add(color)

		# Count the number colors all constrained neighbors can still be assigned
		for neighbor in constraints[var]:
			if assignment[neighbor] is None:
				if constraints[neighbor] is not None:
					# Look at each neighbor of the neighbor
					for n_neighbor in constraints[neighbor]:
						# If the constrained neighbor has an assignment then the neighbor can not have the same assignment
						if assignment[n_neighbor] is not None:
							illegalValues.add(assignment[n_neighbor])

					# Add the amount of legal values for the neighbor
					colors[color] += K - len(illegalValues)
					illegalValues.clear()
					illegalValues.add(color)

	# Map the colors array onto LCV array
	for i in range(0, K):
		lcv[i] = colors.index(max(colors))
		colors[colors.index(max(colors))] = -1

	return lcv

#---------------------------------------------------
# Implements AC3 algorithm
#---------------------------------------------------
def AC3(arcs):
	global constraints, N

	# Remove inconsistent values from domains of all arcs
	while arcs:
		arc = arcs.pop(0)
		if removeInconsistentValues(arc):
			# For each neighbor that isn't the current neighbor
			for neighbor in constraints[arc[0]]:
				if neighbor != arc[1]:
					# Add it's constraint to the current source to the queue
					arcs.append([neighbor, arc[0]])

#---------------------------------------------------
# Implements removeInconsistentValues subroutine
# INPUT: arc[0] is x and arc[1] is y where (x, y)
#        is the directed edge.
#---------------------------------------------------
def removeInconsistentValues(arc):
	global prunings, domains
	removed = False
	satisfiable = False

	for xcolor in domains[arc[0]]:
		for ycolor in domains[arc[1]]:
			if ycolor != xcolor:
				satisfiable = True

		if not satisfiable:
			domains[arc[0]].remove(xcolor)
			removed = True
			prunings += 1

		satisfiable = False

	return removed

#---------------------------------------------------
# Main
#---------------------------------------------------
N = 0               # Number of variables
M = 0               # Number of constraings
K = 0               # Size of domain
searches = 0		# Number of search calls
prunings = 0		# Number of arc prunings
constraints = []    # List of constraints with the interpretation that the ith index is a list of constraints for the ith variable
assignment = []     # Assignment of colors to variables
domains = []		# Domains used in AC3
arcs = []  			# Queue of arcs

generateCSP()

start = time.clock() * 1000.0

if int(sys.argv[3]) == 0:
	solution = dfsb(time.time() + 60)
else:
	solution = improved_dfsb()

end = time.clock() * 1000.0

writeAssignment(solution)
print("Solution Time: " + str(end - start) + "ms")

print("Searches: " + str(searches))
print("Prunings: " + str(prunings))