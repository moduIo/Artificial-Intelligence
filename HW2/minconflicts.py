# Tim Zhang
# 110746199
# CSE537 HW 2
#---------------------------------------------------
import sys
import time
import re
import random

#---------------------------------------------------
# Implements Min Conflicts algorithm
#---------------------------------------------------
def minConflicts(maxSteps):
	global N, M, K, assignment, constraints, searches

	# Generate a random initial assignment
	greedyAssignment()

	# Run the algorithm
	for i in range(0, maxSteps):
		if isConsistent(assignment, constraints):
			return True

		var = pickConflictedVariable()
		assignment[var] = minimizeConflicts(var)

		searches += 1

	return False

#---------------------------------------------------
# Greedily assigns each variable the best value
#---------------------------------------------------
def greedyAssignment():
	global assignment, constraints, N

	for i in range(0, N):
		assignment[i] = minimizeConflicts(i)

#---------------------------------------------------
# Returns a random conflicted variable
#---------------------------------------------------
def pickConflictedVariable():
	global assignment, constraints
	conflicted = []

	for i in range(0, N):
		for constraint in constraints[i]:
			if assignment[i] == assignment[constraint]:
				conflicted.append(i)
				break

	return random.choice(conflicted)

#---------------------------------------------------
# Returns an assignment which minimizes conflicts
#---------------------------------------------------
def minimizeConflicts(var):
	global assignment, constraints, K
	conflicts = []  # List of conflicts
	conflict = 0

	# Measure the amount of conflicts per assignment
	for i in range(0, K):
		for neighbor in constraints[var]:
			if assignment[neighbor] == i:
				conflict += 1

		# Add the total number of conflicts to the list
		conflicts.append(conflict)
		conflict = 0

	minimum = min(conflicts)
	indices = [i for i, v in enumerate(conflicts) if v == minimum]
	
	return random.choice(indices)

#---------------------------------------------------
# Transforms input file into graph representation
#---------------------------------------------------
def generateCSP():
	global N, M, K, constraints, assignment

	fin = open(sys.argv[1])
	lines = fin.readlines()

	parameters = re.findall(r'\d+', str(lines[0].split(' ')))

	N = int(parameters[0])
	M = int(parameters[1])
	K = int(parameters[2])
	constraints = [[] for x in range(N)]
	assignment = [None] * N

	# Remove CSP parameters
	lines.pop(0)

	# Set constraints
	for l in lines:
		constraint = re.findall(r'\d+', str(l.split(' ')))
		constraints[int(constraint[0])].append(int(constraint[1]))
		constraints[int(constraint[1])].append(int(constraint[0]))

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
# Main
#---------------------------------------------------
N = 0               # Number of variables
M = 0               # Number of constraings
K = 0               # Size of domain
searches = 0		# Number of search calls
constraints = []    # List of constraints with the interpretation that the ith index is a list of constraints for the ith variable
assignment = []     # Assignment of colors to variables
failure = True      # Fail flag of algorithm

generateCSP()

start = time.clock() * 1000.0
current = (time.clock() * 1000.0) - start

# Random restart
while current < 60000:
	if minConflicts(100000):
		writeAssignment(assignment)
		print(assignment)
		failure = False
		break

	print('Restart')
	# Random restart
	assignment = [None] * N
	current = (time.clock() * 1000.0) - start

if failure:
	writeAssignment('failure')
	print('Failure')

end = time.clock() * 1000.0

print("Solution Time: " + str(end - start) + "ms")
print("Searches: " + str(searches))