import random
import sys

# moves = UP, RIGHT, DOWN, LEFT
moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]

def isPositionLegal(board, x, y):
    n = len(board)
    return ((x >= 0) and (x < n) and (y >= 0) and (y < n))

def nextPos(x,y, move):
    nextX = x + move[0]
    nextY = y + move[1]

    return nextX, nextY

def canMove(board, direction):

    mv = moves[direction]
    x, y = findGap(board)
    x2, y2 = nextPos(x, y, mv)

    return isPositionLegal(board, x2, y2)

# def canMove(board):
#     x, y = findGap(board)
#
#     for mv in moves:
#         x2, y2 = nextPos(x, y, mv)
#         if isPositionLegal(board, x2, y2):
#             return True
#
#     return False

def possibleMoves(board):

    global moves
    x, y = findGap(board)

    res = []
    for mv in moves:
        x2, y2 = nextPos(x, y, mv)
        if isPositionLegal(board, x2, y2):
            res.append(mv)

    return res


def moveGap(board, move):

    x, y = findGap(board)
    x2, y2 = nextPos(x, y, move)

    tmp = board[x][y]
    board[x][y] = board[x2][y2]
    board[x2][y2] = tmp

def findGap(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                return i,j
    return -1, -1

def printBoard(board):

    print("")
    for row in board:
        row_str = ""
        for cell in row:
            row_str += str(cell) + " "
        print(row_str)


if __name__ == '__main__':

    n = 0
    k = -1
    out_file = ''

    if len(sys.argv) == 4:
        n = int(sys.argv[1])
        k = int(sys.argv[2])
        out_file = open(sys.argv[3], 'w')
    elif len(sys.argv) == 3:
        n = int(sys.argv[1])
        out_file = open(sys.argv[2], 'w')
    else:
        print('Wrong number of arguments. Usage:\npuzzleGenerator.py <N> <K - number of moves> <OUTPATH>\npuzzleGenerator.py <N> <OUTPATH>')
    print('n = ' + str(n))

    if k == -1:
        a = list(range(1, n*n + 1))
        random.shuffle(a)

        for i in range(n):
            for j in range(n):
                cur = a[i * n + j]
                if cur == (n*n):
                    out_file.write('')
                else:
                    out_file.write(str(cur))
                if j != (n-1):
                    out_file.write(',')
            out_file.write('\n')
    else:
        board = []
        for i in range(n):
            board.append([])
            for j in range(n):
                if (n*i+j+1) == n*n:
                    board[i].append(0)
                else:
                    board[i].append(n * i + j + 1)

        printBoard(board)

        for move_cnt in range(k):
            pos_moves = possibleMoves(board)
            move = random.choice(pos_moves)
            moveGap(board, move)

        printBoard(board)

        for row in board:
            for i in range(len(row)):
                cell = row[i]
                if cell != 0:
                    out_file.write(str(cell))
                if i != (len(row) - 1):
                    out_file.write(",")


            out_file.write("\n")

    out_file.close()
