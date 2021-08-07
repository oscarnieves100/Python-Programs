"""
Tic-Tac-Toe game (vs Computer)

By: Oscar A. Nieves
Updated: August 7, 2021
"""
from random import randint

# Main game
def TicTacToe():
    """
    To play: enter a number from 1 to 9 corresponding to board 
    positions when prompted. First row: 1 to 3, 2nd row: 
    4 to 6, and 3rd row: 7 to 9.
    """
    
    # Initialize game board
    board = {str(n):' ' for n in range(1,10)}
    
    # Game start
    human = str(input('Choose X or O: '))
    if human == 'X':
        computer = 'O'
    else:
        computer = 'X'
    print(TicTacToe.__doc__)
    printBoard(board)
    while True:
        # Player X
        checkMove('X',board,computer)
        status = checkWin('X',board)
        if status == True:
            break
        
        # Player O
        checkMove('O',board,computer)
        status = checkWin('O',board)
        if status == True:
            break
    return
        
# Check move
def checkMove(S,moves,C):
    # S is a string 'X' or 'O', same with C
    if S == C: # computer's turn
        N = randint(1,9)
    else:
        N = input(S + ' move: ')
    if (int(N) in range(1,10)) == False:
        if S != C:
            print('Number must be between 1 and 9, try again...')
        checkMove(S,moves,C)
    else:
        if moves[str(N)] == ' ':
            moves[str(N)] = S
            printBoard(moves)
            return moves
        else:
            if S != C:
                print('Position ' + str(N) + ' taken, try again...')
            checkMove(S,moves,C)
    
# Print board
def printBoard(moves):
    print(' ')
    print(moves['1']+'|'+moves['2']+'|'+moves['3'])
    print('-----')
    print(moves['4']+'|'+moves['5']+'|'+moves['6'])
    print('-----')
    print(moves['7']+'|'+moves['8']+'|'+moves['9'])
    print(' ')
    
# Check Win
def checkWin(S,moves):    
    # Horizontal win
    if (moves['1']==moves['2']==moves['3']==S or 
        moves['4']==moves['5']==moves['6']==S or
        moves['7']==moves['8']==moves['9']==S):
        print(S + ' wins!')
        return True
    
    # Vertical win
    elif (moves['1']==moves['4']==moves['7']==S or 
        moves['2']==moves['5']==moves['8']==S or
        moves['3']==moves['6']==moves['9']==S):
        print(S + ' wins!')
        return True
    
    # Diagonal win
    elif (moves['1']==moves['5']==moves['9']==S or 
            moves['3']==moves['5']==moves['7']==S):
        print(S + ' wins!')
        return True

    # Draw
    elif (' ' in moves.values()) == False:
        print('Draw!')
        return True
    else:
        return False
    
TicTacToe()