"""
Tic-Tac-Toe game

By: Oscar A. Nieves
Updated: August 7, 2021
"""

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
    print(TicTacToe.__doc__)
    printBoard(board)
    while True:
        # Player X
        checkMove('X',board)
        status = checkWin('X',board)
        if status == True:
            break
        
        # Player O
        checkMove('O',board)
        status = checkWin('O',board)
        if status == True:
            break
    return
        
# Check move
def checkMove(S,moves):
    # S is a string 'X' or 'O'
    N = input(S + ' move: ')
    if (int(N) in range(1,10)) == False:
        print('Number must be between 1 and 9, try again...')
        checkMove(S,moves)
    else:
        if moves[str(N)] == ' ':
            moves[str(N)] = S
            printBoard(moves)
            return moves
        else:
            print('Position ' + str(N) + ' taken, try again...')
            checkMove(S,moves)
    
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