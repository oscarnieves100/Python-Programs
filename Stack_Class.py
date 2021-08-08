"""
A Stack class

Author: Oscar A. Nieves
Updated: August 8, 2021
"""
class Stack:
    def __init__(self):
        self._list = []
    
    def __len__(self):
        return len(self._list)    
    
    def push(self, element):
        self._list.append(element)
    
    def pop(self):
        return self._list.pop(-1)
    
    def peek(self):
        return self._list[-1]
    
    def see(self):
        return self._list

s = Stack()
s.push('A')
print( s.peek() )
s.push('B')
s.push('C')
print( s.peek() )
print( len(s) )
s.pop()
print( s.peek() ) # remove 1st element
