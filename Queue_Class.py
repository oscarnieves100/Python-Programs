"""
A queue class

Author: Oscar A. Nieves
Updated: August 8, 2021
"""
class Queue:
    def __init__(self):
        self._list = []
    
    def __len__(self):
        return len(self._list)    
    
    def enqueue(self, element):
        self._list.append(element)
    
    def dequeue(self):
        return self._list.pop(0)
    
    def peek(self):
        return self._list[0]

q = Queue()
q.enqueue('A')
print( q.peek() )
q.enqueue('B')
q.enqueue('C')
print( q.peek() )
print( len(q) )
q.dequeue()
print( q.peek() ) # remove 1st element
