"""
Product of an array except self.

Task: create a function that outputs an array B (arbitrary
length) in which each element is calculated from the product of every
element in another array A, except for the element 
corresponding to the same index.

For example: A = [1,2,3]
Then B[0] = 2*3 = 6
     B[1] = 1*3 = 3
     B[2] = 1*2 = 2
--> B = [6,3,2]     

Author: Oscar A. Nieves
Updated: August 8, 2021
"""
class Array:
    """ Creates array with product method """
    def __init__(self,value=[]):
        self._value = value
    
    def peek(self):
        return self._value
    
    def __len__(self):
        return len(self._value)
    
    def prod(self, exception):
        value = 1
        for n in range(0,len(self._value)):
            if n == exception:
                continue
            else: 
                value *= self._value[n]
        return value
        
    def prodArray(self):
        new_array = []
        for i in range(0,len(self._value)):
            P = self.prod(i)
            new_array.append( P )
        return Array(new_array)

A = Array([1,2,3,4,5,6])
B = A.prodArray()
print('A = ', A.peek())
print('B = ', B.peek())