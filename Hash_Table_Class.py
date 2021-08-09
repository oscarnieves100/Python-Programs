"""
A hash table (dictionary) class that uses
linear probing to make sure no collisions occur.

Author: Oscar A. Nieves
Updated: August 9, 2021
"""
class HashTable:
    def __init__(self,MAX=10):
        self.MAX = MAX
        self.arr = [None for i in range(self.MAX)]
        self.keys = [ [], [] ]

    def get_hash(self, key):
        h = 0
        for char in key:
            h += ord(char)
            return h % self.MAX
    
    # Check collision using linear probing
    def checkCollision(self,hash_number):
        h = hash_number
        if self.arr[h] != None:
            return True
        else:
            return False
        
    # This function finds a suitable slot for the value
    def __setitem__(self,key,val):
        h = self.get_hash(key)
        if self.arr[h] == None:
            self.arr[h] = val
            check_col = False
        else:
            check_col = True
            counter = 0
            while check_col == True:
                if (h+1) < self.MAX:
                    h += 1
                    check_col = self.checkCollision(h)
                else:
                    h = 0
                    check_col = self.checkCollision(h)
                counter += 1
                if counter > self.MAX:
                    raise Exception('Hashmap full!')
            self.arr[h] = val
            self.keys[0].append(key)
            self.keys[1].append(h)
        
    def __getitem__(self,key):
        h = None
        for i in range(len(self.keys[0])):
            if key == self.keys[0][i]:
                h = self.keys[1][i]
                break
            else:
                continue
        else:    
            h = self.get_hash(key)
        return self.arr[h]
        
    def __delitem__(self,key):
        h = None
        for i in range(len(self.keys[0])):
            if key == self.keys[0][i]:
                h = self.keys[1][i]
                break
            else:
                continue
        if h == None:  
            h = self.get_hash(key)
        self.arr[h] = None
        self.keys[0][i] = None
        self.keys[1][i] = None

def main():
    A = HashTable()
    h1 = A.get_hash('march 6')
    h2 = A.get_hash('march 17')
    print(h1)
    print(h2)
    A['march 6'] = 540
    A['march 17'] = 320
    print( A['march 6'] )
    print( A['march 17'] )
    
if __name__ == '__main__':
    main()