"""
Function that takes in a string, eliminates any
repeated adjacent characters until there are no
more adjacent characters that are the same

Author: Oscar A. Nieves
Updated: August 8, 2021
"""
def cabbage(string_var):    
    stop = False
    while stop == False:
        counter = 0
        array = []
        for j in range(0,len(string_var)):
            array.append(string_var[j])
    
        for j in range(1,len(string_var)):
            if array[j] == array[j-1]:
                array.pop(j-1)
                array.pop(j-1)
                counter = 1
                break
        
        if counter == 0:
            break
        else:
            new_string = ''
            for j in range(0,len(array)):
                new_string += array[j]
        
            string_var = new_string
    
    return string_var

new = cabbage('jsufhuooueeefjjjfbbb')
new0 = cabbage(new)
print(new)
print(new0)