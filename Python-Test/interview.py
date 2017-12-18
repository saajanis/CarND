from treeset import TreeSet
import random

elem_list = [];
treeset = TreeSet([]);


def add_brute(n):
    elem_list.append(n);

def add_smart(n):
    treeset.add(n);


def max_xor_brute(n):
    max = -1;
    for e in elem_list:
        res = n ^ e;
        if res > max:
            max = res;
    return max;

def max_xor_smart(n):
    return max([n ^ treeset.floor(~n), n ^ treeset.ceiling(~n)]);2




def generate_random_list(n, left, right):
    a = []
    for j in range(n):
        a.append(random.randint(left, right))
    #print('Randomised list is: ', a)
    return a

correct = 0
incorrect = 0
for i in range(200):
    elem_list = [];
    treeset = TreeSet([]);

    left = random.randint(1, 10)
    right = random.randint(left, left + 400000)
    arr = generate_random_list(20, left, right)
    #print "--arr = ",arr
    num = random.randint(left, right)

    for a in arr:
        add_brute(a);
        add_smart(a);

    try:
        print(max_xor_brute(num))
        print(max_xor_smart(num))
        assert max_xor_brute(num) == max_xor_smart(num)
        correct += 1
    except AssertionError:
        #global_list.append("error")
        print ("Assert error at")
        incorrect += 1

print('Percentage correct: ' + str(correct / (correct + incorrect)))

#print "global list = ",len(global_list)