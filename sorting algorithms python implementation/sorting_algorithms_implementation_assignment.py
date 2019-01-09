
# coding: utf-8

# # Module Import

# In[1]:


import numpy as np
import time 
import math
import sys
import timeit
from threading import Thread
import csv


# # Insertion Sort

# In[2]:


#implementation of insertion sort algorithm
def insertion_sort(unsorted_is):
    for j in range(1, unsorted_is.shape[0]):
        key = unsorted_is[j]
        i = j-1
        while i>-1 and unsorted_is[i]>=key:
            unsorted_is[i+1] = unsorted_is[i]
            i-=1
            
        unsorted_is[i+1] = key
    
    return unsorted_is


# # Selection Sort

# In[3]:


#implementation of selection sort algorithm
def selection_sort(unsorted_ss):
    for i in range(unsorted_ss.shape[0] - 1):
        lowindex = i #unsorted_ss_min = unsorted_ss[i]
        for j in reversed(range(i + 1, unsorted_ss.shape[0])):
            if unsorted_ss[j]<unsorted_ss[lowindex]:
                lowindex=j #unsorted_ss[i] = unsorted_ss[j]
                #unsorted_ss[j] = unsorted_ss_min
        unsorted_ss[i], unsorted_ss[lowindex] = unsorted_ss[lowindex], unsorted_ss[i]
    
        
    return unsorted_ss


# # Merge Sort

# In[4]:


###implementation of merge sort algorithm

#definition of merge function
def merge(left_sub_array, right_sub_array):
    merged_array = np.zeros(left_sub_array.shape[0] + right_sub_array.shape[0])
    
    left_index = 0
    right_index = 0
    
    for i in range(merged_array.shape[0]):
        if left_sub_array[left_index] <= right_sub_array[right_index] :
            merged_array[i] = left_sub_array[left_index]
            if not (left_index + 1) >= left_sub_array.shape[0]:
                left_index +=1
            else:
                left_sub_array[left_index] = sys.maxsize #sys.maxsize used as ininfity integer number
        else :
            merged_array[i] = right_sub_array[right_index]
            if not (right_index + 1) >= right_sub_array.shape[0]:
                right_index +=1
            else:
                right_sub_array[right_index] = sys.maxsize
        
    return merged_array

#definition of merge_sort function
def merge_sort(unsorted_ms):
    if unsorted_ms.shape[0] == 1 :
        return unsorted_ms
    else :
        L_sub_array, R_sub_array = np.array_split(ary=unsorted_ms, indices_or_sections=2)
        
        L_sub_array = merge_sort(L_sub_array)
        R_sub_array = merge_sort(R_sub_array)
        
        return merge(L_sub_array, R_sub_array)


# # Quick Sort

# In[5]:


### implementation of quick sort algorithm

#definition of partition function
def partition (unpartitioned):
    unpartitioned_size = unpartitioned.shape[0]
    
    pivot = unpartitioned[unpartitioned_size - 1]
    min_index = -1
    
    for i in range(unpartitioned_size-1):
        if unpartitioned[i] <= pivot :
            min_index += 1
            unpartitioned[min_index], unpartitioned[i] = unpartitioned[i], unpartitioned[min_index]
            
    unpartitioned[min_index + 1], unpartitioned[unpartitioned_size - 1] = unpartitioned[unpartitioned_size - 1], unpartitioned[min_index + 1]
    
    return min_index + 1 , unpartitioned

def quick_sort(unsorted_qs):
    if unsorted_qs.shape[0] <= 1 :
        return unsorted_qs
    else:
        partition_index, unsorted_qs = partition(unsorted_qs)
        left_qs = unsorted_qs[:partition_index]
        pivot_qs = np.array((1,))
        pivot_qs[0] = unsorted_qs[partition_index]
        right_qs = unsorted_qs[partition_index + 1 :]
        
        left_qs = quick_sort(left_qs)
        right_qs = quick_sort(right_qs)
        
        return np.concatenate((left_qs, pivot_qs, right_qs))
        


# # Heap Sort

# In[5]:


### implementation of heap sort algorithm

#definition of max_heapify function
def max_heapify (non_heap_array, heap_start_index):
    largest = heap_start_index
    heap_size = non_heap_array.shape[0] - 1
    #if left_child_index > (non_heap_array.shape[0] - 1)
    left_child_index = 2*heap_start_index + 1
    right_child_index = left_child_index + 1
    
    heap_size = non_heap_array.shape[0]
    
    if left_child_index < heap_size and non_heap_array[left_child_index] > non_heap_array[heap_start_index] :
        largest = left_child_index
    else :
        largest = heap_start_index
    
    if right_child_index < heap_size and non_heap_array[right_child_index] > non_heap_array[largest] :
        largest = right_child_index
    
    if not (largest == heap_start_index) :
        non_heap_array[heap_start_index], non_heap_array[largest] = non_heap_array[largest], non_heap_array[heap_start_index]
        non_heap_array = max_heapify(non_heap_array, largest)
        
    return non_heap_array

#definition of build_max_heap function
def build_max_heap (non_max_heap) :
    for i in reversed(range(math.floor((non_max_heap.shape[0] - 1)/2))) :
        non_max_heap = max_heapify(non_max_heap, i)
        
    return non_max_heap

#definition of heap_sort function
def heap_sort(unsorted_hs) :
    unsorted_hs = build_max_heap(unsorted_hs)
    
    for i in reversed(range(1,unsorted_hs.shape[0])) :
        unsorted_hs[0], unsorted_hs[i] = unsorted_hs[i], unsorted_hs[0]
        temp_array = unsorted_hs[:i] #simulating heap_size - 1 of unsorted_hs
        unsorted_hs = np.concatenate((max_heapify(temp_array, 0), unsorted_hs[i:]), axis=0)
        
    return unsorted_hs


# # Start of Tests

# In[8]:


##prepare intervals size (input size)
start = 0
stop = 10000
step = 20
rep = 4
range_step = int(stop/step)

intervals_array = np.empty((0,1), dtype=int)
for i in range(start, stop+1, range_step):
    intervals_array = np.append(intervals_array, int(i))


# In[9]:


intervals_array


# In[10]:


#generate arrays for each interval size

#dictionary of arrays with presortedness=0 for each interval
dict0 = {0:np.array([1])}
j = 1
for i in intervals_array[1:]:
    dict0[j] = np.flip(np.sort(np.random.randint(0, high=i, size=i)), axis=0)
    j += 1
    
#dictionary of arrays with presortedness=0.5 for each interval
dict0_5 = {0:np.array([1])}
j = 1
for i in intervals_array[1:]:
    dict0_5[j] = np.random.randint(start, high=i, size=i)
    j += 1    
    
#dictionary of arrays with presortedness=1 for each interval
dict1 = {0:np.array([1])}
j = 1
for i in intervals_array[1:]:
    dict1[j] = np.sort(np.random.randint(0, high=i, size=i))
    j += 1



# ### Thread to run sort of the whole intervals concurently

# In[11]:


class my_thread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


# ## Test for presortedness = 0

# In[12]:


#(number of algorithms X number of intervals) array holding the runtime of each algorithm(row)
#for sorting each interval's(column) array with presortedness=0
rt0 = np.zeros((5,21))
dict0_Isorted={}
dict0_Ssorted={}
dict0_Msorted={}
dict0_Qsorted={}
dict0_Hsorted={}

range_in = len(dict0)


# ### testing insertion sort

# In[ ]:


###testing insertion sort
def isort_run_time(dict0_Isorted, dict0, intervals):
    start_time = time.clock()
    dict0_Isorted[intervals] = insertion_sort(dict0[intervals])
    end_time = time.clock()
    
    return end_time - start_time

sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=isort_run_time, args=(dict0_Isorted, dict0, intervals))
        sorting[intervals].start()
        rt0[0][intervals] = sorting[intervals].join()


# In[13]:


### display insertion sort run time
print(rt0[0][:])#step = 10000


# ### testing selection sort

# In[ ]:


###testing selection sort
def ssort_run_time(dict0_Ssorted, dict0, intervals):
    start_time = time.clock()
    dict0_Ssorted[intervals] = selection_sort(dict0[intervals])
    end_time = time.clock()
    
    return end_time - start_time

sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=ssort_run_time, args=(dict0_Ssorted, dict0, intervals))
        sorting[intervals].start()
        rt0[1][intervals] = sorting[intervals].join()


# In[41]:


### display selection sort run time
print(rt0[1][:]) #step = 10000


# ### testing merge sort

# In[66]:


###testing merge sort
def msort_run_time(dict0_Msorted, dict0, intervals):
    start_time = time.clock()
    dict0_Msorted[intervals] = merge_sort(dict0[intervals])
    end_time = time.clock()
    
    return end_time - start_time

sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=msort_run_time, args=(dict0_Msorted, dict0, intervals))
        sorting[intervals].start()
        rt0[2][intervals] = sorting[intervals].join()


# In[67]:


### display merge sort run time
print(rt0[2][:])#step = 10000


# ### testing quick sort

# In[ ]:


### ATTENTION!!!! increasing recursion limit might cause overflow and python might crush
sys.setrecursionlimit(5000)

###testing quick sort
def qsort_run_time(dict0_Qsorted, dict0, intervals):
    start_time = time.clock()
    dict0_Qsorted[intervals] = quick_sort(dict0[intervals])
    end_time = time.clock()
    
    return end_time - start_time

sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=qsort_run_time, args=(dict0_Qsorted, dict0, intervals))
        sorting[intervals].start()
        rt0[3][intervals] = sorting[intervals].join()


# In[43]:


### display quick sort run time
print(rt0[3][:])#step = 10000


# ### testing heap sort

# In[46]:


###testing heap sort
def hsort_run_time(dict0_Hsorted, dict0, intervals):
    start_time = time.clock()
    dict0_Hsorted[intervals] = heap_sort(dict0[intervals])
    end_time = time.clock()
    
    return end_time - start_time

sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=hsort_run_time, args=(dict0_Hsorted, dict0, intervals))
        sorting[intervals].start()
        rt0[4][intervals] = sorting[intervals].join()


# In[47]:


### display heap sort run time
print(rt0[4][:])#step = 10000


# In[32]:


### exporting runtime array into a .csv file
with open('presorted0_10000.csv', 'w') as p0:
    #p0.write('''Intervals,Insertion sort,Selection sort,Merge sort,Quick sort,Heap sort''')
    writer = csv.writer(p0)
    writer.writerows(np.transpose(rt0))


# ## Test for presortedness = 0.5

# In[25]:


#(number of algorithms X number of intervals) array holding the runtime of each algorithm(row)
#for sorting each interval's(column) array with presortedness=0.5
rt0_5 = np.zeros((5,21))
dict0_5_Isorted={}
dict0_5_Ssorted={}
dict0_5_Msorted={}
dict0_5_Qsorted={}
dict0_5_Hsorted={}

range_in = len(dict0_5)


# ### testing insertion sort

# In[53]:


###testing insertion sort
sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=isort_run_time, args=(dict0_5_Isorted, dict0_5, intervals))
        sorting[intervals].start()
        rt0_5[0][intervals] = sorting[intervals].join()


# In[54]:


### display insertion sort run time
print(rt0_5[0][:])#step = 10000


# ### testing selection sort

# In[55]:


###testing selection sort
sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=ssort_run_time, args=(dict0_5_Ssorted, dict0_5, intervals))
        sorting[intervals].start()
        rt0_5[1][intervals] = sorting[intervals].join()


# In[56]:


### display selection sort run time
print(rt0_5[1][:])#step = 10000


# ### testing merge sort

# In[61]:


###testing merge sort
sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=msort_run_time, args=(dict0_5_Msorted, dict0_5, intervals))
        sorting[intervals].start()
        rt0_5[2][intervals] = sorting[intervals].join()


# In[62]:


### display merge sort run time
print(rt0_5[2][:])#step = 10000


# ### testing quick sort

# In[ ]:


###testing quick sort
sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=qsort_run_time, args=(dict0_5_Qsorted, dict0_5, intervals))
        sorting[intervals].start()
        rt0_5[3][intervals] = sorting[intervals].join()


# In[47]:


### display quick sort run time
print(rt0_5[3][:])#step = 10000


# ### testing heap sort

# In[70]:


###testing heap sort
sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=hsort_run_time, args=(dict0_5_Hsorted, dict0_5, intervals))
        sorting[intervals].start()
        rt0_5[4][intervals] = sorting[intervals].join()


# In[71]:


### display heap sort run time
print(rt0_5[4][:])#step = 10000


# In[33]:


### exporting runtime array into a .csv file
with open('presorted0_5_10000.csv', 'w') as p0_5:
    #p0.write('''Intervals,Insertion sort,Selection sort,Merge sort,Quick sort,Heap sort''')
    writer = csv.writer(p0_5)
    writer.writerows(np.transpose(rt0_5))


# ## Test for presortedness = 1

# In[28]:


#(number of algorithms X number of intervals) array holding the runtime of each algorithm(row)
#for sorting each interval's(column) array with presortedness=1
rt1 = np.zeros((5,21))
dict1_Isorted={}
dict1_Ssorted={}
dict1_Msorted={}
dict1_Qsorted={}
dict1_Hsorted={}

range_in = len(dict1)


# ### testing insertion sort

# In[75]:


###testing insertion sort
sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=isort_run_time, args=(dict1_Isorted, dict1, intervals))
        sorting[intervals].start()
        rt1[0][intervals] = sorting[intervals].join()


# In[76]:


### display insertion sort run time
print(rt1[0][:])#step = 10000


# ### testing selection sort

# In[77]:


###testing selection sort
sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=ssort_run_time, args=(dict1_Ssorted, dict1, intervals))
        sorting[intervals].start()
        rt1[1][intervals] = sorting[intervals].join()


# In[78]:


### display selection sort run time
print(rt1[1][:])#step = 10000


# ### testing merge sort

# In[79]:


###testing merge sort
sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=msort_run_time, args=(dict1_Msorted, dict1, intervals))
        sorting[intervals].start()
        rt1[2][intervals] = sorting[intervals].join()


# In[83]:


### display merge sort run time
print(rt1[2][:])#step = 10000


# ### testing quick sort

# In[ ]:


###testing quick sort
sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=qsort_run_time, args=(dict1_Qsorted, dict1, intervals))
        sorting[intervals].start()
        rt1[3][intervals] = sorting[intervals].join()


# In[55]:


### display quick sort run time
print(rt1[3][:])#step = 10000


# ### testing heap sort

# In[84]:


###testing heap sort
sorting = [None]*range_in
for reptition in range(rep):
    for intervals in range(range_in):
        sorting[intervals] = my_thread(target=hsort_run_time, args=(dict1_Hsorted, dict1, intervals))
        sorting[intervals].start()
        rt1[4][intervals] = sorting[intervals].join()


# In[85]:


### display heap sort run time
print(rt1[4][:])#step = 10000


# In[34]:


### exporting runtime array into a .csv file
with open('presorted1_10000.csv', 'w') as p1:
    #p0.write('''Intervals,Insertion sort,Selection sort,Merge sort,Quick sort,Heap sort''')
    writer = csv.writer(p1)
    writer.writerows(np.transpose(rt1))


# # End of tests
