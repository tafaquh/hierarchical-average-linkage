
## Hierarchical Clustering using Average Linkage


```python
from pprint import pprint
import math, csv, copy, sys, time, pandas as pd
```


```python
#variables
k = 3 #number of cluster
m_distance = [] #matrix distances between two points
m_data = [] #matrix of data training 

```


```python
#method read iris file
def readFile(m_data):
    with open(str('iris.data.txt'), 'r') as csvfile:
        lines = csv.reader(csvfile)
        for row in lines:
            row.pop(4)
            m_data.append([ float(x) for x in row ])
```


```python
#method for count distance between two data
def euclideanDistance(p, q):
    result = 0
    # Substract each coordinate point 
    #(p1 - q1, p2 - q2, ... pn - qn)

    for idx,item in enumerate(p):
        result += math.pow(math.fabs(item - q[idx]),2)
    return math.pow(result,0.5)    
```


```python
#method for measure distances
def distances():
    for idx,datum in enumerate(m_data):
        distance = [] #matriks distances between target and data
        if(idx == 0): distance.append(0)
        else:
            for i in range(0, idx+1):
                distance.append(euclideanDistance(m_data[idx], m_data[i]))
        m_distance.append(distance)

```


```python
readFile(m_data)
distances()
m_cluster = dict(zip([x for x in range(0,len(m_distance))], [[x] for x in range(0,len(m_distance))]))
```


```python
def getClusterDistance(x,y):
    averages = []
    
    for x_item in m_cluster[x]:
        for y_item in m_cluster[y]:
            #print(x_item,y_item)
            averages.append(m_distance[max(x_item,y_item)][min(x_item,y_item)])
    return sum(averages) / len(averages)

```


```python
start_time = time.time()

while(len(m_cluster.keys()) > k):
    print("cluster length so far: "+str(len(m_cluster.keys())))
    #update cluster length
    c_length = len(m_cluster.keys())
    
    try:
        for idx in range(0, c_length):
            key = list(m_cluster.keys())[idx]
            closest_distance = []
            closest_index = []
            
            #Get closest cluster
            for j,target in enumerate(m_cluster.keys()):
                if(target >= key+1):
                    closest_distance.append(getClusterDistance(target, key))
                    closest_index.append([target,key])
            try:
                new_cluster = closest_index[closest_distance.index(min(closest_distance))][0]
                
                #Merge cluster
                if(new_cluster not in m_cluster[key]):
                    for item in m_cluster[new_cluster]:
                        m_cluster[key].append(item)
                    m_cluster.pop(new_cluster)
                    c_length = len(m_cluster.keys())

            except Exception:
                continue
                
    except IndexError:
        continue

elapsed_time = time.time() - start_time
print("time elapse: "+str(elapsed_time))
```

    cluster length so far: 150
    cluster length so far: 75
    cluster length so far: 38
    cluster length so far: 19
    cluster length so far: 10
    cluster length so far: 5
    time elapse: 0.021824359893798828
    


```python
#pprint(m_cluster)
```


```python
data = []
with open(str('iris.data.txt'), 'r') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        data.append(row)
        
for item in m_cluster.keys():
    for value in m_cluster[item]:
        data[value].append(item)
#pprint(m_data)
```


```python
pprint(data)
```

    [['5.1', '3.5', '1.4', '0.2', 'Iris-setosa', 0],
     ['4.9', '3.0', '1.4', '0.2', 'Iris-setosa', 0],
     ['4.7', '3.2', '1.3', '0.2', 'Iris-setosa', 0],
     ['4.6', '3.1', '1.5', '0.2', 'Iris-setosa', 0],
     ['5.0', '3.6', '1.4', '0.2', 'Iris-setosa', 0],
     ['5.4', '3.9', '1.7', '0.4', 'Iris-setosa', 0],
     ['4.6', '3.4', '1.4', '0.3', 'Iris-setosa', 0],
     ['5.0', '3.4', '1.5', '0.2', 'Iris-setosa', 0],
     ['4.4', '2.9', '1.4', '0.2', 'Iris-setosa', 0],
     ['4.9', '3.1', '1.5', '0.1', 'Iris-setosa', 0],
     ['5.4', '3.7', '1.5', '0.2', 'Iris-setosa', 0],
     ['4.8', '3.4', '1.6', '0.2', 'Iris-setosa', 0],
     ['4.8', '3.0', '1.4', '0.1', 'Iris-setosa', 0],
     ['4.3', '3.0', '1.1', '0.1', 'Iris-setosa', 0],
     ['5.8', '4.0', '1.2', '0.2', 'Iris-setosa', 0],
     ['5.7', '4.4', '1.5', '0.4', 'Iris-setosa', 0],
     ['5.4', '3.9', '1.3', '0.4', 'Iris-setosa', 0],
     ['5.1', '3.5', '1.4', '0.3', 'Iris-setosa', 0],
     ['5.7', '3.8', '1.7', '0.3', 'Iris-setosa', 0],
     ['5.1', '3.8', '1.5', '0.3', 'Iris-setosa', 0],
     ['5.4', '3.4', '1.7', '0.2', 'Iris-setosa', 0],
     ['5.1', '3.7', '1.5', '0.4', 'Iris-setosa', 0],
     ['4.6', '3.6', '1.0', '0.2', 'Iris-setosa', 0],
     ['5.1', '3.3', '1.7', '0.5', 'Iris-setosa', 0],
     ['4.8', '3.4', '1.9', '0.2', 'Iris-setosa', 0],
     ['5.0', '3.0', '1.6', '0.2', 'Iris-setosa', 0],
     ['5.0', '3.4', '1.6', '0.4', 'Iris-setosa', 0],
     ['5.2', '3.5', '1.5', '0.2', 'Iris-setosa', 0],
     ['5.2', '3.4', '1.4', '0.2', 'Iris-setosa', 0],
     ['4.7', '3.2', '1.6', '0.2', 'Iris-setosa', 0],
     ['4.8', '3.1', '1.6', '0.2', 'Iris-setosa', 0],
     ['5.4', '3.4', '1.5', '0.4', 'Iris-setosa', 0],
     ['5.2', '4.1', '1.5', '0.1', 'Iris-setosa', 0],
     ['5.5', '4.2', '1.4', '0.2', 'Iris-setosa', 0],
     ['4.9', '3.1', '1.5', '0.1', 'Iris-setosa', 0],
     ['5.0', '3.2', '1.2', '0.2', 'Iris-setosa', 0],
     ['5.5', '3.5', '1.3', '0.2', 'Iris-setosa', 0],
     ['4.9', '3.1', '1.5', '0.1', 'Iris-setosa', 0],
     ['4.4', '3.0', '1.3', '0.2', 'Iris-setosa', 0],
     ['5.1', '3.4', '1.5', '0.2', 'Iris-setosa', 0],
     ['5.0', '3.5', '1.3', '0.3', 'Iris-setosa', 0],
     ['4.5', '2.3', '1.3', '0.3', 'Iris-setosa', 0],
     ['4.4', '3.2', '1.3', '0.2', 'Iris-setosa', 0],
     ['5.0', '3.5', '1.6', '0.6', 'Iris-setosa', 0],
     ['5.1', '3.8', '1.9', '0.4', 'Iris-setosa', 0],
     ['4.8', '3.0', '1.4', '0.3', 'Iris-setosa', 0],
     ['5.1', '3.8', '1.6', '0.2', 'Iris-setosa', 0],
     ['4.6', '3.2', '1.4', '0.2', 'Iris-setosa', 0],
     ['5.3', '3.7', '1.5', '0.2', 'Iris-setosa', 0],
     ['5.0', '3.3', '1.4', '0.2', 'Iris-setosa', 0],
     ['7.0', '3.2', '4.7', '1.4', 'Iris-versicolor', 50],
     ['6.4', '3.2', '4.5', '1.5', 'Iris-versicolor', 50],
     ['6.9', '3.1', '4.9', '1.5', 'Iris-versicolor', 50],
     ['5.5', '2.3', '4.0', '1.3', 'Iris-versicolor', 0],
     ['6.5', '2.8', '4.6', '1.5', 'Iris-versicolor', 50],
     ['5.7', '2.8', '4.5', '1.3', 'Iris-versicolor', 50],
     ['6.3', '3.3', '4.7', '1.6', 'Iris-versicolor', 50],
     ['4.9', '2.4', '3.3', '1.0', 'Iris-versicolor', 0],
     ['6.6', '2.9', '4.6', '1.3', 'Iris-versicolor', 50],
     ['5.2', '2.7', '3.9', '1.4', 'Iris-versicolor', 0],
     ['5.0', '2.0', '3.5', '1.0', 'Iris-versicolor', 0],
     ['5.9', '3.0', '4.2', '1.5', 'Iris-versicolor', 50],
     ['6.0', '2.2', '4.0', '1.0', 'Iris-versicolor', 50],
     ['6.1', '2.9', '4.7', '1.4', 'Iris-versicolor', 50],
     ['5.6', '2.9', '3.6', '1.3', 'Iris-versicolor', 0],
     ['6.7', '3.1', '4.4', '1.4', 'Iris-versicolor', 50],
     ['5.6', '3.0', '4.5', '1.5', 'Iris-versicolor', 50],
     ['5.8', '2.7', '4.1', '1.0', 'Iris-versicolor', 0],
     ['6.2', '2.2', '4.5', '1.5', 'Iris-versicolor', 50],
     ['5.6', '2.5', '3.9', '1.1', 'Iris-versicolor', 0],
     ['5.9', '3.2', '4.8', '1.8', 'Iris-versicolor', 50],
     ['6.1', '2.8', '4.0', '1.3', 'Iris-versicolor', 50],
     ['6.3', '2.5', '4.9', '1.5', 'Iris-versicolor', 50],
     ['6.1', '2.8', '4.7', '1.2', 'Iris-versicolor', 50],
     ['6.4', '2.9', '4.3', '1.3', 'Iris-versicolor', 50],
     ['6.6', '3.0', '4.4', '1.4', 'Iris-versicolor', 50],
     ['6.8', '2.8', '4.8', '1.4', 'Iris-versicolor', 50],
     ['6.7', '3.0', '5.0', '1.7', 'Iris-versicolor', 50],
     ['6.0', '2.9', '4.5', '1.5', 'Iris-versicolor', 50],
     ['5.7', '2.6', '3.5', '1.0', 'Iris-versicolor', 50],
     ['5.5', '2.4', '3.8', '1.1', 'Iris-versicolor', 0],
     ['5.5', '2.4', '3.7', '1.0', 'Iris-versicolor', 0],
     ['5.8', '2.7', '3.9', '1.2', 'Iris-versicolor', 0],
     ['6.0', '2.7', '5.1', '1.6', 'Iris-versicolor', 50],
     ['5.4', '3.0', '4.5', '1.5', 'Iris-versicolor', 50],
     ['6.0', '3.4', '4.5', '1.6', 'Iris-versicolor', 50],
     ['6.7', '3.1', '4.7', '1.5', 'Iris-versicolor', 50],
     ['6.3', '2.3', '4.4', '1.3', 'Iris-versicolor', 50],
     ['5.6', '3.0', '4.1', '1.3', 'Iris-versicolor', 50],
     ['5.5', '2.5', '4.0', '1.3', 'Iris-versicolor', 0],
     ['5.5', '2.6', '4.4', '1.2', 'Iris-versicolor', 50],
     ['6.1', '3.0', '4.6', '1.4', 'Iris-versicolor', 50],
     ['5.8', '2.6', '4.0', '1.2', 'Iris-versicolor', 50],
     ['5.0', '2.3', '3.3', '1.0', 'Iris-versicolor', 0],
     ['5.6', '2.7', '4.2', '1.3', 'Iris-versicolor', 0],
     ['5.7', '3.0', '4.2', '1.2', 'Iris-versicolor', 50],
     ['5.7', '2.9', '4.2', '1.3', 'Iris-versicolor', 50],
     ['6.2', '2.9', '4.3', '1.3', 'Iris-versicolor', 50],
     ['5.1', '2.5', '3.0', '1.1', 'Iris-versicolor', 50],
     ['5.7', '2.8', '4.1', '1.3', 'Iris-versicolor', 0],
     ['6.3', '3.3', '6.0', '2.5', 'Iris-virginica', 50],
     ['5.8', '2.7', '5.1', '1.9', 'Iris-virginica', 50],
     ['7.1', '3.0', '5.9', '2.1', 'Iris-virginica', 102],
     ['6.3', '2.9', '5.6', '1.8', 'Iris-virginica', 50],
     ['6.5', '3.0', '5.8', '2.2', 'Iris-virginica', 50],
     ['7.6', '3.0', '6.6', '2.1', 'Iris-virginica', 102],
     ['4.9', '2.5', '4.5', '1.7', 'Iris-virginica', 50],
     ['7.3', '2.9', '6.3', '1.8', 'Iris-virginica', 102],
     ['6.7', '2.5', '5.8', '1.8', 'Iris-virginica', 102],
     ['7.2', '3.6', '6.1', '2.5', 'Iris-virginica', 102],
     ['6.5', '3.2', '5.1', '2.0', 'Iris-virginica', 50],
     ['6.4', '2.7', '5.3', '1.9', 'Iris-virginica', 50],
     ['6.8', '3.0', '5.5', '2.1', 'Iris-virginica', 102],
     ['5.7', '2.5', '5.0', '2.0', 'Iris-virginica', 50],
     ['5.8', '2.8', '5.1', '2.4', 'Iris-virginica', 50],
     ['6.4', '3.2', '5.3', '2.3', 'Iris-virginica', 50],
     ['6.5', '3.0', '5.5', '1.8', 'Iris-virginica', 50],
     ['7.7', '3.8', '6.7', '2.2', 'Iris-virginica', 102],
     ['7.7', '2.6', '6.9', '2.3', 'Iris-virginica', 102],
     ['6.0', '2.2', '5.0', '1.5', 'Iris-virginica', 50],
     ['6.9', '3.2', '5.7', '2.3', 'Iris-virginica', 102],
     ['5.6', '2.8', '4.9', '2.0', 'Iris-virginica', 50],
     ['7.7', '2.8', '6.7', '2.0', 'Iris-virginica', 102],
     ['6.3', '2.7', '4.9', '1.8', 'Iris-virginica', 50],
     ['6.7', '3.3', '5.7', '2.1', 'Iris-virginica', 102],
     ['7.2', '3.2', '6.0', '1.8', 'Iris-virginica', 102],
     ['6.2', '2.8', '4.8', '1.8', 'Iris-virginica', 50],
     ['6.1', '3.0', '4.9', '1.8', 'Iris-virginica', 50],
     ['6.4', '2.8', '5.6', '2.1', 'Iris-virginica', 102],
     ['7.2', '3.0', '5.8', '1.6', 'Iris-virginica', 102],
     ['7.4', '2.8', '6.1', '1.9', 'Iris-virginica', 102],
     ['7.9', '3.8', '6.4', '2.0', 'Iris-virginica', 102],
     ['6.4', '2.8', '5.6', '2.2', 'Iris-virginica', 50],
     ['6.3', '2.8', '5.1', '1.5', 'Iris-virginica', 50],
     ['6.1', '2.6', '5.6', '1.4', 'Iris-virginica', 50],
     ['7.7', '3.0', '6.1', '2.3', 'Iris-virginica', 102],
     ['6.3', '3.4', '5.6', '2.4', 'Iris-virginica', 50],
     ['6.4', '3.1', '5.5', '1.8', 'Iris-virginica', 102],
     ['6.0', '3.0', '4.8', '1.8', 'Iris-virginica', 50],
     ['6.9', '3.1', '5.4', '2.1', 'Iris-virginica', 102],
     ['6.7', '3.1', '5.6', '2.4', 'Iris-virginica', 102],
     ['6.9', '3.1', '5.1', '2.3', 'Iris-virginica', 50],
     ['5.8', '2.7', '5.1', '1.9', 'Iris-virginica', 50],
     ['6.8', '3.2', '5.9', '2.3', 'Iris-virginica', 102],
     ['6.7', '3.3', '5.7', '2.5', 'Iris-virginica', 102],
     ['6.7', '3.0', '5.2', '2.3', 'Iris-virginica', 50],
     ['6.3', '2.5', '5.0', '1.9', 'Iris-virginica', 50],
     ['6.5', '3.0', '5.2', '2.0', 'Iris-virginica', 50],
     ['6.2', '3.4', '5.4', '2.3', 'Iris-virginica', 50],
     ['5.9', '3.0', '5.1', '1.8', 'Iris-virginica', 50]]
    


```python
df = pd.DataFrame(data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "class", "cluster"])
df.to_csv('../output/output_average.csv', index=False) 
```
