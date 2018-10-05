import numpy as np
import matplotlib.pyplot as plt


# Conditions
numOfClusters = 6
dim = 2
numOfPoints = 500
coordRange = 500
coord = np.random.random((numOfPoints, dim))
avg = np.zeros(dim)
label = []  #n번째 원소 = n번째 점의 label
color_list = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.']
color_list_cen = ['b*', 'g*', 'r*', 'c*', 'm*', 'y*']
centroid = np.zeros((numOfClusters))
listOfDist = np.zeros((numOfClusters,numOfPoints))
E = [] # 매 루프


# Functions
def generatePointCluster(numOfClusters, numOfPoints, coordRange):
    for i in range(dim):
        avg = coord.sum(axis=0)/numOfPoints
        coord[:,i] = (coord[:,i] - avg[i]) * coordRange

def initializeLabel(numOfClusters):
    return np.random.randint(numOfClusters,size=numOfPoints)

def computeDistance():
    # second-order
    for c in range(numOfClusters):
        for i in range(numOfPoints):
            listOfDist[c][i] = sum((centroid[c] - coord[i])**2)

def assignLabel():
    for i in range(numOfPoints):
        label[i] = np.argmin(listOfDist[:,i])

def computeCentroid():
    num = np.zeros((numOfClusters))
    c = np.zeros((numOfClusters, dim))

    for i in range(numOfPoints):
        num[label[i]] += 1
        c[label[i]] += coord[i]

    for i in range(numOfClusters):
        c[i] /= num[i]
    return c

def computeEnergy(): #must decrease
    energy = 0
    for i in range(numOfPoints):
        energy += sum((centroid[label[i]] - coord[i])**2)
    energy /= numOfPoints
    return energy

# Test Code
generatePointCluster(numOfClusters=numOfClusters, numOfPoints=numOfPoints, coordRange=coordRange)
plt.figure(1)
plt.plot(coord[:,0],coord[:,1],"k.")
plt.xlim(-coordRange/2,coordRange/2)
plt.ylim(-coordRange/2,coordRange/2)
plt.grid(True)
plt.show()

label = initializeLabel(numOfClusters=numOfClusters)

plt.figure(2)
for i in range(numOfPoints):
    plt.plot(coord[i][0],coord[i][1],color_list[label[i]])
plt.xlim(-coordRange/2,coordRange/2)
plt.ylim(-coordRange/2,coordRange/2)
plt.grid(True)
plt.show()


iter_num = 0
centroid = computeCentroid()
E.append(computeEnergy())
iter_num += 1

plt.figure(3)
for c in range(numOfClusters):
    plt.plot(centroid[c][0], centroid[c][1], color_list_cen[c])
plt.xlim(-coordRange/2,coordRange/2)
plt.ylim(-coordRange/2,coordRange/2)
plt.grid(True)
plt.show()

while True:
    computeDistance()
    assignLabel()
    centroid = computeCentroid()
    E.append(computeEnergy())
    if E[-1] == E[-2]:
        break
    iter_num += 1


plt.figure(4)
for i in range(numOfPoints):
    plt.plot(coord[i][0], coord[i][1], color_list[label[i]])
for c in range(numOfClusters):
    plt.plot(centroid[c][0], centroid[c][1], color_list_cen[c])
plt.xlim(-coordRange/2, coordRange/2)
plt.ylim(-coordRange/2, coordRange/2)
plt.grid(True)
plt.show()



plt.figure(5)
x_range = np.arange(iter_num+1)
plt.plot(x_range,E,"g")
plt.grid(True)
plt.show()