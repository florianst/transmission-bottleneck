import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import random as rnd
import imageio


def update_grids(A,B,C,AD,BD,CD,size,b):
    # a function determining the next state of every cell
    A_next = A.copy() # create next state grids for A,B,C
    B_next = B.copy()    
    C_next = C.copy()
    af = 0 # calculate present frequencies of strains
    bf = 0
    cf = 0
    for i in range(1,size-1): # iterate through each cell in a grid
        for j in range(1,size-1):
            if A[i,j] == 1 or B[i,j] == 1 or C[i,j] == 1:
                r = rnd.randint(1, 8)
                neighbours_to_infect = rnd.sample(range(0, 8), r)
                for each in neighbours_to_infect:
                    if b == 1:
                        b_new = int(rnd.gauss(b,1))
                        if b_new < 1:
                            b_new = 1
                        if b_new > 3:
                            b_new = 2
                    elif b == 2:
                        b_new = int(rnd.gauss(b,1))
                        if b_new < 1:
                            b_new = 1
                        if b_new > 3:
                            b_new = 3
                    elif b == 3:
                        b_new = int(rnd.gauss(b,1))
                        if b_new < 1:
                            b_new = 1
                        if b_new > 3:
                            b_new = 3
                    strains_transmitted = rnd.sample(range(1, 4), b_new) # 1=A, 2=B, 3=C

                    # find which specific neighbour and what to transmit
                    if each == 0:
                        if 1 in strains_transmitted:
                            A_next[i-1,j+1] = 1
                        elif 2 in strains_transmitted:
                            B_next[i-1,j-1] = 1
                        elif 3 in strains_transmitted:
                            C_next[i-1,j-1] = 1
                   
                    if each == 1:
                        if 1 in strains_transmitted:
                            A_next[i-1,j] = 1
                        elif 2 in strains_transmitted:
                            B_next[i-1,j] = 1
                        elif 3 in strains_transmitted:
                            C_next[i-1,j] = 1

                            
                    if each == 2:
                        if 1 in strains_transmitted:
                            A_next[i-1,j+1] = 1
                        elif 2 in strains_transmitted:
                            B_next[i-1,j+1] = 1
                        elif 3 in strains_transmitted:
                            C_next[i-1,j+1] = 1


                    if each == 3:
                        if 1 in strains_transmitted:
                            A_next[i,j-1] = 1
                        elif 2 in strains_transmitted:
                            B_next[i,j-1] = 1
                        elif 3 in strains_transmitted:
                            C_next[i,j-1] = 1


                    if each == 4:
                        if 1 in strains_transmitted:
                            A_next[i,j+1] = 1
                        elif 2 in strains_transmitted:
                            B_next[i,j+1] = 1
                        elif 3 in strains_transmitted:
                            C_next[i,j+1] = 1

                    if each == 5:
                        if 1 in strains_transmitted:
                            A_next[i+1,j+1] = 1
                        elif 2 in strains_transmitted:
                            B_next[i+1,j+1] = 1
                        elif 3 in strains_transmitted:
                            C_next[i+1,j+1] = 1

                            
                    if each == 6:
                        if 1 in strains_transmitted:
                            A_next[i+1,j-1] = 1
                        elif 2 in strains_transmitted:
                            B_next[i+1,j-1] = 1
                        elif 3 in strains_transmitted:
                            C_next[i+1,j-1] = 1

                            
                    if each == 7:
                        if 1 in strains_transmitted:
                            A_next[i+1,j] = 1
                        elif 2 in strains_transmitted:
                            B_next[i+1,j] = 1
                        elif 3 in strains_transmitted:
                            C_next[i+1,j] = 1
    A = A_next
    B = B_next
    C = C_next
    for i in range(1,size-1): # update age of strains
        for j in range(1,size-1):
            if A[i,j] == 1:
                AD[i,j] += 1
            if B[i,j] == 1:
                BD[i,j] += 1
            if C[i,j] == 1:
                CD[i,j] += 1
    for i in range(1,size-1): # update age of strains
        for j in range(1,size-1):
            if AD[i,j] > 10:
                AD[i,j] = 0
                A[i,j] = 0
            if BD[i,j] > 15:
                BD[i,j] = 0
                B[i,j] = 0
            if CD[i,j] > 20:
                CD[i,j] = 0
                C[i,j] = 0
                
    af = np.sum(A)
    bf = np.sum(B)
    cf = np.sum(C)
    return (A,B,C,AD,BD,CD,af,bf,cf)
                
### Initialize variables

size = 30 # grid dimensions
half_size = int(size/2)
b = 2 # bottleneck size, range (1-3)
t = 50 # number of iterations
A = np.zeros([size,size])
B = np.zeros([size,size])
C = np.zeros([size,size])
AD = np.zeros([size,size])
BD = np.zeros([size,size])
CD = np.zeros([size,size])

A[half_size,half_size] = 1
B[half_size,half_size] = 1
C[half_size,half_size] = 1

a_freq = []
b_freq = []
c_freq = []

# create animation and plots
images = []
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
for a in range(t):
    A,B,C,AD,BD,CD,af,bf,cf = update_grids(A,B,C,AD,BD,CD,size,b)
    ax1 = plt.imshow(A,interpolation='nearest',cmap='binary')
    ax2 = plt.imshow(B,interpolation='nearest',cmap='binary')
    ax3 = plt.imshow(C,interpolation='nearest',cmap='binary')
    images.append([ax1,ax2,ax3])
    a_freq.append(af/(np.sum(A)+np.sum(B)+np.sum(C)))
    b_freq.append(bf/(np.sum(A)+np.sum(B)+np.sum(C)))
    c_freq.append(cf/(np.sum(A)+np.sum(B)+np.sum(C)))

ani = anm.ArtistAnimation(fig,images,interval=250,repeat=False,blit=False)
plt.show()



t_array = np.linspace(1,t,t)
plt.figure()
plt.plot(t_array, a_freq, 'b', label = 'frequency of A')
plt.plot(t_array, b_freq, 'r', label = 'frequency of B')
plt.plot(t_array, c_freq, 'g', label = 'frequency of C')
plt.legend()
plt.show()



