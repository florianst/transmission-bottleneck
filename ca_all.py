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
    af = np.sum(A)
    bf = np.sum(B)
    cf = np.sum(C)
    for i in range(1,size-1): # iterate through each cell in a grid
        for j in range(1,size-1):
            if A[i,j] == 1 or B[i,j] == 1 or C[i,j] == 1:
                r = rnd.randint(0, 3)
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
                    if af == 0 and bf == 0 and cf == 0:
                        strains_transmitted = rnd.sample(range(1, 4), b_new) # 1=A, 2=B, 3=C
                    else:
                        strains_transmitted = []
                        strains = rnd.uniform(0,1)
                        if strains <= af:
                            strains_transmitted.append(1)
                        if strains > af and strains <= bf:
                            strains_transmitted.append(2)
                        if strains > bf and strains <= cf:
                            strains_transmitted.append(3)                            

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
            if AD[i,j] > int(rnd.gauss(10,5)):
                AD[i,j] = 0
                A[i,j] = 0
            if BD[i,j] > int(rnd.gauss(15,5)):
                BD[i,j] = 0
                B[i,j] = 0
            if CD[i,j] > int(rnd.gauss(20,5)):
                CD[i,j] = 0
                C[i,j] = 0
                
    af = np.sum(A)
    bf = np.sum(B)
    cf = np.sum(C)
    return (A,B,C,AD,BD,CD,af,bf,cf)
                
### Initialize variables

size = 100 # grid dimensions
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
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
for a in range(t):
    A,B,C,AD,BD,CD,af,bf,cf = update_grids(A,B,C,AD,BD,CD,size,b)
    a1=ax1.imshow(A,interpolation='nearest',cmap='Blues')
    a2=ax2.imshow(B,interpolation='nearest',cmap='Reds')
    a3 =ax3.imshow(C,interpolation='nearest',cmap='Greens')
    images.append([a1,a2,a3])
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



