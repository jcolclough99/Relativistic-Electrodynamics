# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:21:24 2020

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
q = 1.602 * 10**-19
#m = 9.109*10**-31
#m = 1.884*10**-28
m = 2.496 * 10**-28
eps = 8.85*1.0006*10**-12
n = 1.00029
c = 2.998*10**8
En = gamma(c-1)*(9.109*10**-31)*c**2
cn = c/n
B = np.array((0,0,0))
r0 = np.zeros(3)
v0 = np.array((0,0,np.sqrt(((c**2)*(1-((m*c**2)/En)**2)))))
dt = 10**-15
K= 10**6
mu = 1/(eps*cn**2)
print(0.150*np.arccos(cn/v0[2]))
#%%
def acc(v):
    F = q*np.cross(v,B)
    a = F/(gamma(np.linalg.norm(v))*m)
    return a

def gamma(v):
    g = 1/ np.sqrt(1- (v**2 / (c)**2))
    return g

def power(v,a):
    P = (mu * q**2 *gamma(np.linalg.norm(v))**6)/(6*np.pi*c)
    P = P*(np.linalg.norm(a)**2 -(((np.linalg.norm(np.cross(v,a)))**2)/c**2))
    return P
 
def radius(v):
    r = (np.linalg.norm(v))**2/np.linalg.norm(acc(v))
    return r

def evolve(r,v):
    R = np.zeros((K+1,3))
    V = np.zeros((K+1,3))
    A = np.zeros((K+1,3))
    rad = np.zeros((K+1))
    magv = np.zeros((K+1))
    R[0,:] = r
    V[0,:] = v
    A[0,:] = acc(v)
    rad[0] = radius(v)
    magv[0] = np.linalg.norm(v)
    for i in range(1,K+1):
        V[i,:] = V[i-1,:] + A[i-1,:]*dt
        vn = V[i,:]/np.linalg.norm(V[i,:])
        magv[i] = c*np.sqrt(1-((m*c**2)/(gamma(np.linalg.norm(V[i-1,:]))*m*c**2 - power(V[i-1,:],A[i-1,:])*dt))**2)
        V[i,:] = vn * magv[i]
        R[i,:] = R[i-1,:] + V[i-1,:]*dt
        A[i,:] = acc(V[i,:]) + ((magv[i]-magv[i-1])/dt)*vn
        rad[i] = radius(V[i,:])
    
    return R,V,A,rad
        
def CHevolve(r,v):
    R = np.zeros((K+1,3))
    V = np.zeros((K+1,3))
    A = np.zeros((K+1,3))
    magv = np.zeros((K+1))
    R[0,:] = r
    V[0,:] = v
    #A[0,:] = acc(v)
    magv[0] = np.linalg.norm(v)
    for i in range(1,K+1):
        V[i,:] = V[i-1,:]#+ A[i-1,:]*dt
        #vn = V[i,:]/np.linalg.norm(V[i,:])
        #magv[i] = c*np.sqrt(1-((m*c**2)/(gamma(np.linalg.norm(V[i-1,:]))*m*c**2 - power(V[i-1,:],A[i-1,:])*dt))**2)
        #V[i,:] = vn * magv[i]
        R[i,:] = R[i-1,:] + V[i-1,:]*dt
        #A[i,:] = acc(V[i,:])
    return R,V,A
    
r,v,a = CHevolve(r0,v0)
 
#%%
z = r[:,2]
x = r[:,0]
fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
ax.plot(z[:],x[:])
plt.xlabel('z position / m')
plt.ylabel('x position / m')
plt.title('Relativistic trajectory of an electron in a constant B-field')
#%%
t = [] 

for i in range(K+1):
    t.append(i*dt)
plt.plot(t,rad)
plt.xlabel('Time')
plt.ylabel('Radius / m')
#%%
print(np.argmax(x[5:10]))
#%%
O = np.array((rad[0],0,0))
arg = [0]
magr = []
R = [np.array((0,0,0))]
for i in range(K):
    magr.append(np.linalg.norm(r[i,:]))
for i in range(10):
    o = np.argmin(magr[arg[i]+1:(i+1)*10000])+arg[i]+1
    arg.append(o)
    R.append(r[o])

print(R)
print(arg)
#%%           
O = np.array((rad[0],0,0))
rad0 = []
for i in range(len(R)):
    o = R[i] 
    rad0.append(o[0])

k = [] 
for i in range(len(rad0)):
    k.append(i)

plt.plot(k,rad0)
#%%
def maxevolve(r,v):
    R = np.zeros((K+1,3))
    V = np.zeros((K+1,3))
    A = np.zeros((K+1,3))
    rad = np.zeros((K+1))
    R[0,:] = r
    V[0,:] = v
    A[0,:] = acc(v)
    rad[0] = radius(v) 
    P = power(V[0,:],A[0,:])
    for i in range(1,K+1):
        V[i,:] = V[i-1,:] + A[i-1,:]*dt
        V[i,:] = V[i,:]/np.linalg.norm(V[i,:])
        v = c*np.sqrt(1-((m*c**2)/(gamma(np.linalg.norm(V[i-1,:]))*m*c**2 - P*dt))**2)
        V[i,:] = V[i,:] * v
        R[i,:] = R[i-1,:] + V[i-1,:]*dt
        A[i,:] = acc(V[i,:])
        rad[i] = radius(V[i,:])
    
    return R,V,A,rad
        
    
maxr,maxv,maxa,maxrad = maxevolve(r0,v0)

#%%
O = np.array((maxrad[0],0,0))
maxR = [maxr[0,:]]
for i in range(K):
    if len(maxR) % 2 == 0:
        if np.linalg.norm(maxr[i+1,:]) > np.linalg.norm(maxr[i,:]):
            o = maxr[i,:]
            maxR.append(o)
    else:
        if np.linalg.norm(maxr[i+1,:]) < np.linalg.norm(maxr[i,:]):
            o = maxr[i,:]
            maxR.append(o)

#%%
maxR0 = []            
for i in range(len(maxR)):
    if i % 2 == 0:
        maxR0.append(maxR[i])
     
maxrad0 = np.zeros((len(maxR0)))            
O = np.array((maxrad[0],0,0))
for i in range(len(maxR0)):
    maxrad0[i] = np.linalg.norm(maxR0[i] - O)

maxk = [] 

for i in range(len(maxrad0)):
    maxk.append(i)

#%%
vf = v[K,:]
af = a[K,:]
def minevolve(r,v):
    R = np.zeros((K+1,3))
    V = np.zeros((K+1,3))
    A = np.zeros((K+1,3))
    rad = np.zeros((K+1))
    P = power(vf,af)
    R[0,:] = r
    V[0,:] = v
    A[0,:] = acc(v)
    rad[0] = radius(v) 
    for i in range(1,K+1):
        V[i,:] = V[i-1,:] + A[i-1,:]*dt
        V[i,:] = V[i,:]/np.linalg.norm(V[i,:])
        v = c*np.sqrt(1-((m*c**2)/(gamma(np.linalg.norm(V[i-1,:]))*m*c**2 - P*dt))**2)
        V[i,:] = V[i,:] * v
        R[i,:] = R[i-1,:] + V[i-1,:]*dt
        A[i,:] = acc(V[i,:])
        rad[i] = radius(V[i,:])
    
    return R,V,A,rad
        
    
minr,minv,mina,minrad = minevolve(r0,v0) 
#%%
O = np.array((minrad[0],0,0))
minR = [minr[0,:]]
for i in range(K):
    if len(minR) % 2 == 0:
        if np.linalg.norm(minr[i+1,:]) > np.linalg.norm(minr[i,:]):
            o = minr[i,:]
            minR.append(o)
    else:
        if np.linalg.norm(minr[i+1,:]) < np.linalg.norm(minr[i,:]):
            o = minr[i,:]
            minR.append(o)

#%%
minR0 = []            
for i in range(len(minR)):
    if i % 2 == 0:
        minR0.append(minR[i])
     
minrad0 = np.zeros((len(minR0)))            
O = np.array((minrad[0],0,0))
for i in range(len(minR0)):
    minrad0[i] = np.linalg.norm(minR0[i] - O)

mink = [] 

for i in range(len(minrad0)):
    mink.append(i)
plt.plot(mink,minrad0,label = 'min')
plt.plot(maxk,maxrad0,label = 'max')
plt.plot(k,rad0)
plt.legend()
#%%
def ret(k0,k,r_o,n):
    f = np.zeros((k+1-k0))
    t = np.zeros((k+1-k0))
    ret_t = []
    ret_r = []
    ret_v = []
    ret_k = []
    ret_a = []
    for i in range(k0,k+1):
        if len(ret_k) == n:
            break
        j = i - k0
        t[j] = i*dt 
        f[j] = cn*(k*dt- t[j]) - np.linalg.norm(r_o -r[i])
        if (f[j] < 0 and f[j-1] > 0) or (f[j] > 0 and f[j-1] < 0):
            m = (f[j]-f[j-1])/(t[j]-t[j-1])
            ct = f[j] - m*t[j]
            o = -ct/m
            ret_t.append(o)
            ret_k.append(i-1)
            mx,my,mz = (r[i,0] - r[i-1,0])/(t[j]-t[j-1]), (r[i,1] - r[i-1,1])/(t[j]-t[j-1]),(r[i,2] - r[i-1,2])/(t[j]-t[j-1])
            cx,cy,cz = r[i,0] - mx*t[j], r[i,1] - my*t[j],r[i,2] - mz*t[j]
            x,y,z = mx*o + cx , my*o + cy, mz*o + cz
            l = np.array((x,y,z))
            ret_r.append(l)
            mx,my,mz = (v[i,0] - v[i-1,0])/(t[j]-t[j-1]), (v[i,1] - v[i-1,1])/(t[j]-t[j-1]),(v[i,2] - v[i-1,2])/(t[j]-t[j-1])
            cx,cy,cz = v[i,0] - mx*t[j], v[i,1] - my*t[j], v[i,2] - mz*t[j]
            x,y,z = mx*o + cx , my*o + cy, mz*o + cz
            l = np.array((x,y,z))
            ret_v.append(l)
            mx,my,mz = (a[i,0] - a[i-1,0])/(t[j]-t[j-1]), (a[i,1] - a[i-1,1])/(t[j]-t[j-1]),(a[i,2] - a[i-1,2])/(t[j]-t[j-1])
            cx,cy,cz = a[i,0] - mx*t[j], a[i,1] - my*t[j], a[i,2] - mz*t[j]
            x,y,z = mx*o + cx , my*o + cy, mz*o + cz
            l = np.array((x,y,z))
            ret_a.append(l)
            
    #if len(ret_k) == 0:
        #return 'No retarded times found'        
    return ret_r,ret_v,ret_a,ret_k
#%%
t = (c*(8*10**5 * dt) + 25)/cn
print(t/dt)
#%%
RR,VV,KK = ret(0,t*dt,np.array((0,0,0.025)),2)
print(KK)
#%%
def ret_pot(ret_r,ret_v,r_o):
    V = 0
    for i in range(len(ret_r)):
        R = r_o - ret_r[i]
        V = V + q/(4*np.pi*eps*abs(np.linalg.norm(R)-np.dot(ret_v[i]/cn,R)))
    return V

def ret_A(ret_r,ret_v,r_o):
    A = np.zeros(3)
    for i in range(len(ret_r)):
        R = r_o - ret_r[i]
        A = A + (q*mu*ret_v[i])/(4*np.pi*abs(np.linalg.norm(R)-np.dot(ret_v[i]/cn,R)))
    return A
#%%
def ret2(k0,T,K,r_o,n):
    t = []
    f = [] 
    ret_k = []
    ret_r = []
    ret_v = []
    ret_a = []
    if k0 < 0:
        k0 = 0
    for i in range(k0,K+1):
        if n != 0:
            if len(ret_k) == n:
                break
        j = i - k0
        t.append(i*dt)
        f.append(cn*(T-t[j]) - np.linalg.norm(r_o-r[i]))
        if abs(f[j]) < 10**-16:
            ret_k.append(i)
            ret_r.append(r[i])
            ret_v.append(v[i])
            ret_a.append(a[i])
        elif ((f[j] < 0 and f[j-1] > 0) or (f[j] > 0 and f[j-1] < 0)) and abs(f[j-1])>10**-16:
            m = (f[j]-f[j-1])/(t[j]-t[j-1])
            ct = f[j] - m*t[j]
            o = -ct/m
            mx,my,mz = (r[i,0] - r[i-1,0])/(t[j]-t[j-1]),(r[i,1] - r[i-1,1])/(t[j]-t[j-1]),(r[i,2] - r[i-1,2])/(t[j]-t[j-1])
            cx,cy,cz = r[i,0] - mx*t[j], r[i,1] - my*t[j],r[i,2] - mz*t[j]
            x,y,z = mx*o + cx , my*o + cy, mz*o + cz
            l = np.array((x,y,z))
            ret_r.append(l)
            mx,my,mz = (v[i,0] - v[i-1,0])/(t[j]-t[j-1]),(v[i,1] - v[i-1,1])/(t[j]-t[j-1]),(v[i,2] - v[i-1,2])/(t[j]-t[j-1])
            cx,cy,cz = v[i,0] - mx*t[j],v[i,1] - my*t[j], v[i,2] - mz*t[j]
            x,y,z = mx*o + cx , my*o + cy, mz*o + cz
            l = np.array((x,y,z))
            ret_v.append(l)
            ret_k.append(i-1)
            mx,my,mz = (a[i,0] - a[i-1,0])/(t[j]-t[j-1]),(a[i,1] - a[i-1,1])/(t[j]-t[j-1]),(a[i,2] - a[i-1,2])/(t[j]-t[j-1])
            cx,cy,cz = a[i,0] - mx*t[j], a[i,1] - my*t[j], a[i,2] - mz*t[j]
            x,y,z = mx*o + cx , my*o + cy, mz*o + cz
            l = np.array((x,y,z))
            ret_a.append(l)
    return ret_r,ret_v,ret_a,ret_k
#%%
pos = r[8*10**4]+np.array((0.04,0,0)) 
retr,retv,reta,retk = ret2(0,dt*9*10**4,K,pos,2)
Rt = pos - r[9*10**4]
theta = np.pi - np.arctan(Rt[0]/Rt[2])
print(theta)
print(ret_pot(retr,retv,pos))
print(q/(2*np.pi*eps*np.linalg.norm(Rt)*np.sqrt(1-((np.linalg.norm(v)/cn)**2)*np.sin(theta)**2)))
#%%
def sphere(R,dthe,dphi,w):
    r2 = []
    r = []
    for i in range(1,np.int(1/dthe)):
        for j in range(np.int(2/dphi)):
            o = np.array((R,i*dthe*np.pi,j*dphi*np.pi))
            r2.append(o)
            o = np.array((R*np.sin(i*dthe*np.pi)*np.cos(j*dphi*np.pi),
                          R*np.sin(i*dthe*np.pi)*np.sin(j*dphi*np.pi),
                          R*np.cos(i*dthe*np.pi)))+w
            r.append(o)
    return r,r2

R,dthe,dphi = 100,1/100,1/100
kret = 1000
w = r[kret]
points,points2 = sphere(R,dthe,dphi,w)
pos,pos2 = np.array(points),np.array(points2)

#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs=[]
ys=[]
zs=[]
for i in range(np.int(len(pos)/6)):
    o = pos[i*6]
    xs.append(o[0])
    ys.append(o[1])
    zs.append(o[2])
 
ax.scatter(xs,ys,zs,s=2)
ax.set_xticks([-10,-5,0,5,10])
ax.set_yticks([-10,-5,0,5,10])
ax.set_zticks([-10,-5,0,5,10])
#%%
chi = np.arccos(cn/v0[2])
D = r[10**6][2]
h = D*np.tan(chi)
delta = 0.00002
lam = 80000
print(h/delta)
print(2*np.pi*h*lam)
#%%
pos=[]
for i in range(1,np.int(h/delta)):
    l = []
    n = np.int(2*np.pi*i*delta*lam)
    for j in range(n):
        o = np.array((i*delta*np.cos(2*np.pi*j/n),i*delta*np.sin(2*np.pi*j/n),D))
        l.append(o)
    if len(l) >0:
        pos.append(l)

#%%
pos = []
delta = 2*10**-5
gam = np.arcsin(cn/v0[2])
D = r[10**6][2]
for i in range(1,10):
    l = []
    h = (np.tan(gam)*(i*1*10**-7))-(10**-5)
    for j in range((2*np.int(h/delta))-1):
        o = np.array(((j*delta)-((np.int(h/delta)-1)*delta),0,D-(i*10**-7)))
        l.append(o)
    if len(l)>0:
        pos.append(l)

#%%
pos = []
for i in range(3*10**5,(7*10**5),5000):
    l = []
    h2 = np.tan(chi)*(D-r[i][2])
    n = np.int(2*np.pi*h2*lam)
    for j in range(n):
       o = np.array((h2*np.cos(2*np.pi*j/n),h2*np.sin(2*np.pi*j/n),D))
       l.append(o) 
    pos.append(l)


#%%
count = 0
for i in range(len(pos)):
    count = count + len(pos[i])

POS = np.zeros((count,3))
o = 0
for i in range(len(pos)):
    POS[o:o+len(pos[i]),:] = pos[i]
    o = o + len(pos[i])
print(len(POS))
#%%
plt.scatter(POS[:,2]*10**5,POS[:,0]*10**5)
#%%
print(pos[101][0],pos[101][-1])
#%%
t = np.sqrt((D-r[5*10**5][2])**2 + ((D-r[5*10**5][2])*np.tan(chi))**2)/cn
t = t + (dt*5*10**5)
print((D-r[5*10**5][2]))
#%%
t=dt*10**6
#%%
d = 10**-6
gam = np.arcsin(cn/v0[2])
print(d*np.tan(gam))
#%%
t=dt*10**6
retr0,retv0,reta,k = ret2(0,t,K,POS[4]-np.array((0,0,dz)),2)
retr1,retv1,reta,k = ret2(0,t,K,POS[4]+np.array((0,0,dz)),2)
retr2,retv2,reta,k = ret2(0,t+Dt,K,POS[4]-np.array((0,0,0)),2)
retr3,retv3,reta,k = ret2(0,t-Dt,K,POS[4]-np.array((0,0,0)),2)
#%%
Vmdz = ret_pot(retr0,retv0,POS[4]-np.array((0,0,dz)))
Vpdz = ret_pot(retr1,retv1,POS[4]+np.array((0,0,dz)))
Apdt = ret_A(retr2,retv2,POS[4])
Amdt = ret_A(retr3,retv3,POS[4])
print(-((Vpdz-Vmdz)/(2*dz))-((Apdt-Amdt)/(2*Dt))[2])
#%%
print((D-r[5*10**5][2])*np.tan(chi))
#%%
R = np.array((0.00,0,-0.00005))
tret = t-((np.dot(-v0,R) + np.sqrt((np.dot(v0,R)**2 - (np.linalg.norm(v0)**2 - cn**2)*np.linalg.norm(R)**2)))/(np.linalg.norm(v0)**2 - cn**2))
tret1 = t-((np.dot(-v0,R) - np.sqrt((np.dot(v0,R)**2 - (np.linalg.norm(v0)**2 - cn**2)*np.linalg.norm(R)**2)))/(np.linalg.norm(v0)**2 - cn**2))
print(tret/dt)
print(tret1/dt)

#%%
n = len(pos)
pos0 = []
k = 0
for i in range(n):
        pos0.append(np.array(pos[i]))
#%%
dx = 10**-8
dy = 10**-8
dz = 10**-8
Dt = dt/5
def conefields(t,pos,tol):
    E=[]
    B=[]
    pdx = pos + np.array((dx,0,0))
    pdy = pos + np.array((0,dy,0))
    pdz = pos + np.array((0,0,dz))
    mdx = pos - np.array((dx,0,0))
    mdy = pos - np.array((0,dy,0))
    mdz = pos - np.array((0,0,dz))
    for i in range(len(pos)):
        print(i)
        ret_r,ret_v,ret_a,k = ret2(900000,t,K,mdx[i],2)
        Vmdx = ret_pot(ret_r,ret_v,mdx[i])
        Amdx = ret_A(ret_r,ret_v,mdx[i])
        ret_r,ret_v = [],[]
        for j in range(2):
            r,v,a,k2 = ret2(k[j]-tol,t,k[j]+tol,mdy[i],1)
            ret_r.append(r[0])
            ret_v.append(v[0])
            
        Vmdy = ret_pot(ret_r,ret_v,mdy[i])
        Amdy = ret_A(ret_r,ret_v,mdy[i])
        
        ret_r,ret_v = [],[]
        for j in range(2):
            r,v,a,k2 = ret2(k[j]-tol,t,k[j]+tol,mdz[i],1)
            ret_r.append(r[0])
            ret_v.append(v[0])
            print(k2)
            
        Vmdz = ret_pot(ret_r,ret_v,mdz[i])
        Amdz = ret_A(ret_r,ret_v,mdz[i])
        
        ret_r,ret_v = [],[]
        for j in range(2):
            r,v,a,k2 = ret2(k[j]-tol,t,k[j]+tol,pdx[i],1)
            ret_r.append(r[0])
            ret_v.append(v[0])
            
        Vpdx = ret_pot(ret_r,ret_v,pdx[i])
        Apdx = ret_A(ret_r,ret_v,pdx[i])
        
        ret_r,ret_v = [],[]
        for j in range(2):
            r,v,a,k2 = ret2(k[j]-tol,t,k[j]+tol,pdy[i],1)
            ret_r.append(r[0])
            ret_v.append(v[0])

        Vpdy = ret_pot(ret_r,ret_v,pdy[i])
        Apdy = ret_A(ret_r,ret_v,pdy[i])
        
        ret_r,ret_v = [],[]
        for j in range(2):
            r,v,a,k2 = ret2(k[j]-tol,t,k[j]+tol,pdz[i],1)
            ret_r.append(r[0])
            ret_v.append(v[0])
            print(k2)
        Vpdz = ret_pot(ret_r,ret_v,pdz[i])
        Apdz = ret_A(ret_r,ret_v,pdz[i])
        
        ret_r,ret_v = [],[]
        for j in range(2):
            r,v,a,k2 = ret2(k[j]-(tol),t+Dt,k[j]+(tol),pos[i],1)
            ret_r.append(r[0])
            ret_v.append(v[0])
            
        Apdt = ret_A(ret_r,ret_v,pos[i])
        
        ret_r,ret_v = [],[]
        for j in range(2):
            r,v,a,k2 = ret2(k[j]-(tol),t-Dt,k[j]+(tol),pos[i],1)
            ret_r.append(r[0])
            ret_v.append(v[0])
            
        Amdt = ret_A(ret_r,ret_v,pos[i])
        o = -np.array(((Vpdx-Vmdx)/(2*dx),(Vpdy-Vmdy)/(2*dy),(Vpdz-Vmdz)/(2*dz)))
        o = o - ((Apdt - Amdt)/(2*Dt))
        Ax = (Apdx - Amdx) / (2*dx)
        Ay = (Apdy - Amdy) / (2*dy)
        Az = (Apdz - Amdz) / (2*dz)
        l = np.array((Ay[2] - Az[1],Az[0] - Ax[2],Ax[1] - Ay[0]))
        E.append(o)
        B.append(l)
    E,B = np.array(E),np.array(B)
    return E,B
        
E,B = conefields(dt*10**6,POS[2:],9000)   
#%%
Rq = POS[4][0]/((cn/v0[2]))
dq = POS[4][0]/np.tan(np.arcsin(cn/v0[2]))
t = (r[10**6][2]-POS[4][2]-dq)/v0[2]
Eqt = (q*(((v0[2]/cn)**2)-1)**(1/4))/(2**(3/2) *np.pi *eps*cn**(3/2) *Rq**(1/2))
Eqt = Eqt*(-1/2)*t**(-3/2)
Rqn = np.array((POS[4][0],0,-dq))
Rqn = Rqn/np.linalg.norm(Rqn)
print(Eqt*Rqn)
print(E[2])
#%%
dx = 10**-8
dy = 10**-8
dz = 10**-9
Dt = dt/10
def planefields(t,pos,tol,tol2):
    n = len(pos)
    E = []
    B = []
    k0 = ret2(0,t,K,pos[0][0],0)[3]
    for j in range(n):
        print(j)
        pos0 = np.array(pos[j])
        pdx = pos0 + np.array((dx,0,0))
        pdy = pos0 + np.array((0,dy,0))
        pdz = pos0 + np.array((0,0,dz))
        mdx = pos0 - np.array((dx,0,0))
        mdy = pos0 - np.array((0,dy,0))
        mdz = pos0 - np.array((0,0,dz))
        k1 = []
        k2 = []
        k3 = []
        k4 = []
        k5 = []
        k6 = []
        k7 = []
        for k in range(len(k0)):
            k0[k] = ret2(k0[k]-(tol2+(j*3000)),t,K,mdx[0],1)[3]#[0]          
            if len(k0[k]) == 1:
                k0[k] = k0[k][0]
            else:
                k0[k] = 0
                break
            print(k0)
            if k0[1] == k0[0]:
                k0[1] = ret2(k0[0]+1,t,K,mdx[0],1)[3][0]
            print(k0)
            k1.append(ret2(k0[k]-50,t,K,mdy[0],1)[3][0])
            k2.append(ret2(k0[k]-500,t,K,mdz[0],1)[3][0])
            k3.append(ret2(k0[k]-50,t,K,pdx[0],1)[3][0])
            k4.append(ret2(k0[k]-50,t,K,pdy[0],1)[3][0])
            k5.append(ret2(k0[k]-500,t,K,pdz[0],1)[3][0])
            k6.append(ret2(k0[k]-50,t+Dt,K,pos0[0],1)[3][0])
            k7.append(ret2(k0[k]-20000,t-Dt,K,pos0[0],1)[3][0])      
        for i in range(len(pos0)):
            ret_r,ret_v = [],[]
            for k in range(len(k0)):
                if k0[k]==0:
                    break
                r,v,a,k0[k] = ret2(k0[k]-tol,t,k0[k]+tol,mdx[i],1)
                k0[k] = k0[k][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
            Vmdx = ret_pot(ret_r,ret_v,mdx[i])
            Amdx = ret_A(ret_r,ret_v,mdx[i])
            if Vmdx == 0:
                continue
            
            ret_r,ret_v = [],[]
            for k in range(len(k1)):
                r,v,a,k1[k] = ret2(k1[k]-tol,t,k1[k]+tol,mdy[i],1)
                k1[k] = k1[k][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
            Vmdy = ret_pot(ret_r,ret_v,mdy[i])
            Amdy = ret_A(ret_r,ret_v,mdy[i])
            if Vmdy == 0:
                continue
            
            ret_r,ret_v = [],[]
            for k in range(len(k2)):
                r,v,a,k2[k] = ret2(k2[k]-tol,t,k2[k]+tol,mdz[i],1)
                k2[k] = k2[k][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
            Vmdz = ret_pot(ret_r,ret_v,mdz[i])
            Amdz = ret_A(ret_r,ret_v,mdz[i])
            if Vmdz == 0:
                continue
            
            ret_r,ret_v = [],[]
            for k in range(len(k3)):
                r,v,a,k3[k] = ret2(k3[k]-tol,t,k3[k]+tol,pdx[i],1)
                k3[k] = k3[k][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
            Vpdx = ret_pot(ret_r,ret_v,pdx[i])
            Apdx = ret_A(ret_r,ret_v,pdx[i])
            if Vpdx == 0:
                continue
            
            ret_r,ret_v = [],[]
            for k in range(len(k4)):
                r,v,a,k4[k] = ret2(k4[k]-tol,t,k4[k]+tol,pdy[i],1)
                k4[k] = k4[k][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
            Vpdy = ret_pot(ret_r,ret_v,pdy[i])
            Apdy = ret_A(ret_r,ret_v,pdy[i])
            if Vpdy == 0:
                continue
            
            ret_r,ret_v = [],[]
            for k in range(len(k5)):
                r,v,a,k5[k] = ret2(k5[k]-tol,t,k5[k]+tol,pdz[i],1)
                k5[k] = k5[k][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
            Vpdz = ret_pot(ret_r,ret_v,pdz[i])
            Apdz = ret_A(ret_r,ret_v,pdz[i])
            if Vpdz == 0:
                continue
            
            ret_r,ret_v= [],[]
            for k in range(len(k6)):
                r,v,a,k6[k] = ret2(k6[k]-tol,t+ Dt,k6[k]+tol,pos0[i],1)
                k6[k] = k6[k][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
            Apdt = ret_A(ret_r,ret_v,pos0[i])
            if np.linalg.norm(Apdt) == 0:
                continue
            
            ret_r,ret_v = [],[]
            for k in range(len(k7)):
                r,v,a,k7[k] = ret2(k7[k]-tol,t - Dt,k7[k]+tol,pos0[i],1)
                k7[k] = k7[k][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
            Amdt = ret_A(ret_r,ret_v,pos0[i])
            if np.linalg.norm(Amdt) == 0:
                continue
            
            o = -np.array(((Vpdx-Vmdx)/(2*dx),(Vpdy-Vmdy)/(2*dy),(Vpdz-Vmdz)/(2*dz)))
            o = o - ((Apdt - Amdt)/(2*Dt))
            Ax = (Apdx - Amdx) / (2*dx)
            Ay = (Apdy - Amdy) / (2*dy)
            Az = (Apdz - Amdz) / (2*dz)
            l = np.array((Ay[2] - Az[1],Az[0] - Ax[2],Ax[1] - Ay[0]))
            E.append(o)
            B.append(l)

    E,B = np.array(E),np.array(B)
    return E,B
E,B = planefields(t,pos0,10,5000)
#%%
S = poynt(E,B)
#%%
I = []
for i in range(len(S)):
    o = np.linalg.norm(S[i])
    o = np.log(o)
    I.append(o)
#%%
count = 0
for i in range(len(pos0)):
    count = count + len(pos0[i])

POS = np.zeros((count,3))
o = 0
for i in range(len(pos0)):
    POS[o:o+len(pos0[i]),:] = pos0[i]
    o = o + len(pos0[i])
#%%
Xs = POS[:,0]
Ys = POS[:,1]
fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111)
ax.set_facecolor('black')
sc = plt.scatter(Xs[:],Ys[:], s = 1, c=I[:], cmap=plt.cm.gray)
ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
cbar = fig.colorbar(sc, orientation='horizontal',label='log(S) / Wm$^{-2}$')    
plt.savefig('/Users/Owner/Documents/Project/Graphs/pion+,E=6GeV,B=0.png')
#%%
h = POS[4][0]
D = r[10**6][2]-POS[4][2]
gam = np.arcsin(cn/v0[2])
Rq = h/np.sin(gam)
d = h/np.tan(gam)
tq = (D-d)/v0[2]
Eq = (-q*((v0[2]/cn)**2 -1)**(1/4))/(2**(3/2) *np.pi*eps*cn**(3/2) *Rq**(1/2))
Eq = Eq*(-(1/2)*tq**(-3/2))
Rqn = np.array((h,0,-d))/np.linalg.norm(np.array((h,0,-d)))
print(Eq)
print(POS[4])
print(E[3])
#%%
Rq = -r[10**6]+POS[4]
D = -Rq[2]
h = Rq[0]
d = h/np.tan(gam)
magR = np.linalg.norm(Rq)
theta = np.arctan(h/D)
theta = np.pi - theta
Eq = (-q*((v0[2]/cn)**2 -1))/(2*np.pi*eps*magR**2 *(1-(np.sin(theta)**2 *(v0[2]/cn)**2))**(3/2))
print(Eq*(Rq/magR))
print(POS[4])
print(E[2])
#%%
fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111)
ax.quiver(POS[2:,2], POS[2:,0], E[0:,2], E[0:,0])
ax.set_xticks([POS[23][2],POS[15][2],POS[10][2],POS[5][2],POS[2][2],POS[2][2]+(POS[2][2]-POS[5][2])])
ax.set_xlabel('z / m')
ax.set_ylabel('x / m')
#%%
ax = plt.figure().add_subplot(projection='3d')
ax.quiver(POSX, POSY, POSZ, SX, SY, SZ, length=0.002, normalize=True)
#%%
plt.plot('elec,v=cn+8x10^4,B=50uT.png')
#%%
np.save('/Users/Owner/Documents/Project/Data/muon,E=6GeV,B=0.npy',[Xs,Ys,I])

#%%
dx = 10**-6
dy = 10**-6
dz = 10**-6
Dt = dt/2
def fields(t,pos,tol):
    n = len(pos)
    pdx = pos + np.array((dx,0,0))
    pdy = pos + np.array((0,dy,0))
    pdz = pos + np.array((0,0,dz))
    mdx = pos - np.array((dx,0,0))
    mdy = pos - np.array((0,dy,0))
    mdz = pos - np.array((0,0,dz))
    E = np.zeros((n,3))
    B = np.zeros((n,3))
    k0 = ret2(0,t,K,mdx[0],1)[3]
    k1 = ret2(k0[0]-tol,t,K,mdy[0],1)[3]
    k2 = ret2(k0[0]-tol,t,K,mdz[0],1)[3]
    k3 = ret2(k0[0]-tol,t,K,pdx[0],1)[3]
    k4 = ret2(k0[0]-tol,t,K,pdy[0],1)[3]
    k5 = ret2(k0[0]-tol,t,K,pdz[0],1)[3]
    k6 = ret2(k0[0]-tol,t+Dt,K,pos[0],1)[3]
    k7 = ret2(k0[0]-tol,t-Dt,K,pos[0],1)[3]
    for i in range(n):
        ret_r,ret_v,ret_a,k0 = ret2(k0[0]-tol,t,k0[0]+tol,mdx[i],1)
        if len(ret_r) == 0:
            ret_r,ret_v,ret_a,k0 = ret2(0,t,K,mdx[i],1)
        Vmdx = ret_pot(ret_r,ret_v,mdx[i])
        Amdx = ret_A(ret_r,ret_v,mdx[i])
        if Vmdx == 0:
            continue
        
        ret_r,ret_v,ret_a,k1 = ret2(k1[0]-tol,t,k1[0]+tol,mdy[i],1)
        if len(ret_r) == 0:
            ret_r,ret_v,ret_a,k1 = ret2(0,t,K,mdy[i],1)
        Vmdy = ret_pot(ret_r,ret_v,mdy[i])
        Amdy = ret_A(ret_r,ret_v,mdy[i])
        if Vmdy == 0:
            continue
        
        ret_r,ret_v,ret_a,k2 = ret2(k2[0]-tol,t,k2[0]+tol,mdz[i],1)
        if len(ret_r) == 0:
            ret_r,ret_v,ret_a,k2 = ret2(0,t,K,mdz[i],1)
        Vmdz = ret_pot(ret_r,ret_v,mdz[i])
        Amdz = ret_A(ret_r,ret_v,mdz[i])
        if Vmdz == 0:
            continue
        
        ret_r,ret_v,ret_a,k3 = ret2(k3[0]-tol,t,k3[0]+tol,pdx[i],1)
        if len(ret_r) == 0:
            ret_r,ret_v,ret_a,k3 = ret2(0,t,K,pdx[i],1)
        Vpdx = ret_pot(ret_r,ret_v,pdx[i])
        Apdx = ret_A(ret_r,ret_v,pdx[i])
        if Vpdx == 0:
            continue
        
        ret_r,ret_v,ret_a,k4 = ret2(k4[0]-tol,t,k4[0]+tol,pdy[i],1)
        if len(ret_r) == 0:
            ret_r,ret_v,ret_a,k4 = ret2(0,t,K,pdy[i],1)
        Vpdy = ret_pot(ret_r,ret_v,pdy[i])
        Apdy = ret_A(ret_r,ret_v,pdy[i])
        if Vpdy == 0:
            continue
        
        ret_r,ret_v,ret_a,k5 = ret2(k5[0]-tol,t,k5[0]+tol,pdz[i],1)
        if len(ret_r) == 0:
            ret_r,ret_v,ret_a,k5 = ret2(0,t,K,pdz[i],1)
        Vpdz = ret_pot(ret_r,ret_v,pdz[i])
        Apdz = ret_A(ret_r,ret_v,pdz[i])
        if Vpdz == 0:
            continue
        
        ret_r,ret_v,ret_a,k6 = ret2(k6[0]-tol,t+ Dt,k6[0]+tol,pos[i],1)
        if len(ret_r) == 0:
            ret_r,ret_v,ret_a,k6 = ret2(0,t + Dt,K,pos[i],1)
        Apdt = ret_A(ret_r,ret_v,pos[i])
        if np.linalg.norm(Apdt) == 0:
            continue
        
        ret_r,ret_v,ret_a,k7 = ret2(k7[0]-tol,t - Dt,k7[0]+tol,pos[i],1)
        if len(ret_r) == 0:
            ret_r,ret_v,ret_a,k7 = ret2(0,t-Dt,K,pos[i],1)
        Amdt = ret_A(ret_r,ret_v,pos[i])
        if np.linalg.norm(Amdt) == 0:
            continue
        
        E[i] = -np.array(((Vpdx-Vmdx)/(2*dx),(Vpdy-Vmdy)/(2*dy),(Vpdz-Vmdz)/(2*dz)))
        E[i] = E[i] - ((Apdt - Amdt)/(2*Dt))
        Ax = (Apdx - Amdx) / (2*dx)
        Ay = (Apdy - Amdy) / (2*dy)
        Az = (Apdz - Amdz) / (2*dz)
        B[i,0] = Ay[2] - Az[1]
        B[i,1] = Az[0] - Ax[2]
        B[i,2] = Ax[1] - Ay[0]
    return E,B
E,B = fields(t,pos,1000)
#%%
print(S)
#%%
def LWfields2(t,pos):
    n = len(pos)
    E = np.zeros((n,3))
    B = np.zeros((n,3))
    ret_r,ret_v,ret_a,ret_k = ret2(0,t,K,pos[0],1)
    for i in range(n):
        R = pos[i] - ret_r[0]
        Rn = R/np.linalg.norm(R)
        u = c*Rn - ret_v[0]
        o = (q*np.linalg.norm(R))/(4*np.pi*eps*(np.dot(R,u))**3)
        E[i] = E[i]+(o*(np.cross(R,np.cross(u,ret_a[0]))))
        B[i] = B[i]+(np.cross(Rn,E[i])/c)
    return E,B
E1,B1 = LWfields2(t,pos)         
#%%
dr = 1*10**-8
def fields2(k,pos,tol):
    n = len(pos)
    pdx = pos + np.array((dr,0,0))
    pdy = pos + np.array((0,dr,0))
    pdz = pos + np.array((0,0,dr))
    mdx = pos - np.array((dr,0,0))
    mdy = pos - np.array((0,dr,0))
    mdz = pos - np.array((0,0,dr))
    E = np.zeros((n,3))
    B = np.zeros((n,3))
    k0 = ret(0,k,mdx[0],1)[2]
    for i in range(n):
        ret_r,ret_v,k0 = ret(k0[0]-tol,k,mdx[i],1)
        Vmdx = ret_pot(ret_r,ret_v,mdx[i])
        Amdx = ret_A(ret_r,ret_v,mdx[i])
        
        ret_r,ret_v = ret(k0[0]-tol,k,mdy[i],1)[0:2]
        Vmdy = ret_pot(ret_r,ret_v,mdy[i])
        Amdy = ret_A(ret_r,ret_v,mdy[i])
        
        ret_r,ret_v = ret(k0[0]-tol,k,mdz[i],1)[0:2]
        Vmdz = ret_pot(ret_r,ret_v,mdz[i])
        Amdz = ret_A(ret_r,ret_v,mdz[i])
        
        ret_r,ret_v = ret(k0[0]-tol,k,pdx[i],1)[0:2]
        Vpdx = ret_pot(ret_r,ret_v,pdx[i])
        Apdx = ret_A(ret_r,ret_v,pdx[i])
        
        ret_r,ret_v = ret(k0[0]-tol,k,pdy[i],1)[0:2]
        Vpdy = ret_pot(ret_r,ret_v,pdy[i])
        Apdy = ret_A(ret_r,ret_v,pdy[i])
        
        ret_r,ret_v = ret(k0[0]-tol,k,pdz[i],1)[0:2]
        Vpdz = ret_pot(ret_r,ret_v,pdz[i])
        Apdz = ret_A(ret_r,ret_v,pdz[i])
        
        ret_r,ret_v = ret(k0[0]-tol,k+1,pos[i],1)[0:2]
        Apdt = ret_A(ret_r,ret_v,pos[i])
    
        ret_r,ret_v = ret(k0[0]-tol,k-1,pos[i],1)[0:2]
        Amdt = ret_A(ret_r,ret_v,pos[i])
       
        E[i] = -np.array((Vpdx-Vmdx,Vpdy-Vmdy,Vpdz-Vmdz))/(2*dr)
        E[i] = E[i] - ((Apdt - Amdt)/(2*dt))
        Ax = (Apdx - Amdx) / (2*dr)
        Ay = (Apdy - Amdy) / (2*dr)
        Az = (Apdz - Amdz) / (2*dr)
        B[i,0] = Ay[2] - Az[1]
        B[i,1] = Az[0] - Ax[2]
        B[i,2] = Ax[1] - Ay[0]
    return E,B
E,B = fields2(8*10**4,np.array((0,0,0.001)),10) 
#%%
def LWfields(k,pos,tol):
    n = len(pos)
    E = np.zeros((n,3))
    B = np.zeros((n,3))
    k0 = ret(0,k,pos[0],1)[3]
    if len(k0) == 0:
        return 'increase timestep'
    k0 = k0[0] - tol
    for i in range(n):
        ret_r,ret_v,ret_a = ret(k0,k,pos[i],1)[0:3]
        R = pos[i] - ret_r[0]
        Rn = R/np.linalg.norm(R)
        u = c*Rn - ret_v[0]
        o = (q*np.linalg.norm(R))/(4*np.pi*eps*(np.dot(R,u))**3)
        E[i] = E[i]+(o*(np.cross(R,np.cross(u,ret_a[0]))))#+((c**2 - np.linalg.norm(ret_v[0])**2)*u)))
        B[i] = B[i]+(np.cross(Rn,E[i])/c)
            
    return E,B

E1,B1 = LWfields(8*10**4,pos,10)
#%%
def poynt(E,B):
    S = np.zeros((len(E),3))
    for i in range(len(E)):
        S[i] = np.cross(E[i],B[i]) / mu
        
    return S
#S = poynt(E,B)
#S1 = poynt(E1,B1)
#%%
def power2(S,pos):
    P = 0  
    a = dthe*dphi*(np.pi**2)*R**2
    beta = v[kret]/c
    for i in range(len(pos)):
        nhat = (pos[i]-w)/np.linalg.norm(pos[i] - w)
        g = 1 - np.dot(beta,nhat)
        o = g*np.dot(S[i],(pos[i]-w)/np.linalg.norm(pos[i]-w))
        o = o*np.sin(pos2[i,1])*a
        P = P + o
    return P

P = power2(S,pos)
#P1 = power2(S1,pos)
print(P)
#print(P1)
#%%
Power = (2*(q**2)*gamma(np.linalg.norm(v[kret]))**6)/(4*np.pi*eps*3*c**3)
Power = Power*((np.dot(a[kret],a[kret])) - (np.linalg.norm(np.cross(v[kret],a[kret]))/c)**2)
print(Power)
#%%
dr = 10**-8
Dt = dt/10
def fields3(k,pos,num,tol,tol2):
    pdx = pos + np.array((dr,0,0))
    pdy = pos + np.array((0,dr,0))
    pdz = pos + np.array((0,0,dr))
    mdx = pos - np.array((dr,0,0))
    mdy = pos - np.array((0,dr,0))
    mdz = pos - np.array((0,0,dr))
    E = np.zeros((num,3))
    B = np.zeros((num,3))
    k00 = k
    for i in range(num):
        k0 = ret2(0,k*dt,K,mdx,0)[3]
        n = len(k0)
        if n != 0:
            break
        k= k+1
    print(n)
    k1 = []
    k2 = []
    k3 = []
    k4 = []
    k5 = []
    k6 = []
    k7 = []
    
    for i in range(n):
        o = ret2(k0[i]-tol,k*dt,k0[i]+tol,mdy,1)[3]
        k1.append(o[0])
    
        o = ret2(k0[i]-tol,k*dt,k0[i]+tol,mdz,1)[3]
        k2.append(o[0])
    
        o = ret2(k0[i]-tol,k*dt,k0[i]+tol,pdx,1)[3]
        k3.append(o[0])
    
        o = ret2(k0[i]-tol,k*dt,k0[i]+tol,pdy,1)[3]
        k4.append(o[0])
    
        o = ret2(k0[i]-tol,k*dt,k0[i]+tol,pdz,1)[3]
        k5.append(o[0])
    
        o = ret2(k0[i]-(10*tol),k*dt-Dt,k0[i]+(10*tol),pos,1)[3]
        k6.append(o[0])
    
        o = ret2(k0[i]-(10*tol),(k*dt)+Dt,k0[i]+(10*tol),pos,1)[3]
        k7.append(o[0])
    l = 0
    for i in range(num-(k-k00)):
        if i == 10:
            l=1
        print(i)
        ret_r,ret_v = [],[]
        
        for j in range(l,n):
            if j ==1:
                r,v,a,k0[j] = ret2(k0[j]-1,(k+i)*dt,k0[j]+tol2,mdx,1)
                k0[j] = k0[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])  
            else:
                r,v,a,k0[j] = ret2(k0[j]-tol2,(k+i)*dt,k0[j]+1,mdx,1)
                k0[j] = k0[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])  
        print(ret_r)
        Vmdx = ret_pot(ret_r,ret_v,mdx)
        Amdx = ret_A(ret_r,ret_v,mdx)
    
        ret_r,ret_v = [],[]
        for j in range(l,n):
            if j ==1:
                r,v,a,k1[j] = ret2(k1[j]-1,(k+i)*dt,k1[j]+tol2,mdy,1)
                k1[j] = k1[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])  
            else:
                r,v,a,k1[j] = ret2(k1[j]-tol2,(k+i)*dt,k1[j]+1,mdy,1)
                k1[j] = k1[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
        Vmdy = ret_pot(ret_r,ret_v,mdy)
        Amdy = ret_A(ret_r,ret_v,mdy)
        
        ret_r,ret_v = [],[]
        for j in range(l,n):
            if j ==1:
                r,v,a,k2[j] = ret2(k2[j]-1,(k+i)*dt,k2[j]+tol2,mdz,1)
                k2[j] = k2[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])  
            else:
                r,v,a,k2[j] = ret2(k2[j]-tol2,(k+i)*dt,k2[j]+1,mdz,1)
                k2[j] = k2[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
        Vmdz = ret_pot(ret_r,ret_v,mdz)
        Amdz = ret_A(ret_r,ret_v,mdz)
        
        ret_r,ret_v = [],[]
        for j in range(l,n):
            if j ==1:
                r,v,a,k3[j] = ret2(k3[j]-1,(k+i)*dt,k3[j]+tol2,pdx,1)
                k3[j] = k3[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])  
            else:
                r,v,a,k3[j] = ret2(k3[j]-tol2,(k+i)*dt,k3[j]+1,pdx,1)
                k3[j] = k3[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
        Vpdx = ret_pot(ret_r,ret_v,pdx)
        Apdx = ret_A(ret_r,ret_v,pdx)
        
        ret_r,ret_v = [],[]
        for j in range(l,n):
            if j ==1:
                r,v,a,k4[j] = ret2(k4[j]-1,(k+i)*dt,k4[j]+tol2,pdy,1)
                k4[j] = k4[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])  
            else:
                r,v,a,k4[j] = ret2(k4[j]-tol2,(k+i)*dt,k4[j]+1,pdy,1)
                k4[j] = k4[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
        Vpdy = ret_pot(ret_r,ret_v,pdy)
        Apdy = ret_A(ret_r,ret_v,pdy)
        
        ret_r,ret_v = [],[]
        for j in range(l,n):
            if j ==1:
                r,v,a,k5[j] = ret2(k5[j]-1,(k+i)*dt,k5[j]+tol2,pdz,1)
                k5[j] = k5[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])  
            else:
                r,v,a,k5[j] = ret2(k5[j]-tol2,(k+i)*dt,k5[j]+1,pdz,1)
                k5[j] = k5[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
        Vpdz = ret_pot(ret_r,ret_v,pdz)
        Apdz = ret_A(ret_r,ret_v,pdz)
        
        ret_r,ret_v = [],[]
        for j in range(l,n):
            if j == 1:
                r,v,a,k6[j] = ret2(k6[j]-1,((k+i)*dt)-Dt,k6[j]+tol2,pos,1)
                k6[j] = k6[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
            else:
                r,v,a,k6[j] = ret2(k6[j]-tol2,((k+i)*dt)-Dt,k6[j]+1,pos,1)
                k6[j] = k6[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
        Amdt = ret_A(ret_r,ret_v,pos)
        
        ret_r,ret_v = [],[]
        for j in range(l,n):
            if j == 1:
                r,v,a,k7[j] = ret2(k7[j]-1,((k+i)*dt)+Dt,k7[j]+tol2,pos,1)
                k7[j] = k7[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
            else:
                r,v,a,k7[j] = ret2(k7[j]-tol2,((k+i)*dt)+Dt,k7[j]+1,pos,1)
                k7[j] = k7[j][0]
                ret_r.append(r[0])
                ret_v.append(v[0])
                
        Apdt = ret_A(ret_r,ret_v,pos)
        
        E[i+(k-k00)] = -np.array((Vpdx-Vmdx,Vpdy-Vmdy,Vpdz-Vmdz))/(2*dr)
        E[i+(k-k00)] = E[i+(k-k00)] - ((Apdt - Amdt)/(2*Dt))
        Ax = (Apdx - Amdx) / (2*dr)
        Ay = (Apdy - Amdy) / (2*dr)
        Az = (Apdz - Amdz) / (2*dr)
        B[i+(k-k00),0] = Ay[2] - Az[1]
        B[i+(k-k00),1] = Az[0] - Ax[2]
        B[i+(k-k00),2] = Ax[1] - Ay[0]
    
    return E,B
#%%
num = 400
Er,Br = fields3(np.int(t/dt),pos0[65][0],num,5000,80000)
print(Er,Br)
#S1 = poynt(Er1,Br1)
#%%
def LWfields2(k,pos,K):
    E = np.zeros((K,3))
    B = np.zeros((K,3))
    k0 = ret(0,k,pos,1)[3]
    for i in range(K):
        ret_r,ret_v,ret_a,k0 = ret(k0[0],k+i,pos,1)
        R = pos - ret_r[0]
        Rn = R/np.linalg.norm(R)
        u = c*Rn - ret_v[0]
        o = (q*np.linalg.norm(R))/(4*np.pi*eps*(np.dot(R,u))**3)
        #E[i] = (c**2 - np.linalg.norm(ret_v[0])**2)*u
        E[i] = o*(E[i] + np.cross(R,np.cross(u,ret_a[0])))
        B[i] = (np.cross(Rn,E[i])/c)
            
    return E,B

#E,B = LWfields2((10**5),np.array((0.001,-rad[0],0)),5000,0.5)
num = 10**6
Er,Br = LWfields2(T/dt,np.array((0,0,10)),10)
#%%
St = poynt(Er,Br)
print(St)
#%%
Ts = []
for i in range(len(St)):
    o = i*dt
    Ts.append(o)

Stnorm = []
for i in range(len(St)):
    Stnorm.append((np.log(np.linalg.norm(St[i,:]))))
#%%
fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111)
ax.set_xlabel('Time / s')
ax.set_ylabel('log(S) / Wm$^{-2}$ ')
plt.plot(Ts[:1500],Stnorm[:1500])

#%%
plt.plot(Ts[5:],Br1[5:,1])
#%%
plt.plot(Ts[:],S1[:,2])
#%%
plt.plot(Ts[:],S1[:,2])
#%%
np.save('synchv=0.9numstart2.npy',S)
#%%
S = np.load('synchv=0.9numstart.npy')
#%%
S1norm = []
for i in range(len(S)):
    S1norm.append(np.linalg.norm(S[i,:]))
    
FFT = np.fft.fft(S1norm)
N = len(FFT)
freq = np.zeros(N)
for i in range(N):
    freq[i] = i/(N*dt)

plt.plot(freq[1:13],abs(FFT[1:13]))
#%%
print(freq[])
#%%
plt.plot(freq1[1:500],abs(FFT1[1:500]))
#%%
plt.plot(freq[1:500],abs(FFT5[1:500])) 

#%% 
print(freq[np.argmax(abs(FFT1[60:100]))+60]) 
print(np.max(abs(FFT1[60:100])))  
#%%
dthe = np.pi/200
tol = 200
dr = 1*10**-5
R = 10
Ss = np.zeros(200)
k_ret = np.argmin(r[K-500:K,0]) + K-500
t_ret = k_ret*dt
T = R/c + t_ret
#%%
k0 = ret2(0,T,K,np.array((r[k_ret,0],0,R)),1)[3]
for i in range(200):
    pos = np.array((r[k_ret,0],R*np.sin((i*dthe)-np.pi/2),R*np.cos((i*dthe)-np.pi/2)))
    pdx = pos + np.array((dr,0,0))
    pdy = pos + np.array((0,dr,0))
    pdz = pos + np.array((0,0,dr))
    mdx = pos - np.array((dr,0,0))
    mdy = pos - np.array((0,dr,0))
    mdz = pos - np.array((0,0,dr))

    ret_r,ret_v = ret2(k0[0] - tol,T,K,pdx,1)[0:2]
    Vpdx = ret_pot(ret_r,ret_v,pdx)
    Apdx = ret_A(ret_r,ret_v,pdx)
    
    ret_r,ret_v = ret2(k0[0] - tol,T,K,mdx,1)[0:2]
    Vmdx = ret_pot(ret_r,ret_v,mdx)
    Amdx = ret_A(ret_r,ret_v,mdx)
    
    ret_r,ret_v = ret2(k0[0] - tol,T,K,pdy,1)[0:2]
    Vpdy = ret_pot(ret_r,ret_v,pdy)
    Apdy = ret_A(ret_r,ret_v,pdy)
    
    ret_r,ret_v = ret2(k0[0] - tol,T,K,mdy,1)[0:2]
    Vmdy = ret_pot(ret_r,ret_v,mdy)
    Amdy = ret_A(ret_r,ret_v,mdy)
    
    ret_r,ret_v = ret2(k0[0] - tol,T,K,pdz,1)[0:2]
    Vpdz = ret_pot(ret_r,ret_v,pdz)
    Apdz = ret_A(ret_r,ret_v,pdz)
    
    ret_r,ret_v = ret2(k0[0] - tol,T,K,mdz,1)[0:2]
    Vmdz = ret_pot(ret_r,ret_v,mdz)
    Amdz = ret_A(ret_r,ret_v,mdz)
    
    ret_r,ret_v = ret2(k0[0] - tol,T+dt,K,pos,1)[0:2]
    Vpdt = ret_pot(ret_r,ret_v,pos)
    Apdt = ret_A(ret_r,ret_v,pos)
    
    ret_r,ret_v = ret2(k0[0] - tol,T-dt,K,pos,1)[0:2]
    Vmdt = ret_pot(ret_r,ret_v,pos)
    Amdt = ret_A(ret_r,ret_v,pos)
    
    E = -np.array((Vpdx-Vmdx,Vpdy-Vmdy,Vpdz-Vmdz))/(2*dr)
    E = E - ((Apdt - Amdt)/(2*dt))
    Ax = (Apdx - Amdx) / (2*dr)
    Ay = (Apdy - Amdy) / (2*dr)
    Az = (Apdz - Amdz) / (2*dr)
    B = np.array((Ay[2] - Az[1],Az[0] - Ax[2],Ax[1] - Ay[0]))
    Ss[i] = np.linalg.norm(np.cross(E,B) / mu)
#%%
nhat = np.array((0,0,1))
gret = 1-np.dot(nhat,v0/c)
Ea = (q/(4*np.pi*eps)) * ((np.cross(nhat,np.cross(nhat-(v0/c),acc(v0)/c)))/(c*(gret**3)*R))
Ba = (mu*q/(4*np.pi)) * ((((np.cross((v0/c),nhat)*np.dot(acc(v0)/c,nhat)))+(gret*np.dot(acc(v0)/c,nhat)))/((gret**3)*R))
print(np.linalg.norm(Ea))
#%%
Ss2 = np.zeros(500)
k_ret = np.argmin(r[K-10000:K,0]) + K-10000
t_ret = k_ret*dt
T = R/c + t_ret
k0 = ret2(np.array((r[k_ret,0],0,R)),T,0)[3]

for i in range(500):    
    pos = np.array((r[k_ret,0],R*np.sin((i*dthe)-np.pi/2),R*np.cos((i*dthe)-np.pi/2))) 
    ret_r,ret_v,ret_a = ret2(pos,T,k0[0] - 100)[0:3]
    R2 = pos- ret_r[0]
    Rn = R2/np.linalg.norm(R2)
    u = c*Rn - ret_v[0]
    o = (q*np.linalg.norm(R2))/(4*np.pi*eps*(np.dot(R2,u))**3)
    E = o*(np.cross(R2,np.cross(u,ret_a[0])))
    B = (np.cross(Rn,E)/c)
    Ss2[i] = np.linalg.norm(np.cross(E,B) / mu)
    
#%%
np.save('synchthetav=0.5.npy',Ss)
#%% 
theta = []    
for i in range(200):
    o = i*dthe - np.pi/2
    theta.append(o)
    
plt.plot(theta[:],Ss[:])