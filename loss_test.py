import itasca as it
import numpy as np
from glob import glob

class _gpa:
    def disp(self):
        return np.array(tuple(gp.disp() for gp in it.gridpoint.list()))
    def pos(self):
        return np.array(tuple(gp.pos() for gp in it.gridpoint.list()))
    def force_unbal(self):
        return np.array(tuple(gp.force_unbal() for gp in it.gridpoint.list()))    
    def force_app(self):
        return np.array(tuple(gp.force_app() for gp in it.gridpoint.list()))    
    def force_load(self):
        return np.array(tuple(gp.force_load() for gp in it.gridpoint.list()))  
    def set_extra(self, n, arr):
        for v, gp in zip(arr, it.gridpoint.list()):
            gp.set_extra(n, v)


gpa = _gpa()


it.command("""
python-reset-state false
program load module 'zonepython'
model new
zone create quad size 10 4
model large-strain off
zone cmodel assign elastic

;zone face apply velocity-normal 0 range pos-x 0
;zone face apply velocity-normal 0 range pos-x 10
;zone face apply velocity-normal 0 range pos-y 0
;zone face apply stress-normal 1e6 range pos-y 4 pos-x 0 1.1
""")

p = gpa.pos()
left_boundary = p[:,0] == 0
right_boundary = p[:,0] == 10.0
lower_boundary = p[:,1] == 0
vec_mask = np.ones_like(p)
vec_mask[:,0][left_boundary | right_boundary] = 0
vec_mask[:,1][lower_boundary] = 0

load_comp = np.zeros_like(p)
load_boundary = (p[:,1]==4) & (p[:,0]<=1.1)
load_comp[:,1][load_boundary] = 5e5

#gpa.set_extra(1, vec_mask[:,0]+vec_mask[:,1])

X = np.load("tc_X.npy")
e0, e1, e2, e3 = X
Y = np.load("tc_Ya.npy")
Y = Y.reshape(55,2)

for v, gp in zip(Y, it.gridpoint.list()):
    gp.set_vel(v)

it.command(f"""
    zone ini stress 
    zone grid ini displacement 0 0
    zone gridpoint fix velocity 
    zone property density 2800 young {e0} poisson 0.225 range pos-y 0 1
    zone property density 2800 young {e1} poisson 0.225 range pos-y 1 2
    zone property density 2800 young {e2} poisson 0.225 range pos-y 2 3
    zone property density 2800 young {e3} poisson 0.225 range pos-y 3 4
    model cycle 1    
    ;model solve ratio 1e-6
    """)

f = gpa.force_unbal()*vec_mask + load_comp
gpa.set_extra(1, np.linalg.norm(f, axis=1)/1e6)


#sum(np.linalg.norm(f, axis=1)/1e6)
#Out[59]: 1.822968775889534e-05
#
#sum(np.linalg.norm(f0, axis=1)/1e6)
#Out[60]: 1.299678721626384e-05
