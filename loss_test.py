import itasca as it
import numpy as np
from glob import glob

class _gpa:
    def disp(self):
        return np.array(tuple(gp.disp() for gp in it.gridpoint.list()))
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
zone ini stress 
zone grid ini displacement 0 0
""")

X = np.load("tc_X.npy")
e0, e1, e2, e3 = X
Y = np.load("tc_Ya.npy")
Y = Y.reshape(55,2)

for v, gp in zip(Y, it.gridpoint.list()):
    gp.set_vel(v/10.0)

it.command(f"""
    zone ini stress 
    zone grid ini displacement 0 0
    zone gridpoint fix velocity 
    zone property density 2800 young {e0} poisson 0.225 range pos-y 0 1
    zone property density 2800 young {e1} poisson 0.225 range pos-y 1 2
    zone property density 2800 young {e2} poisson 0.225 range pos-y 2 3
    zone property density 2800 young {e3} poisson 0.225 range pos-y 3 4
    model cycle 10
    zone gridpoint fix velocity 0 0
    ;zone face apply stress-normal 1e6 range pos-y 4 pos-x 0 1.1
    """)
