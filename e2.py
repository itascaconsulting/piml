import itasca as it
import numpy as np
from glob import glob

class _gpa:
    def disp(self):
        return np.array(tuple(gp.disp() for gp in it.gridpoint.list()))
    def pos(self):
        return np.array(tuple(gp.pos() for gp in it.gridpoint.list()))
gpa = _gpa()


it.command("""
python-reset-state false
program load module 'zonepython'
model new
zone create quad size 10 4
model large-strain off
zone cmodel assign elastic

zone face apply velocity-normal 0 range pos-x 0
zone face apply velocity-normal 0 range pos-x 10
zone face apply velocity-normal 0 range pos-y 0

zone face apply stress-normal -1e6 range pos-y 4 pos-x 0 1.1
""")

for cube in [f"cube_4_{_}.npy" for _ in (1,)]:
    print(cube)
    X = np.load(cube)
    Y = []
    for i, (e0,e1,e2,e3) in enumerate(X):
        e0,e1,e2,e3 = [3.93976043e+08, 2.57031364e+09, 2.15342752e+08, 4.96253093e+09]
        if i % 1000 == 0: print(i)
        it.command(f"""
            zone ini stress 
            zone grid ini displacement 0 0
            zone property density 2800 young {e0} poisson 0.225 range pos-y 0 1
            zone property density 2800 young {e1} poisson 0.225 range pos-y 1 2
            zone property density 2800 young {e2} poisson 0.225 range pos-y 2 3
            zone property density 2800 young {e3} poisson 0.225 range pos-y 3 4
            model solve ratio 1e-6
            
            """)
        Y.append(gpa.disp())
    1/0
    np.save("result_" + cube, Y)