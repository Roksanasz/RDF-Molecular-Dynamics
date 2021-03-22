import numpy as np
import matplotlib.pyplot as plt
from asap3 import LennardJones, Atoms, units
from ase.lattice.cubic import  FaceCenteredCubic
from asap3.md.langevin import Langevin
from asap3.md.velocitydistribution import *
from ase.md.velocitydistribution import ZeroRotation
from asap3.analysis.rdf import RadialDistributionFunction
#d=5.5
#d=7.0
#d=9.0
d=10.5 
#- zwiększając d zmniejszamy gęstość, będziemy mieć gaz, wtedy zwiększamy ilość kroków
#to spowoduje wygładzenie wykresu
n1=5
#T=120
#T=140
#T=160
T=180
#T=200
eps=226*units.kB        #epsilon: 226 K
sigmXe=3.95              #sigma: 3.95 Angstrema

atoms = FaceCenteredCubic(size=(n1,n1,n1), symbol="Xe",    pbc=(1,1,1),latticeconstant=d)
atoms.set_calculator(LennardJones([54],epsilon=eps,sigma=sigmXe))
# wybieramy krok czasowy = 5 fs i tarcie = 0.01 
dyn =Langevin(atoms, 5 * units.fs, T * units.kB, 0.01)
MaxwellBoltzmannDistribution(atoms, 1.2*T*units.kB)
Stationary(atoms)
ZeroRotation(atoms)

print("start")
# 2000. kroków DM 
dyn.run(2000)
ekin = atoms.get_kinetic_energy() / atoms.get_global_number_of_atoms()
ro=atoms.get_global_number_of_atoms()/atoms.get_volume()*sigmXe**3 
print ("T = %.1f K   ro=%.2f" %( 2.0/3.0*ekin/units.kB,ro))
print('pomiar RDF')
rMax=10
nBins=200 
#M=200
RDFobj = RadialDistributionFunction(atoms, rMax, nBins)
dyn.attach(RDFobj.update, interval=5)
RDFobj.output_file("Xe_rdf1")
dyn.run(100000)
ekin = atoms.get_kinetic_energy() / atoms.get_global_number_of_atoms()
print ("T = %.1f K   ro=%.2f" %( 2.0/3.0*ekin/units.kB,ro))
# Get the RDF and plot it.
rdf = RDFobj.get_rdf()
x = np.arange(nBins)/ nBins * rMax 

plt.plot(x, rdf)
tyt="T="+str(T)+'  ro='+str(round(ro,2))
plt.title(tyt)
plt.ylabel('RDF')
plt.xlabel('r')
plt.show()