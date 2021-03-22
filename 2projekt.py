#w ustalonej temperaturze
import numpy as np
import matplotlib.pyplot as plt
from asap3 import LennardJones,Atoms, units
from ase.lattice.cubic import SimpleCubic
from asap3.md.langevin import Langevin
from asap3.md.velocitydistribution import *
from ase.md.velocitydistribution import ZeroRotation
from ase.lattice.cubic import  FaceCenteredCubic
from asap3.analysis.rdf import RadialDistributionFunction

av_kin=[]
av_vir=[]
def pomiar(a):
    n=a.get_global_number_of_atoms()
    ekin = a.get_kinetic_energy()/n
    av_kin.append(ekin)
    a.wrap()
    wirial= np.vdot(a.get_positions(), a.get_forces())/n
    av_vir.append(wirial)
eps=226*units.kB        #epsilon: 163 K
sigmXe=3.95              #sigma: 3.6 Angstrema
# każdy pomiar wymaga ustawienia warunków początkowych, 
#funkcja za każdym razem tworzy nowe atomy- linia 31, zadaje prędkości
#warunki początkowe nie są dobrze dostosowane do warunków równowagi- to powoduje, że są potrzebne dodatkowe obliczenia
def pomiary(d,T0,n1,tr,tpom):
    
    
    #mamy n1 komórek w każdej osi:
    #FaceCenteredCubic - dla metali szlachetnych takich jak Ksenon
    atoms = FaceCenteredCubic(size=(n1,n1,n1), symbol="Xe",    pbc=(1,1,1), #za każdym razem deklaruję strukturę atomową
                       latticeconstant=d)   #pierwszy agument funkcji pomiary jest przypisywany stałej sieci
                       #jesli zmieniamy stałą sieci (np. gdy ją zwiększamy to będzie sięzwiększać objętość)
    atoms.set_calculator(LennardJones([54],epsilon=eps,sigma=sigmXe))
    dyn =Langevin(atoms, 6 * units.fs, T0 * units.kB, 0.01)
    MaxwellBoltzmannDistribution(atoms, 1.2*T0*units.kB) #wykorzystujemy T0, ale jest specjalnie przemnożone przez 1.2, by byla trochę wyższa - wg wykładu- usprawnienie działania
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn.run(tr)     #uruchomienie tr kroków początkowych
    dyn.attach(pomiar, 5, atoms)
    dyn.run(tpom)       #uruchomienie dynamiki
    Ek_av=sum(av_kin)/len(av_kin)       #obliczenie średnich
    Tav=2.0/3.0*Ek_av/units.kB
# cisnienie w barach
    n=atoms.get_global_number_of_atoms()
    obj=atoms.get_volume()      #obliczenie objętości
    ro=n/obj*sigmXe**3          #obliczenie gęstości
    Pav=(2*Ek_av+sum(av_vir)/len(av_vir))/(3*obj/n)*1.602e6  #obliczenie ciśnienia -> (2 średnie Ek+średni wiriał/len)/3 objetosci n liczbę cząstek
    return n, ro, Tav, Pav
#pojawi sie kilka wynikow
#liczba atomow N, ro-gestosc- ok.0.11-dosc niska, zwlaszcza w wysokich temp.-zachowanie jak gaz, temperatura pomiaru, cisnienie oraz stosunek P/T


n1=4        #n1 zadaje liczbę komórek wzdłóż każdej osi czyli 256 w 3D
tr=2000
tpom=50000
T0 =100
rozm=[]
Pzm=[]
Tzm=[]
for d in np.arange(7.0,3.4,-0.25):
    av_kin=[]
    av_vir=[]
    Na,ro,Tsr,Psr=  pomiary(d,T0,n1,tr,tpom)
    rozm.append(ro)
    Pzm.append(Psr)
    Tzm.append(Tsr)
    print("N=%5d ro=%10.6f  T=%8.2f  P=%12.4f  P/T=%8.4f"%(Na,ro,Tsr,Psr,Psr/Tsr) )
#blisko konca jak ro okolo 0.9
#plt.rcParams["figure.figsize"] = (10,5)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,5)
v=[]
for x in rozm:
    v.append(1/x)
plt.plot(v,Pzm)
plt.plot(v,Pzm,'o')
tyt="T="+str(T0)
plt.title(tyt)
plt.ylabel('P')
plt.xlabel('v')
plt.show()
#wykres zaleznosci cisnienia od objetosci- beznadzieja
#gdyby to byl gaz to spodziewamy sie paraboli
#brzuszek- w duzych objetosciach są tendencje do wspoistnienia gazu i cieczy
# w mniejszych objetosciach bedzie sie skraplał i tworzyl ciecz
#wynik pokazuje pewne trudnosci- nalezy wybrac temperature np. 300K- bedzie gaz a wykres bedzie hiperbolą
#z prawa zaleznosci cisnienia od objetosci dla gazu doskonalego
# wykres- wymagane jest wykonanie kolenych krokow by wyjasnic kolejne kroki dynamiki molekularnej
#jesli zwiekszamy liczbe kroków to poprawiamy dokładność obliczeń


# wnioski - temperatura sie trzym +/- kilka dziesiatych stopnia od zadanej temperatury

#Radialna funkcja korelacyjna:
# jeżeli mamy atomy rozłożone w jakimś układzie to mamy jakąś średnią gęśtosć -> liczba atomów do objętości
#możemy policzyc:
#jaka jest ilosc atomow w otoczeniu, objetosc otoczenia, lokalną gęstość każdego punktu, to trzeba uśrednić
#RDF- stosunek lokalnej gęstości do średniej gęstości w układzie
#ma duże znaczenie w fizyce teoretycznej

#RADIALNA FUNKCJA KORELACYJNA

d=5.5
#d=10.5 - zwiększając zmniejszamy gęstość, będziemy mieć gaz, wtedy zwiększamy ilość kroków
#to spowoduje wygłądzenie wykresu
#T=60
T=140
#T=200
atoms = FaceCenteredCubic(size=(n1,n1,n1), symbol="Xe",    pbc=(1,1,1),latticeconstant=d)
atoms.set_calculator(LennardJones([18],epsilon=eps,sigma=sigmXe))
# wybieramy krok czasowy = 5 fs i tarcie = 0.01 
# wybieramy krok czasowy = 5 fs i tarcie = 0.01 
dyn =Langevin(atoms, 5 * units.fs, T * units.kB, 0.01)
MaxwellBoltzmannDistribution(atoms, 1.2*T*units.kB)
Stationary(atoms)
ZeroRotation(atoms)

print("start")
# 2000. kroków DM - używamy tyle kroków początkowych
dyn.run(2000)
ekin = atoms.get_kinetic_energy() / atoms.get_global_number_of_atoms()
ro=atoms.get_global_number_of_atoms()/atoms.get_volume()*sigmXe**3 
print ("T = %.1f K   ro=%.2f" %( 2.0/3.0*ekin/units.kB,ro)) #drukujemy temp.po fazie początkowej
print('pomiar RDF') #pomiar danej funkcji
rMax=10     #pomiar do 10 Angstremów
nBins=200 
#M=200
RDFobj = RadialDistributionFunction(atoms, rMax, nBins) #tworzymy obiekt RDF, wie do jakiego R ma mierzyc i ile utworzyć binów
dyn.attach(RDFobj.update, interval=5)       #stosujemy Langevena, tu podczepiamy obiekt metodą update, pobiera konfiguracje z interwałem 5(bo mierzymy ja jaka srednią - nie z 1 konf.)
RDFobj.output_file("Xe_rdf1")   #wrzucamy do pliku
dyn.run(10000)                  #tyle kroków dynamiki; to podzielne przez 5 daje ilosc konfiguracji
#zwiększenie ilości kroków!
ekin = atoms.get_kinetic_energy() / atoms.get_global_number_of_atoms()
print ("T = %.1f K   ro=%.2f" %( 2.0/3.0*ekin/units.kB,ro)) #zapis temperatury i gęstości
#Biorę RDF i robię wykres
rdf = RDFobj.get_rdf()          #pobieram rdf (wartości funkcji radialnej, korelacyjnej we wszystkich binach)
x = np.arange(nBins)/ nBins * rMax #zakres od zera do liczby binów/ dzielmy przez nBins (czyli mamy zakres od zera do 1 * rMax)


plt.plot(x, rdf)
tyt="T="+str(T)+'  ro='+str(round(ro,2))
plt.title(tyt)
plt.ylabel('RDF')
plt.xlabel('r')
plt.show()

#załóżmy że wartości równowagowe między atomami są 4 Angstremów to ja merzę do rMax = 10 Angstremow
#radialna funkcja korelacyjna zalezy od promienia, tu jest wyrażana w Angstremach
#nasz układ periodyczny ma n komórek elementarnych, w FaceCenteredCubic zadana jest stała sieci d
#n1*d= 4*5.5=22 rozmiar krawedzi szeszcianu w ktorym sa wszystkei atomy, rMax jest mniejsze od polowy, nie zadawałam za duzego
#rozmiar jednego przedzialiku 
#nBin- 200- zakres to 10 A. Jak zielimy przez 100 to 0.1, jak przez 2 to 0.05 to rozmiar przedzialiku.
#sprawdzamy ile odległosci wpada do takiego przedzialiku

#Wykres
#dobry wynik pokazuje to że dla dużych R-ów ma dążyć do 1

#czasami oscylacje - np drugie maksmum- schodzimy z temperaturą np do 40 K (sprawdz temp dla Ksenonu)
#przy gestosci 0.14 w temp 40 K atomy chcą sięjuż skraplać wiec model DM jest wątpliwy
#Końcówka RFK nie do końca jest prawdziwa - ok czyli będę badać w temp 120 K lub wyżej
# to iluzja bo mamy 256 atomów, jeśli zwiększymy n1=5 to mamy więcej atomów: 500
# troche wydluza sie czs obliczen