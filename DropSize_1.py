import numpy as np   
import matplotlib.pyplot as plt
plt.close("all")

#----------------------------------
#Input parameters
#----------------------------------

figure_name = 'Fig_DropSize_MagmaOcean.pdf'
rho_m = 8000 # Metal density, kg / m^3
rho_s = 4000 #Silicate density, kg/m^3
G = 6.67e-11#Gravitational constant, SI units
fm = 0.16#Volume metal fraction in impactor. Cf Bonsor et al. 
fm_t = 0.16 #Volume metal fraction in target. 
Cp = 1.e3 # Heat capacity
DT = 1000 # Temperature jump across the mantle, in K 
alpha = 1.3e-5 #mantle thermal expansion coefficient, in 1/K
mu = 0.1 #Dynamic visocsity in magma ocean, Pa.s, Stixrude et al. 2012
nu = mu/rho_s #Maagma ocean kinematic viscosity, m^2/s
k = 3.e-7 #Thermal diffusivity, m^2/s
K = k * rho_s * Cp # thermal conductivity



F = [1.e4,1.e5,1.e6]#Heat flux from magma ocean, W/m^2
Rt = np.arange(1.e5,3.e7,1.e5) #Target radius in m 
Ri = [1e-2,1.e5,1.e6]#Impacor radius in m 
Ri = np.array(Ri)

rho_i = rho_s + (rho_m-rho_s)*fm   #Impactor density
rho_t = rho_s + (rho_m-rho_s)*fm_t #Target density
Mt    = rho_t * 4./3.*np.pi * Rt**3. #Target mass, in kg
Mi    = rho_i * 4./3.*np.pi * Ri**3. #Impactor mass, in kg

sigma = 1. #Surface tension in J/m^2   
U_Ue = 1.  #Impact to sound speed for large impactors

a_RTI  = 1. #Prefactor for drop size scaling for fragmentation by RTI, TO MATCH WITH DATA FROM AUGUSTIN MALLER
a_buoyancy  = 8.1 #Prefactor for drop size scaling for fragmentation by buoyancy, Landeau et al. in prep
a_conv = 10.#Prefactor for drop size scaling for fragmentation by convection
rho_atm = 5. #Atmosphere density on target planet, in kg / m^3

Cd = 0.2 #Drag coefficient, see Samuel 2012
Wec = 10. #Critical Weber number in Lichtenberg et al. 2021

Nbreakup = 4. #Number of overturns to break into drops, Landeau et al. 2014
Label = ['Pebbles','100 km', '1000 km']
Label_conv = [r'$10^4$',r'$10^5$', r'$10^6$']
Color = ['tab:blue','tab:orange','tab:green']


LineW     = 4
FigSize_x = 4.8
FigSize_y = 4.8
Fontsize_labels = 17
Fontsize_labels_small = 13
#----------------------------------
#Functions
#----------------------------------

# Gravity 
def Compute_Gravity(Rc, Rp, rho_m, rho_s, G):
    #Very simple gravity model as a function of planet's size. Can be improved using e.g. model by Labrosse 2015.
    #Rc = core radius
    #Rp = planet radius
    g = 4./3.*np.pi*G*(rho_s*Rp+(rho_m-rho_s)*Rc**3/Rp**2)
    return g 

#Functions for drop size by impact or buoyancy
#----------------------------------
def MeanDropSize(a,L,We) : 
    #Compute mean drop size for frgamentation in a turbulent flow. 
    #This scaling is from Hinze (1955), tested and validated in Landeau et al. (in prep)
    #We = Weber number We = inertia / surface tension
    #L = largest length scale in system
    #a = dimensionless prefactor to obtain from experiments
    d = a * L * We**(-3/5)
    return d

def MixingLayerThickness_Impact(Fr,Ri,rho_m,rho_s) :
    #Compute the thickness h of the mixing layer from scaling by Lherm et al. 2022, JFM
    #Ri = impactor radius
    c7 = 0.04
    c8 = 2.3
    c9 = 0.21
    h = Ri*c7 * (rho_m/rho_s)**c8*Fr**c9#h = mixing layer thickness due to Rayleigh-Taylor instabilities, 
    #scaling from Lherm et al. 2022, JFM 
    #TO DO : UPDATE WITH SCALING FROM AUGUSTIN MALLER
    return h

def Time_Impact(Fr,Ri,U,rho_m,rho_s) :
    #Compute the time tmax to open a crater from scaling by Lherm et al. 2022, JFM
    #Ri = impactor radius
    #U  = impact velocity
    c3 = 0.87
    c4 = -0.53
    c5 = 0.61
    tmax = Ri/U*c3 * (rho_m/rho_s)**(c4+c5)*Fr**c5#h = mixing layer thickness due to Rayleigh-Taylor instabilities, 
    #scaling from Lherm et al. 2022, JFM 
    #TO DO : UPDATE WITH SCALING FROM AUGUSTIN MALLER
    return tmax

def Weber_RTI(sigma,rho_m,rho_s,Fr,Ri,U) :
    #sigma = interfacial tension between metal and silicates, SI units
    #rho=metal density in kg / m^3
    #U = impact velocity in m / s
    #Ri = Impactor radius in m
    #Fr = Froude number = inertia / gravity
    h    = MixingLayerThickness_Impact(Fr,Ri,rho_m,rho_s)#h = mixing layer thickness due to Rayleigh-Taylor instabilities, see Lherm et al. 2022, JFM 
    tcrat= Time_Impact(Fr,Ri,U,rho_m,rho_s) #Time to open a crater, see Lherm et al. 2022, JFM or Bisighini et al. 2010
    We   = (rho_m*h**3.)/(tcrat**2*sigma) # Weber number We = inertia / surface tension
    return We

def Weber_Impact(sigma,rho_m,Ri,U) :
    #sigma = interfacial tension between metal and silicates, SI units
    #rho=metal density in kg / m^3
    #U = impact velocity in m / s
    #Ri = Impactor radius in m
    We   = (rho_m*U**2*Ri)/sigma # Weber number We = inertia / surface tension
    return We

def Weber_Buoyancy(rho_m,rho_s,g,sigma,R) :
    #Weber number for buoyancy speed
    U = ((rho_m-rho_s) / rho_m * g * R)**(0.5) # Typical velocity
    We = rho_m * U**2 * R/sigma
    return We

def Compute_Froude(Rt,Ri,U_Ue,rho_atm,rho_i, Cd) :
    #Froude number Fr = inertia / gravity
    #Ri = impactor radius
    #Rt = target radius
    #U_Ue = impact to sound speed
    if (Ri > 1.e4) :
        Fr = 2.*(1+Rt/Ri)*(U_Ue)**2. #Impactors larger than 10 km = no drag in atmosphere
    elif  (Ri < 1.) : 
        Fr = 2.*(rho_i-rho_atm)/(3*rho_atm*Cd)#Pebbles
        Fr = Fr * np.ones(len(Rt))
    return Fr

def Compute_ImpactSpeed(Ri,Rt,Mi,Mt,U_Ue,rho_atm,rho_i, Cd, g , G) :
    #Froude number Fr = inertia / gravity
    #Ri = impactor radius
    #Rt = target radius
    #U_Ue = impact to sound speed
    if (Ri > 1.e4) :
        Ue = (2.*G*(Mi+Mt)/(Ri+Rt))**0.5#escape speed
        U = U_Ue * Ue #Impactors larger than 10 km = no drag in atmosphere
    elif  (Ri < 1.) : 
        U = (2*(rho_i-rho_atm)*Ri*g/(rho_atm*3.*Cd))**0.5#Pebbles
    return U


def Compute_BreakupDepth(Ric,Nbreakup) :
    #TO DO : IMPLEMENT NEW SCALING FROM MALLER ET AL. 
    #Lb = depth in magma ocean at which the impactor core breaks into drops
    # Condition from Landeau et al. 2014
    #Froude number Fr = inertia / gravity
    #Ri = impactor radius
    #Rt = target radius
    #U_Ue = impact to sound speed
    Lb = Nbreakup* Ric
    return Lb


#Functions for drop size by turbulent convection
#----------------------------------


def HeatFlux_Convection(alpha,g,L,rho_s,Cp,k,K,nu,DT) :
    #Compute convective speed in magma ocean
    # Scaling from Solomatov 2000 (and reference within), as in Lichtenberg 2021 for comparison
    b_conv = 0.089
    Ra = alpha*rho_s*g*DT*L**3/(k*nu)
    F = b_conv * K * DT * Ra**(1./3.)/L
    return F

def Velocity_Convection(alpha,g,L,rho_s,Cp,k,K,nu,F) :
    #Compute convective speed in magma ocean
    # Scaling from Solomatov 2000 (and reference within), as in Lichtenberg 2021 for comparison
    a_conv = 0.6
    U = a_conv * (alpha*g*L*F/(rho_s*Cp))**(1./3.)
    return U



def Weber_Convection(rho_m,sigma,L,Uconv) :
    #Weber number for turbulent convection in magma ocean
    # TO DO : CHECK WHETHER KOLMOGORV CASCADE IS TRUE IN TURBULENT CONVECTION. EFFECT OF ROTATION ? 
    We = rho_m * Uconv**2 * L/sigma
    return We

def DropSize_LargeScale(a,We,L) :
    #Drop size neglecting the turbulent cascade, assuming all length scales have the same energy
    DropSize = a * L * We**(-1)
    return DropSize

def DropSize_Lichtenberg(sigma,rho_m,rho_s,Uconv,Wec) :
    #Drop size used in Lichtenberg et al. 2021, based on Rubie et al. 2003
    DropSize = sigma*Wec/((rho_m-rho_s)*Uconv**2)
    return DropSize

# -------------------------------
# -------------------------------
# Plots
# -------------------------------
# -------------------------------


#CASE 1 : fragmentation and drops generated by impact at top of magma ocean
# -------------------------------


fig, ax = plt.subplots(3,1,figsize=(FigSize_x, FigSize_y*3))

AX = plt.subplot(3,1,1)
plt.xlabel('Magma ocean depth (m)', fontsize=Fontsize_labels)
plt.ylabel('Drop size (m)', fontsize=Fontsize_labels)

for i in range(len(Ri)) : 
    Fr = Compute_Froude(Rt,Ri[i],U_Ue,rho_atm,rho_i, Cd)  # Froude number
    #print('Fr = '+str(Fr))
    Rc  = fm_t**(1./3.)*Rt#Core radius on target, in m 
    Rmantle = Rt-Rc #Mantle depth on target, in m, assuming fully liquid into magma ocean
    g  = Compute_Gravity(Rc, Rt, rho_m, rho_s, G)
    U  = Compute_ImpactSpeed(Ri[i],Rt,Mi[i],Mt,U_Ue,rho_atm,rho_i, Cd, g , G)
    #print('U = ' +str(U))
    We_Impact = Weber_Impact(sigma,rho_m,Ri[i],U)
    #print('We impact = '+ str(We_Impact))
    We        = Weber_RTI(sigma,rho_m,rho_s,Fr,Ri[i],U) #Weber number
    #print('We = '+ str(We))
    hmax = MixingLayerThickness_Impact(Fr,Ri[i],rho_m,rho_s)
    Drop_size = MeanDropSize(a_RTI,hmax,We)
    Drop_size[We_Impact<40] = Ri[i] #No fragmentation if We_impact < 40, Lhuissier et al. 2013
    Drop_size[Fr<10] = Ri[i] #No fragmentation if Fr < 40, Maller et al. to be submitted
 
    Drop_size[Drop_size>Ri[i]] = Ri[i] 
    X = np.where(Fr>10)
    

    plt.loglog(Rmantle[X],Drop_size[X],linewidth=LineW,color=Color[i],label=Label[i])

AX.set_title('Fragmentation by impact',fontsize=Fontsize_labels)
#plt.xlim((1.e-3,1.))
plt.ylim((1.e-4,2.e-2))
plt.legend(fontsize=Fontsize_labels_small, title = 'Impactor size',title_fontsize=Fontsize_labels_small)







#CASE 2 : fragmentation and drops generated by fall of impactor core in magma ocean
# -------------------------------


AX = plt.subplot(3,1,2)
plt.xlabel('Magma ocean depth (m)', fontsize=Fontsize_labels)
plt.ylabel('Drop size (m)', fontsize=Fontsize_labels)

for i in range(len(Ri)) : 
    Fr = Compute_Froude(Rt,Ri[i],U_Ue,rho_atm,rho_i, Cd)  # Froude number
    #print('Fr = '+str(Fr))
    Rc  = fm_t**(1./3.)*Rt#Core radius on target, in m 
    Rmantle = Rt-Rc #Mantle depth on target, in m, assuming fully liquid into magma ocean
    Ric = fm  **(1./3.)*Ri[i]#Impactor core radius, in m 
    g  = Compute_Gravity(Rc, Rt, rho_m, rho_s, G)
    U  = Compute_ImpactSpeed(Ri[i],Rt,Mi[i],Mt,U_Ue,rho_atm,rho_i, Cd, g , G)
    #print('U = ' +str(U))
    We_Impact = Weber_Impact(sigma,rho_m,Ri[i],U)
    #print('We impact = '+ str(We_Impact))
    We        = Weber_RTI(sigma,rho_m,rho_s,Fr,Ri[i],U) #Weber number
    #print('We = '+ str(We))
    hmax = MixingLayerThickness_Impact(Fr,Ri[i],rho_m,rho_s)
    Drop_size_impact = MeanDropSize(a_RTI,hmax,We)
    Drop_size_impact[We_Impact<40] = Ri[i] #No fragmentation if We_impact < 40, Lhuissier et al. 2013
    Drop_size_impact[Fr<10] = Ri[i] #No fragmentation if Fr < 40, Maller et al. to be submitted
 
    Drop_size_impact[Drop_size_impact>Ri[i]] = Ri[i] 
    
    Lb = Compute_BreakupDepth(Ric,Nbreakup)
    We_buoyancy = Weber_Buoyancy(rho_m,rho_s,g,sigma,Ric)
    #print('We buoyancy = ' + str(We_buoyancy))
    Drop_size_buoyancy = MeanDropSize(a_buoyancy,Ric,We_buoyancy)
    Drop_size_buoyancy[We_buoyancy<5] = Ri[i] #No fragmentation if We < 5, Landeau et al. 2014

    
    Drop_size = Drop_size_impact
    X = np.where(Rmantle>Lb)
    X1 = np.where(Drop_size_buoyancy[X]<Drop_size_impact[X])
    Drop_size[X][X1] = Drop_size_buoyancy[X][X1]
    Drop_size_buoyancy[Drop_size_buoyancy>Ri[i]] = Ri[i] 


    plt.loglog(Rmantle[X],Drop_size_buoyancy[X],linewidth=LineW,color=Color[i],label=Label[i])
    
AX.set_title('Fragmentation during metal fall',fontsize=Fontsize_labels)

#AX.set_title(Title[1],fontsize=Fontsize_labels)
#plt.xlim((1.e-3,1.))
plt.ylim((1.e-4,2.e-2))
plt.legend(fontsize=Fontsize_labels_small, title = 'Impactor size',title_fontsize=Fontsize_labels_small)





#CASE 3 : fragmentation byturbulent convection in magma ocean
# -------------------------------


AX = plt.subplot(3,1,3)
plt.xlabel('Magma ocean depth (m)', fontsize=Fontsize_labels)
plt.ylabel('Drop size (m)', fontsize=Fontsize_labels)

for i in range(len(F)) : 
    
    # Fragmentation by turbulent convection
    # -------------------------------
    Rc  = fm_t**(1./3.)*Rt#Core radius on target, in m 
    Rmantle = Rt-Rc #Mantle depth on target, in m, assuming fully liquid into magma ocean
    Uconv = Velocity_Convection(alpha,g,Rmantle,rho_s,Cp,k,K,nu,F[i]) 
    Drop_size_Lichtenberg = DropSize_Lichtenberg(sigma,rho_m,rho_s,Uconv,Wec)
    We_conv = Weber_Convection(rho_m,sigma,Rmantle,Uconv)
    print('We_conv = '+str(We_conv))
    Drop_size_conv = MeanDropSize(a_conv,Rmantle,We_conv)
    Drop_size_conv_noturbulence = DropSize_LargeScale(a_conv,We_conv,Rmantle)
    
    plt.loglog(Rmantle,Drop_size_conv,linewidth=LineW,color=Color[i],label=Label_conv[i])
    plt.loglog(Rmantle,Drop_size_Lichtenberg,'--',linewidth=LineW,color=Color[i])
    #plt.loglog(Rmantle,Drop_size_conv_noturbulence,':',linewidth=LineW,color=Color[i])
    plt.text(1.e6, 0.4, r'Turbulent convection',fontsize=Fontsize_labels_small,rotation=-4)
    plt.text(4.e4, 0.0003, r'Lichtenberg et al. 2021',fontsize=Fontsize_labels_small,rotation=-12)
    
AX.set_title('Fragmentation by convection',fontsize=Fontsize_labels)

#AX.set_title(Title[1],fontsize=Fontsize_labels)
#plt.xlim((1.e-3,1.))
plt.ylim((1.e-7,1.e3))
plt.legend(loc=3,fontsize=Fontsize_labels_small, title = r'Heat flux (W/m$^{-2}$)',title_fontsize=Fontsize_labels_small,mode = "expand", ncol = 3)





plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.7)

ttt = ['a', 'b', 'c']
axes = fig.get_axes()
for p,l in zip(axes, ttt):
    p.annotate(l, xy=(-0., 1.04), xycoords="axes fraction", fontsize=Fontsize_labels, weight = 'bold')



plt.savefig(figure_name, bbox_inches = 'tight' )
plt.show()

