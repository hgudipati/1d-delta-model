"""
The following codes reproduce 1D delta model 
from Chapter 34 in Gary's Ebook, originally coded in VBA

Tian Dong
 
10/25/2020

future work:
    
    *code style follows MATLAB a bit, future optimization needed
    *mass balance is ~6%, a bit high, probably mainly due to the calculation 
    of mass balance rather than acutal model implementation 
    *add grain-size specific Exner equation [Naito et al., 2019]
    *extend to true 1.5 D, width based backwater [Moodie et al., 2019] or just add a width for mass balance
    *add sea level change funciton
    *add hydrograph function 

"""
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from IPython import get_ipython


#clear variables and close all plots
get_ipython().magic('reset -sf')
plt.close()

#set up input parameters
g = 9.81 
R = 1.65
Qbf = 1800          #water discharge in [m^3/s]
Bbf = 300           #channel width in [m]
qw = Qbf/Bbf        #discharge per width in [m^2/s]
Cf = 0.0044         #friction coef.
D = 0.0005          #bed sediment size in [m]
qtfeed = 6e-4       #sediment feed rate per width [m^2/s]
If = 60/365         #intermittency 
lamp = 0.4          #porosity

#sediment transport parameters
alt = 7.2   #coef.
nt = 2.5    #expo.
tau_c = 0   #dimenisonless critical shear stress

#set up model space
L_f0 = 50000                    #fluvial reach in [m]
M = 100                         #no. of spatial nodes
dx = L_f0/M
x_ft = np.arange(0, L_f0+1, dx) #x coordinate with dimen.
x_f = x_ft.reshape(M+1, 1)      #prelocate 

#set up moving boundary coordinate
dxhat = 1/M
xhat_ft = np.arange(0, 1+dxhat, dxhat)
xhat_f = xhat_ft.reshape(M+1, 1)

#set up time loop
t_span = 10                         #total model run in years
timeyr = 31557600                   #scaler to sceonds
dthat = 0.03                        #time step
dt = dthat * timeyr                 #convert to realtime
tstep = np.int(np.round(t_span*timeyr/dt)) #total no. of time step

#plotting parameters 
ntstep = 10     #temporal output spacing in yearrs
nxstep = M+1    #spatial output spacing
xscalor = 1.5

#initial fluvial bed elevation
S_f0 = 2.5e-4                           #initial slope
eta_sl0 = 3                             #initial elevation at the topset_foreset break, i.e., shoreline [m]
eta_f0 = eta_sl0 + S_f0 * (L_f0 - x_f)  #initial bed elevation in [m] 

#initial antecedent bed profile
eta_b0 = 0                                          #initial elevation at the bottom set
S_a = 0.2                                           #forset slope
x_ante_end = L_f0 + (eta_sl0 - eta_b0)/S_a          #initial antecedent x coord
x_ante = np.vstack((x_f, x_ante_end))
S_base = 0                                          #initial basement slope 
eta_ante = eta_b0 + S_base * (x_ante_end - x_ante)  #initial antecedent bed elevation in [m]

#initial bed (fluvial + bottom set elevation)
eta0 = np.vstack((eta_f0, eta_b0)) 
x0 = x_ante

#initial water surface elevation [m]
zeta = 8.5

#set functions
#backwater equation 
def backwater(qw, L_f, dxhat, Cf, g, H, S):
    """
    computer flow depth using backwater equation
    
    input parameters:
    qw - water discharge per unit width 
    L_f - fluvial reach length
    dxhat - dimensionless dx
    Cf - friction
    H - 
    """    
    fr2p = qw**2 / ( g * H**3 )
    dhxp = L_f * ( S - Cf * fr2p ) / ( 1 - fr2p )
    Hpred = H - dhxp * dxhat
    fr2 =  qw**2 / ( g * Hpred**3 )
    dhdx = L_f * ( S - Cf * fr2 ) / ( 1 - fr2 )
    Hi = H - 0.5 * ( dhxp + dhdx ) * dxhat 
    return Hi

#normal flow
def normalflow(qtf, qw, R, g, D, Cf, alt, nt, tau_c):
    Cz = Cf**-0.5
    tsn = (qtf / ((R * g * D) **0.5 * D * alt))**(1 / nt) + tau_c
    Sn = Cz * (R * D * tsn) ** 1.5 * g**0.5 / qw
    Hn = R * D * tsn / Sn
    return Hn

#sediment transport
def sedi_transport(qw, Cf, R, g, D, H, alt, tau_c, nt):
    tau_star = ( Cf * qw **2 / H **2  ) / ( R * g * D )
    qt_star = alt * ( tau_star - tau_c ) ** nt
    qt = np.sqrt( R * g * D ) * D * qt_star;
    return qt
    
#prelocate bunch parameters 
H0= np.zeros((M+1, 1))
H0[-1] = zeta - eta_sl0
H = np.zeros((M+1, 1))

detadt = np.zeros((tstep, 1))   #bed elevation change per time

sdot_s = np.zeros((tstep, 1))   #forset migration rate
sdot_b = np.zeros((tstep, 1))   #bottomset_migration rate

eta_sl = np.zeros((tstep, 1))   #forset elevation
eta_bott = np.zeros((tstep, 1)) #bottomset elevation

x_top = np.zeros((tstep, 1))    #forset x coord
x_bott = np.zeros((tstep, 1))   #bottomset x corrd

deta = np.zeros((M+1, tstep))   #bed elevation change
etaf = np.zeros((M+1, tstep))   #fbed elevation change_fluvial

Hact = np.zeros((M+1, tstep))   #water surface elevtion (for plotting)


#store output
x_all = np.zeros((M+2, tstep))      #x coord
eta_all = np.zeros((M+2, tstep))    #bed elevation
H_all = np.zeros((M+2, tstep))      #flow depth

cnt = 0 #counter

#MODE RUN starts here!
for j in np.arange(0, tstep):  
    
    #flow calculation
    #normal flow
    Hn = normalflow(qtfeed, qw, R, g, D, Cf, alt, nt, tau_c)
    
    if j == 0: # at time 0 i.e. initial time
       H = H0
       etaf[:, j] = eta_f0.reshape(M+1, )
       x_top[j] = L_f0 #fluvial length
       for i in np.arange(M, 0, -1):    
           S = - (etaf[i, j] - etaf[i-1, j]  ) / ( x_top[j] * dxhat )          
           H[i-1] = backwater(qw, x_top[j], dxhat, Cf, g, H[i], S)
           
           #check normal flow depth (optional)
           # if H[i-1] <= Hn:
           #     H[i-1] = Hn
             
    else: 
        for i in np.arange(M, 0, -1):
            if i == M:
                H[i] = zeta - etaf[-1, j-1]
            S = - (etaf[i, j-1] - etaf[i-1, j-1]  ) / ( x_top[j-1] * dxhat )           
            H[i-1] = backwater(qw, x_top[j-1], dxhat, Cf, g, H[i], S)
            
            #check normal flow depth (optional)
            # if H[i-1] <= Hn:
            #    H[i-1] = Hn
    
    #change flow depth to water surface elevation  
    Htemp = H + etaf[:, j-1].reshape(M+1, 1)         
    Hact[:, j] = Htemp.reshape(M+1, )  
        
    #sediment transport    
    qt_f = sedi_transport(qw, Cf, R, g, D, H, alt, tau_c, nt)
    qt = np.vstack([qtfeed, qt_f])
    if j == 0: # at time 0 i.e. initial time
        dqtdx = np.diff(qt, axis=0) / dxhat
    else:
        dqtdx = np.diff(qt, axis=0) / dxhat
        
    #bed elevation 
    if j == 0: # at time 0 i.e. initial time
        detadx_f = np.diff(etaf[:, j], axis=0) / dxhat
        detadx_end = ( etaf[-1, j] -  etaf[-2, j]) / dxhat
        detadx = np.vstack([detadx_f.reshape(M, 1), detadx_end])        
    else:
        detadx_f = np.diff(etaf[:, j], axis=0) / dxhat
        detadx_end = ( etaf[-1, j] -  etaf[-2, j]) / dxhat
        detadx = np.vstack([detadx_f.reshape(M, 1), detadx_end])      
        
    #sediment balance for delta forsect
    if j == 0: # at time 0 i.e. initial time
        detadt[j] = 0               #bed elevation change per time
        sdot_s[j] = 0               #topset migration speed
        sdot_b[j] = 0               #bottomset migration speed
        deta[j] = 0                 #bed elevation
        eta_sl[j] = etaf[-1, j]     #topset elevation
        x_bott[j]  = x_ante[-1]     #bottomset x coord
        eta_bott[j] = eta_b0        #bottomset elevation
    else:    
        #bed elevation change at the forset
        eta_sl[j] = etaf[-1, j-1]
        detadt[j] = ( eta_sl[j] - eta_sl[j-1] ) / dt
        #topset migration speed
        sdot_s[j] = (1 / S_a) * ( If * qt[-1] / ( ( 1 - lamp ) * ( x_bott[j-1] - x_top[j-1] ) ) - detadt[j] )       
        #bottomset: migration speed
        sdot_b[j] = 1 / ( S_a -  S_base ) * ( S_a * sdot_s[j] + detadt[j] )
        #fluvial sediment balance [Exner]
        deta_temp =  dt * ( - If * dqtdx / ( x_top[j-1] * (1 - lamp) ) + xhat_f * detadx * sdot_s[j] / x_top[j-1] )
        deta[:, j] = deta_temp.reshape(M+1, )        
        
            
    #Updates: topset and foreset elevation and location, fluvial bed elevation
    if j != 0:
        #forset location
        x_top[j] = x_top[j-1] + sdot_s[j] * dt
        #bottom set locaiton
        x_bott[j] = x_bott[j-1] + sdot_b[j] * dt
        #new bed elevation
        etaf[:, j] = etaf[:, j-1] + deta[:, j]
        #bottom set elevtion
        eta_bott[j] = eta_bott[j-1] - S_base * sdot_b[j] * dt  
            
    #store data
    x_r = x_top[j] * xhat_f
    x_all[:, j] = np.vstack([x_r, x_bott[j]]).reshape(M+2,)
    eta_all[:, j] = np.vstack([etaf[:, j].reshape(M+1, 1), eta_bott[j]]).reshape(M+2,)
    H_all[:, j] = np.vstack([Hact[:, j].reshape(M+1, 1), zeta]).reshape(M+2,)    

        
#mass balance
#fluvial mass
abase = eta_ante[0] + S_base * (L_f0  + (eta_f0[0] - x_ante[-1]) / S_a)
etabase = abase - S_base * xhat_f * x_top[j]
etad = etaf[:, -1] - etabase.reshape(M+1, )
etad0 = etaf[:, 0] - etabase.reshape(M+1, )   

v_fin = np.zeros((M+1, 1))
v_ini = np.zeros((M+1, 1))
for i in np.arange(1, M+1):    
    v_fin[i] =  0.5*(etad[i] + etad[i-1])*dxhat*x_top[-1]   
    v_ini[i] =  0.5*(etad0[i] + etad0[i-1])*dxhat*x_top[0]  
V_fin = np.sum(v_fin) + 0.5*etad[-1]*(x_bott[-1] - x_top[-1])
V_ini = np.sum(v_ini) + 0.5*etad0[-1]*(x_bott[0] - x_top[0])
dV = ( V_fin - V_ini ) * lamp
V_in = qtfeed * dt * tstep * If
err = ( np.abs( dV - V_in ) / V_in ) * 100
        
    
#spatial plot resample
pltrge_temp_x = np.linspace(0, M+1, nxstep)
pltrge_x =  [np.int(x) for x in pltrge_temp_x]
#temporal plot resample
pltrge_temp = np.linspace(0, tstep-1, ntstep)
pltrge =  [np.int(x) for x in pltrge_temp]
#set time legend labeling
plttime = np.array(pltrge)*dt/timeyr
#color map
viridis = cm.get_cmap('viridis', len(pltrge))
cmp = viridis(np.arange(len(pltrge)))


#antecendent for plotting
xante_m = np.linspace(0, x_bott[-1]*1.2, 10)
eta_ante_m = - S_base * xante_m + eta_f0[0]
eta_ante_m = - S_base * xante_m + 0

#set plot size
fig, axs = plt.subplots(1, 1, figsize=(12, 6))
#final water surface elevation
plt.plot(x_all[pltrge_x, -1], H_all[pltrge_x, -1], color = 'b', linestyle='solid', linewidth=1, label = 'final ws')
#plot antecendent 
plt.plot(xante_m, eta_ante_m, color = 'k', linestyle='solid', linewidth=2, label ='ante. $\eta$')
#counter
ct = 0    
for k in pltrge: 
    plt.plot(x_all[pltrge_x, k], eta_all[pltrge_x, k], color = cmp[ct, :], 
                 linestyle='solid', linewidth=1, label = 'year ' + format(plttime[ct], '.2f'))        
    ct = ct + 1 

plt.ylabel(r"$\eta$ [m]")
plt.xlabel("x [m]") 
plt.legend(loc = 'best')



