"""Plot results"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmc_robot import ExperimentLogger
from save_figures import save_figures
from parse_args import save_plots
import math

def plot_positions(times, link_data,labels,d):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data,label=labels[i])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Joint Angle [rad]")
    plt.title("Joint Angles for {}".format(d))
    plt.grid(True)
    
def plot_pos_xyz(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data,label=["x","y","z"][i])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Position")
    plt.grid(True)    


def plot_trajectory(link_data):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 2])
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.grid(True)


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear'  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], "r.")
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation="none",
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])
    
def grd_srch(da,db,ex,na,nb,run_plt,grd_strt=0):
    num_simulation=len(da)*len(db)
    vx=np.zeros(num_simulation)
    energy=np.zeros_like(vx)
    meanv=np.zeros_like(vx)
    for i in range(num_simulation):
        with np.load('../../../{}/simulation_{}.npz'.format(ex,i)) as data:
            timestep = float(data["timestep"])
#            amplitude = data["amplitude"]
#            phase_lag = data["phase_lag"]
            link_data = data["links"][:, 0, :]
            joints_data = data["joints"]
        times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
        x=link_data[1000:,0]
        y=link_data[1000:,1]
        z=link_data[1000:,2] 
        vx_v=np.diff(x)/np.diff(times[1000:])
        vy_v=np.diff(y)/np.diff(times[1000:])
        vz_v=np.diff(z)/np.diff(times[1000:])
        vx[i]=np.mean(vx_v)
        vy=np.mean(vy_v)
        vz=np.mean(vz_v)
        meanv[i]=abs(vx[i]+vy+vz)/3
        power=np.abs(joints_data[:,:,1]*joints_data[:,:,3])
        energy[i]=np.sum(np.trapz(power,times,axis=0))
        if i==run_plt :#Plotting a given run 
            plt.figure()
            plot_pos_xyz(times,link_data)
            plt.xlabel("Time [s]")
            plt.ylabel("Position [m]")
            plt.legend(["X","Y","Z"])
            plt.title("Position evolution for swimming")
            plt.figure()
            plot_trajectory(link_data)
            plt.title("Trajectory plot for {} = {:.3f} and {} = {}".format(na,db[i//len(da)],nb,db[i%len(db)]))
        #derivatives
#        plt.figure("speed sim {}".format(i))
#        plt.plot(times[1000:-1],vx_v,label="Vx")
#        plt.plot(times[1000:-1],vy_v,label="Vy")
#        plt.plot(times[1000:-1],vz_v,label="Vz")
#        plt.legend()
    solution=np.zeros((num_simulation,3))
    c=0
    for i in range(len(da)):
        for j in range(len(db)):
            solution[c,0]=da[i]
            solution[c,1]=db[j]
            c=c+1
    solution[:,2]=meanv
    plt.figure()
    plot_2d(solution[grd_strt:], ["{}".format(na),"{}".format(nb),"Speed"], n_data=num_simulation-grd_strt, log=False)
    plt.title("Grid Search for Speed")
    
    solution[:,2]=energy
    plt.figure()
    plot_2d(solution[grd_strt:,:], ["{}".format(na),"{}".format(nb),"Energy"], n_data=num_simulation-grd_strt, log=False) 
    plt.title("Grid Search for Energy")
    
    solution[:,2]=meanv/energy
    plt.figure()
    plot_2d(solution[grd_strt:,:], ["{}".format(na),"{}".format(nb),"Speed/Energy"], n_data=num_simulation-grd_strt, log=False)  
    plt.title("Grid Search for Speed/Energy")

def main(plot=True):
    """Main"""
    plt.close("all")
    # Load data
    phase_lag=np.linspace(math.pi*3/2,4*math.pi,10)
    amplitude = np.linspace(0.3,0.5,10)
    a_head=np.linspace(0.3,0.5,10)
    a_tail=np.linspace(0.3,0.5,10)
    
    
    amps=np.linspace(0,0.6,50)

    #DO GRID SEARCH FOR PARAMETERS FOR 9B AND 9C
    grd_srch(phase_lag,amplitude,"9b","Phase Lag","Amplitude",20)
    grd_srch(a_head,a_tail,"9c","Head Amplitude","Tail Amplitude",20)
    
    #PLOT TURNING MOTION FOR 9D1
    with np.load('../../../9d1/simulation_{}.npz'.format(0)) as data:
        timestep = float(data["timestep"])
    #            amplitude = data["amplitude"]
    #            phase_lag = data["phase_lag"]
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]
    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
    plt.figure()
    plot_pos_xyz(times,link_data)
    plt.title("XYZ positions for turning motion")
    plt.figure()
    plot_trajectory(link_data)
    plt.title("2D Trajectory for turning motion")
    plt.figure()
    plot_positions(times,joints_data[:,:10,0],["q1","q2","q3","q4","q5","q6","q7","q8","q9","q10"],"Turning")
    plt.title("Spine joint angles for turning motion")
    
    #PLOT BACKWARD MOTION FOR 9D12
    with np.load('../../../9d2/simulation_{}.npz'.format(0)) as data:
        timestep = float(data["timestep"])
    #            amplitude = data["amplitude"]
    #            phase_lag = data["phase_lag"]
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]
    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
    plt.figure()
    plot_pos_xyz(times,link_data)
    plt.title("XYZ positions for backward motion")
    plt.figure()
    plot_trajectory(link_data)
    plt.title("2D Trajectory for backward motion")
    plt.figure()
    plot_positions(times[:3000],joints_data[:3000,:10,0],["q1","q2","q3","q4","q5","q6","q7","q8","q9","q10"],"Turning")
    plt.title("Spine joint angles for backward motion")
    
    
    #PLOTTING THE MOTION OF THE WALKING ROBOT 
    with np.load('../../../9f1/simulation_{}.npz'.format(0)) as data:
        timestep = float(data["timestep"])
    #            amplitude = data["amplitude"]
    #            phase_lag = data["phase_lag"]
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]
    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
    plt.figure()
    link_data[:,0]=link_data[:,0]+10
    plot_pos_xyz(times,link_data)
    plt.title("XYZ positions for walking motion")
    plt.figure()
    plot_trajectory(link_data)
    plt.title("2D Trajectory for walking motion")
    plt.figure()
    plot_positions(times,joints_data[:,:10,0],["q1","q2","q3","q4","q5","q6","q7","q8","q9","q10","l1","l2","l3","l4"],4)
    plt.title("Spine Joint Angles for walking motion")
    plt.figure()
    plot_positions(times,joints_data[:,10:,0],["l1","l2","l3","l4"],4)
    plt.title("Limb joint Angles for walking motion")
    #OPTIMAL Leg_body OFFSET SEARCH
    body_legs=np.linspace(0,math.pi*2,50)
    meanv=np.zeros(len(body_legs))
    for i in range(len(body_legs)):
        with np.load('../../../9f/simulation_{}.npz'.format(i)) as data:
#            timestep = float(data["timestep"])
        #            amplitude = data["amplitude"]
        #            phase_lag = data["phase_lag"]
            link_data = data["links"][:, 0, :]
            joints_data = data["joints"]
        times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
        x=link_data[1000:,0]
        y=link_data[1000:,1]
        z=link_data[1000:,2] 
        vx_v=np.diff(x)/np.diff(times[1000:])
        vy_v=np.diff(y)/np.diff(times[1000:])
        vz_v=np.diff(z)/np.diff(times[1000:])
        vx=np.mean(vx_v)
        vy=np.mean(vy_v)
        vz=np.mean(vz_v)
        meanv[i]=abs(vx+vy+vz)/3
        
    plt.figure("Influence of limb to body phase offset on speed")
    plt.scatter(body_legs,meanv)
    plt.plot(body_legs[22],meanv[22],"ro")
    plt.xlabel("Limb-Body Offset [rad]")
    plt.ylabel("Speed [m/s]")
    plt.title("Influence of limb to body phase offset on speed")
    #OPTIMAL Amplitude SEARCH
    amps=np.linspace(0,0.6,50)
    meanv=np.zeros(len(amps))
    for i in range(len(amps)):
        with np.load('../../../9f2/simulation_{}.npz'.format(i)) as data:
#            timestep = float(data["timestep"])
        #            amplitude = data["amplitude"]
        #            phase_lag = data["phase_lag"]
            link_data = data["links"][:, 0, :]
            joints_data = data["joints"]
        times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
        x=link_data[1000:,0]
        y=link_data[1000:,1]
        z=link_data[1000:,2] 
        vx_v=np.diff(x)/np.diff(times[1000:])
        vy_v=np.diff(y)/np.diff(times[1000:])
        vz_v=np.diff(z)/np.diff(times[1000:])
        vx=np.mean(vx_v)
        vy=np.mean(vy_v)
        vz=np.mean(vz_v)
        meanv[i]=abs(vx+vy+vz)/3
        
    plt.figure("Influence of oscillator amplitude on walking speed")
    plt.scatter(amps,meanv)
    plt.plot(amps[29],meanv[29],"ro")
    plt.xlabel("Joint Amplitudes [rad]")
    plt.ylabel("Speed [m/s]")
    plt.title("Influence of oscillator amplitude on walking speed")
    
    
    #PLOT X POSITION AND SPINE/LIMB ANGLES
    with np.load('../../../9g/simulation_{}.npz'.format(0)) as data:
        timestep = float(data["timestep"])
    #            amplitude = data["amplitude"]
    #            phase_lag = data["phase_lag"]
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]
    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
    plt.figure()
    plot_pos_xyz(times,link_data)
    plt.title("XYZ positions for land to water transition")
    plt.plot(times,np.ones_like(times)*0.2)
    plt.figure()
    plot_trajectory(link_data)
    plt.title("2D Trajectory for land to water transition")
    plt.figure()
    plot_positions(times[:4000],joints_data[:4000,:10,0],["q1","q2","q3","q4","q5","q6","q7","q8","q9","q10"],"Turning")
    plt.title("Spine joint angles for land to water transitions")
    plt.figure()
    plot_positions(times,joints_data[:,10:,0],["l1","l2","l3","l4"],4)
    plt.title("Limb joint Angles for land to water transition")
    
    #WATER TO LAND TRANSITION
    with np.load('../../../9g1/simulation_{}.npz'.format(0)) as data:
        timestep = float(data["timestep"])
    #            amplitude = data["amplitude"]
    #            phase_lag = data["phase_lag"]
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]
    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
    plt.figure()
    plot_pos_xyz(times,link_data)
    plt.title("XYZ positions for water to land transition")
    plt.plot(times,np.ones_like(times)*0.2)
    plt.figure()
    plot_trajectory(link_data)
    plt.title("2D Trajectory for water to land transition")
    plt.figure()
    plot_positions(times[:4000],joints_data[:4000,:10,0],["q1","q2","q3","q4","q5","q6","q7","q8","q9","q10"],"Turning")
    plt.title("Spine joint angles water to land transitions")
    plt.figure()
    plot_positions(times,joints_data[:,10:,0],["l1","l2","l3","l4"],4)
    plt.title("Limb joint Angles for water to land transition")



    # Show plots
    if plot:
        plt.show()
    else:
        save_figures() 



if __name__ == '__main__':
    main(plot=not save_plots())

