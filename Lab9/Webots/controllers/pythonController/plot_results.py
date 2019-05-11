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
    plt.title("Joint angles for drive= {}".format(d))
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

    
def main(plot=True):
    """Main"""
    # Load data
    phase_lag=np.linspace(0,4*math.pi,15)
    amplitude = np.linspace(0.3,0.5,15)
    num_simulation=len(phase_lag)*len(amplitude)
    vx=np.zeros(num_simulation)
    energy=np.zeros_like(vx)
    for i in range(num_simulation):
        with np.load('logs/example/simulation_{}.npz'.format(i)) as data:
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
        
        power=np.abs(joints_data[:,:,2]*joints_data[:,:,4])
        energy[i]=np.sum(np.trapz(power,times,axis=0))
        
#        plt.figure("speed sim {}".format(i))
#        plt.plot(times[1000:-1],vx_v,label="Vx")
#        plt.plot(times[1000:-1],vy_v,label="Vy")
#        plt.plot(times[1000:-1],vz_v,label="Vz")
#        plt.legend()
    print(vx)
    print(energy)
    solution=np.zeros((num_simulation,3))
    for i in range(len(phase_lag)):
        for j in range(len(amplitude)): 
            solution[i*j,0]=phase_lag[i]
            solution[i*j,1]=amplitude[j]
    solution[:,2]=vx
    plt.figure("grid speed ")
    plot_2d(solution, ["Phase_lag","Amplitude","speed"], n_data=num_simulation, log=False)
    
    solution[:,2]=energy
    plt.figure("grid energy ")
    plot_2d(solution, ["Phase_lag","Amplitude","energy"], n_data=num_simulation, log=False) 
    
    solution[:,2]=vx/energy
    plt.figure("grid speed/energy ")
    plot_2d(solution, ["Phase_lag","Amplitude","speed/speed"], n_data=num_simulation, log=False)  
    """
    print(np.shape(amplitude))
    print(np.shape(phase_lag))
    print(np.shape(link_data))
    print(np.shape(joints_data))

    # Plot data
    plt.figure("Positions")
    plot_pos_xyz(times, link_data)
    """
    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

