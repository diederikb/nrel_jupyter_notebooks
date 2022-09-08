#postprocessing.py>

import os.path
import re
import numpy as np
import yt
import glob
import pandas as pd
import math

color_list = [
    "#EE2E2F",
    "#008C48",
    "#185AA9",
    "#F47D23",
    "#662C91",
    "#A21D21",
    "#B43894",
    "#010202",
]
linestyle_list = [
    "solid",
    "dashed",
    "dotted",
    "dashdot"
]

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)

def load_plt(plt_dir,i):
    """
    Load the `i`-th plt file in `case_dir` into a yt dataset and return it.
    """
    plt_files = natural_sort(glob.glob(os.path.join(plt_dir, "plt*")))
    ds = yt.load(plt_files[i])
    return ds

def batch_load_plt(root_dir,case_dir,methods,grid_sizes,i,grid_prefix="nx_",**kwargs):
    """
    Load the `i`-th plt file for every method in `methods` and every grid size in `grid_sizes`. The directories should be organized as `root_dir`/method/`case_dir`/`grid_prefix`grid_size/plt...
    """
    ds_list = []
    for method in methods:
        ds_method = []
        for grid_size in grid_sizes:
            full_dir = os.path.join(root_dir,method,case_dir,f'{grid_prefix}{grid_size}')
            ds_method.append(load_plt(full_dir,i))
        ds_list.append(ds_method)
    return ds_list

def load_error(filepath,skiplines=0,nvars=1,loadvars=[0],withglobal=True,**kwargs):
    """
    Load the error file at `filepath` into a pandas dataframe. The first column of the file is assumed to be the time variable and the other columns are assumed to be organized first per level and then per variable the error is calculated for.
    The number of vars `nvars` in the file has to be specified such that the number of levels can be calculated. Per level, only the colunns in `loadvars` are loaded. If there is a global error, it is assumed to be specified in the last columns, with one column per variable, and it only is loaded if `withglobal` is `True`.
    """
    with open(filepath) as f:
        lines = f.readlines()
        time = [float(line.split()[0]) for line in lines[skiplines:]]
        df = pd.DataFrame({"time": time})
        ncolumns = len(lines[1].split())-1
        if withglobal:
            nlevels = int(ncolumns/nvars) - 1
        else:
            nlevels = int(ncolumns/nvars)
        active_column = 1
        for level in range(nlevels):
            for var in range(nvars):
                if var in loadvars:
                    var_error = [float(line.split()[active_column]) for line in lines[skiplines:]]
                    df[f"error_var_{var}_level_{level}"] = var_error
                active_column += 1
        if withglobal:
            for var in range(nvars):
                if var in loadvars:
                    var_error = [float(line.split()[active_column]) for line in lines[skiplines:]]
                    df[f"error_var_{var}_global"] = var_error
                active_column += 1
    return df

def batch_load_error(filename,root_dir,case_dir,methods,grid_sizes,grid_prefix="nx_",skiplines=0,nvars=1,loadvars=[0],withglobal=True,**kwargs):
    """
    Load the error file for every method in `methods` and every grid size in `grid_sizes`. The directories should be organized as `root_dir`/method/`case_dir`/`grid_prefix`grid_size/plt...
    """
    error_df_list = []
    for method in methods:
        error_method = []
        for grid_size in grid_sizes:
            filepath = os.path.join(root_dir,method,case_dir,f'{grid_prefix}{grid_size}',filename)
            df = load_error(filepath,skiplines=skiplines,nvars=nvars,loadvars=loadvars,withglobal=withglobal)
            error_method.append(df) 
        error_df_list.append(error_method) 
    return error_df_list

def sample_ray(ds,ray_start,ray_end,variable="temperature",**kwargs):
    """
    Sample a `yt` dataset on a ray with start coordinate `ray_start` and end coordinate `ray_end` and return the data in a pandas dataframe.
    """
    ray = ds.ray(ray_start,ray_end)
    srt = np.argsort(ray[("gas","x")])
    df = pd.DataFrame({f: np.array(ray[("gas",f)][srt]) for f in ["x","y","z",variable]})
    t = ds.current_time.value.flatten()[0]
    df["time"] = t
    df["dx"] = [ray.fwidth[i][0].value.flatten()[0] for i in range(len(df["x"].values))]
    df["dy"] = [ray.fwidth[i][1].value.flatten()[0] for i in range(len(df["y"].values))]
    df["dz"] = [ray.fwidth[i][2].value.flatten()[0] for i in range(len(df["z"].values))]
    df["ray_coord"] = ((df["x"]-df["x"][0])**2 + (df["y"]-df["y"][0])**2 + (df["z"]-df["z"][0])**2)**0.5    
    return df

def setup_batch_line_plot(ax,methods,grid_sizes,xlim,ylim,xlabel,ylabel,errorlines=False,maxlevel=0,withglobal=False,linestyles=linestyle_list,colors=color_list,linewidth=1,colormethods=True,label_grid_appendix=" cells",**kwargs):
    """

    """
    lines = []
    for i_method in range(len(methods)):
        for i_grid_size in range(len(grid_sizes)):
            if colormethods:
                color = colors[i_method]
                linestyle = linestyles[i_grid_size]
            else:
                color = colors[i_grid_size]
                linestyle = linestyles[i_method]
            if errorlines:
                for level in range(maxlevel+1):
                    label = f'{methods[i_method]}; {grid_sizes[i_grid_size]}{label_grid_appendix}; level {level}'
                    line = ax.plot([], [], color=color, linewidth=linewidth, linestyle=linestyle, label=label)[0]
                    lines.append(line)
                if withglobal:
                    label = f'{methods[i_method]}; {grid_sizes[i_grid_size]}{label_grid_appendix}; global level'
                    line = ax.plot([], [], color=color, linewidth=linewidth, linestyle=linestyle, label=label)[0]
                    lines.append(line)
            else:
                label = f'{methods[i_method]}; {grid_sizes[i_grid_size]}{label_grid_appendix}'
                line = ax.plot([], [], color=color, linewidth=linewidth, linestyle=linestyle, label=label)[0]
                lines.append(line)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return lines

def setup_slice_plot(fig,ax,ds,fieldtype,variable,zlim,logscale=False,size=10,buff_size=[1000,1000],fontsize=12):
    slp = yt.SlicePlot(ds, "z", (fieldtype,variable), fontsize=fontsize, origin='native', buff_size=buff_size)
    slp.set_log((fieldtype,variable), logscale)
    slp.set_xlabel('x')
    slp.set_ylabel('y')
    slp.set_zlim((fieldtype,variable), zlim[0], zlim[1])
    slp.hide_colorbar()
    slp.set_figure_size(size)
    slp.plots[(fieldtype,variable)].figure = fig
    slp.plots[(fieldtype,variable)].axes = ax
    slp._setup_plots()
    return slp

def set_slice_data(sliceplot,ds):
    sliceplot._switch_ds(ds)

def batch_set_ray_data(lines,ds_list,ray_start,ray_end,use_ray_coord=True,return_times=False,variable="temperature",**kwargs):
    """

    """
    df_list_flattened = [sample_ray(ds,ray_start,ray_end,variable=variable) for ds_method in ds_list for ds in ds_method]

    if use_ray_coord:
        ray_length = math.sqrt((ray_end[0] - ray_start[0])**2 + (ray_end[1] - ray_start[1])**2 + (ray_end[2] - ray_start[2])**2)
        [line.set_data(df["ray_coord"].values,df[variable].values) for line, df in zip(lines,df_list_flattened)]
    else:
        [line.set_data(df["x"].values,df[variable].values) for line, df in zip(lines,df_list_flattened)]

    if return_times:
        return [df["time"].values[-1] for df in df_list_flattened]

def batch_set_error_data(lines,df_list,maxtime,loadvars=[0],maxlevel=0,withglobal=False,**kwargs):
    """

    """
    error_df_list_flattened = [df for error_method in df_list for df in error_method]
   
    idx_list = [df["time"].values.tolist().index(min(df["time"].values, key=lambda x:abs(x-maxtime)))+1 for df in error_df_list_flattened]
   
    i_line = 0 
    for df, idx in zip(error_df_list_flattened, idx_list):
        for level in range(maxlevel+1):
            for var in loadvars:
                lines[i_line].set_data(df["time"].values[:idx],df[f"error_var_{var}_level_{level}"].values[:idx]) 
                i_line += 1
        if withglobal:
            lines[i_line].set_data(df["time"].values[:idx],df[f"error_var_{var}_global"].values[:idx]) 
            i_line += 1

def batch_plot_ray(ax,root_dir,case_dir,methods,grid_sizes,i,ray_start,ray_end,ylim=[-0.2,1.4],use_ray_coord=True,variable="temperature",**kwargs):
    """

    """
    ds_list = batch_load_plt(root_dir,case_dir,methods,grid_sizes,i,**kwargs)

    if use_ray_coord:
        ray_length = math.sqrt((ray_end[0] - ray_start[0])**2 + (ray_end[1] - ray_start[1])**2 + (ray_end[2] - ray_start[2])**2)
        lines = setup_batch_line_plot(ax,methods,grid_sizes,[0,ray_length],ylim,"ray coordinate","scalar",**kwargs)
    else:
        lines = setup_batch_line_plot(ax,methods,grid_sizes,[ray_start[0],ray_end[0]],ylim,"x","scalar",**kwargs)

    batch_set_ray_data(lines,ds_list,ray_start,ray_end,use_ray_coord=use_ray_coord,variable=variable,**kwargs)

def batch_plot_error(ax,filename,root_dir,case_dir,methods,grid_sizes,maxtime,xlim=[0,0.5],ylim=[1e-6,1e-1],loadvars=[0],maxlevel=0,withglobal=True,**kwargs):
    """

    """
    error_df_list = batch_load_error(filename,root_dir,case_dir,methods,grid_sizes,loadvars=loadvars,withglobal=withglobal,**kwargs)
    lines = setup_batch_line_plot(ax,methods,grid_sizes,xlim,ylim,"time","error",errorlines=True,maxlevel=maxlevel,withglobal=withglobal,**kwargs)
    batch_set_error_data(lines,error_df_list,maxtime,loadvars=loadvars,maxlevel=maxlevel,withglobal=withglobal,**kwargs)
    




