#postprocessing.py>

import os.path
import re
import numpy as np
import yt
import glob
import pandas as pd
import math
import inspect

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
line_style_list = [
    "solid",
    "dashed",
    "dotted",
    "dashdot",
]
marker_list = [
    "x",
    "+",
    "o",
]

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)

def flatten_list(l):
    while all(isinstance(e, list) for e in l):
        l = [ee for e in l for ee in e]
    return l

def plt_count(plt_dir):
    """
    Counts the number of plt files in the given directory

    Args:
        plt_dir: The directory that contains the plt files.

    Returns:
        An Int which represents the number of plt files in the given directory.
    """
    plt_files = natural_sort(glob.glob(os.path.join(plt_dir, "plt*")))
    return len(plt_files)

def batch_plt_count(root_dir,
                    case_dir,
                    methods,
                    grid_sizes,
                    grid_prefix = "nx_"):
    """
    Counts the number of plt files in the directories that are composed of the
    arguments as {root_dir}/{method}/{case_dir}/{grid_prefix}{grid_size}.

    Args:
        root_dir: The root directory.
        case_dir: The case directory, e.g. twodimgaussianpulse/fine_to_coarse.
        methods: A list of strings of the method names that will be looped
          through when composing all the directories.
        grid_sizes: A list of Int grid_sizes that will be looped through when
          composing all the directories.
        grid_prefix: A string that will be added in front of each element in
          the grid_sizes.

    Returns:
        A nested list with the number of plt files for every grid size in
        `grid_sizes` for every method in `methods`.
        .
    """
    plt_count_list = []
    for method in methods:
        plt_count_method = []
        for grid_size in grid_sizes:
            full_dir = os.path.join(root_dir,
                                    method,
                                    case_dir,
                                    f'{grid_prefix}{grid_size}')
            plt_count_method.append(plt_count(full_dir))
        plt_count_list.append(plt_count_method)
    return plt_count_list

def load_plt(plt_dir, i):
    """
    Loads the `i`-th plt file in `plt_dir` into a yt dataset.

    Args:
        plt_dir: the directory that contains the plt files.

    Returns:
        The yt dataset that contains the data of the i-th plt file.
    """
    plt_files = natural_sort(glob.glob(os.path.join(plt_dir, "plt*")))
    ds = yt.load(plt_files[i])
    return ds

def batch_load_plt(root_dir,
                   case_dir,
                   methods,
                   grid_sizes,
                   i,
                   grid_prefix = "nx_"):
    """
    Loads the `i`-th plt file in all the directories that are composed of the
    arguments as {root_dir}/{method}/{case_dir}/{grid_prefix}{grid_size}.

    Args:
        root_dir: The root directory.
        case_dir: The case directory, e.g. twodimgaussianpulse/fine_to_coarse.
        methods: A list of strings of the method names that will be looped
          through when composing all the directories.
        grid_sizes: A list of Int grid_sizes that will be looped through when
          composing all the directories.
        i: The index of the plt file that should be loaded.
        grid_prefix: A string that will be added in front of each element in
          the grid_sizes.

    Returns:
        A nested list with a yt dataset for every grid size in `grid_sizes` for
        every method in `methods`.
    """
    ds_list = []
    for method in methods:
        ds_method = []
        for grid_size in grid_sizes:
            full_dir = os.path.join(root_dir,
                                    method,
                                    case_dir,
                                    f'{grid_prefix}{grid_size}')
            ds_method.append(load_plt(full_dir, i))
        ds_list.append(ds_method)
    return ds_list

def load_error(file_path,
               skiplines = 0,
               vars_count = 1,
               var_to_load = 0,
               has_global = True,
               load_global = True):
    """
    Loads the error file at `filepath` into a pandas dataframe.

    Args:
        file_path: The full file path of the file that contains the error data.
        skiplines: The number of lines that should be skipped at the top of the
          file.
        vars_count: The number of variables that are in the error file (used to
        calculate the number of levels in the simulation).
        var_to_load: Index of the variable that should be loaded
          (corresponding to the order of the variables in the error file).
        has_global: If True, the last vars_count columns are assumed to be the
          global (cumulative) error on all grid levels.
        load_global: If True, the global error is loaded.

    Returns:
        A pandas dataframe that contains the error for the variable to be
        loaded on every grid level. The error on grid level X will have the
        corresponding key `error_level_X`. If the global level is to be loaded,
        it will have the key `error_global`.
    """
    with open(file_path) as f:
        lines = f.readlines()
        time = [float(line.split()[0]) for line in lines[skiplines:]]
        df = pd.DataFrame({"time": time})
        ncolumns = len(lines[1].split())-1
        if has_global:
            nlevels = int(ncolumns/vars_count) - 1
        else:
            nlevels = int(ncolumns/vars_count)
        active_column = 1
        # This implementation could definitely be made better
        for level in range(nlevels):
            for var in range(vars_count):
                if var == var_to_load:
                    var_error = [float(line.split()[active_column]) for line in
                                 lines[skiplines:]]
                    df[f"error_level_{level}"] = var_error
                active_column += 1
        if load_global and has_global:
            for var in range(vars_count):
                if var == var_to_load:
                    var_error = [float(line.split()[active_column]) for line in 
                                 lines[skiplines:]]
                    df[f"error_global"] = var_error
                active_column += 1
    return df

def batch_load_error(file_name,
                     root_dir,
                     case_dir,
                     methods,
                     grid_sizes,
                     grid_prefix = "nx_",
                     **load_error_kwargs):
    """
    Loads the error file in all the directories that are composed of the
    arguments as {root_dir}/{method}/{case_dir}/{grid_prefix}{grid_size}.

    Args:
        file_name: The name of the error file.
        root_dir: The root directory.
        case_dir: The case directory, e.g. twodimgaussianpulse/fine_to_coarse.
        methods: A list of strings of the method names that will be looped
          through when composing all the directories.
        grid_sizes: A list of Int grid_sizes that will be looped through when
          composing all the directories.
        grid_prefix: A string that will be added in front of each element in
          the grid_sizes.
        **load_error_kwargs: Keyword arguments that will be forwarded to the
          postprocessing.load_error function.

    Returns:
        A nested list with a pandas dataframe containing the error for every
        grid size in `grid_sizes` for every method in `methods`.

    """
    error_df_list = []
    for method in methods:
        error_method = []
        for grid_size in grid_sizes:
            filepath = os.path.join(root_dir,
                                    method,
                                    case_dir,
                                    f'{grid_prefix}{grid_size}',
                                    file_name)
            df = load_error(filepath, **load_error_kwargs)
            error_method.append(df) 
        error_df_list.append(error_method) 
    return error_df_list

def sample_ray(ds,
               ray_start,
               ray_end,
               field_type = "gas",
               variable = "temperature"):
    """
    Samples a yt dataset on a ray. Note that the ray will start and end on
    the closest cell centers to the given start and end coordinates
    respectively.

    Args:
        ds: The yt dataset that will be sampled.
        ray_start: The coordinate where the ray should start.
        ray_end: The coordinate where the ray should end.
        field_type: The first part of the yt field (field_type, variable) that
          will be loaded.
        variable: The second part of the yt field (field_type, variable) that
          will be loaded.

    Returns:
        A pandas dataframe that has a column for x, y, z, time, variable, and
        relative ray coordinate, and has an entry for each cell that is crossed
        by the ray.
    """
    ray = ds.ray(ray_start,ray_end)
    srt = np.argsort(ray[(field_type,"x")])
    df = pd.DataFrame({f: np.array(ray[(field_type,f)][srt]) for f in ["x", 
                      "y", "z", variable]})
    t = ds.current_time.to_value()
    df["time"] = t
    df["ray_coord"] = ((df["x"] - df["x"][0]) ** 2 
                       + (df["y"] - df["y"][0]) ** 2 
                       + (df["z"] - df["z"][0]) ** 2) ** 0.5    
    return df

def setup_batch_line_plot(ax,
                          methods,
                          grid_sizes,
                          as_error_lines = False,
                          max_level = 0,
                          with_global = False,
                          color_methods = True,
                          line_styles = line_style_list,
                          markers = marker_list,
                          colors = color_list,
                          line_width = 1,
                          label_grid_appendix=" cells"):
    """
    Adds one or multiple line plots without data to the provided plot axes
    object for each method in methods and each grid size in grid_sizes. This
    function can be used in conjunction with the batch_set_ray_data or
    batch_set_error_data functions and lends itself to matplotlib's
    funcAnimation. Every line plot will have a corresponding label formatted as
    {method}; {grid_size}{grid_appendix}.

    Args:
        ax: The matplotlib Axes object to which the line plots will be added.
        methods: A list of strings of the method names that will be looped
          through when adding the line plots and composing the labels.
        grid_sizes: A list of Int grid_sizes that will be looped through when
          adding the line plots and composing the labels.
        as_error_lines: If True, set up the line plots for an error plot.
          This means that for each method and grid size, max_level+1 line plots
          will be added, plus an extra one if with_global is True.
        max_level: The maximum level for the error line plots that will be
          added per method and grid size. Only used with as_error_lines is
          True.
        with_global: If True, an extra line plot will be added for each method
          and grid size if as_error_lines is True.
        color_methods: If True, use a different color for each method and
          a different line style for each grid size. Vice versa if False.
        line_styles: A list of the line styles that will be used for the line
          plots.
        markers: A list of the markers that will be used to differentiate
          different grid levels when as_error_lines is True. If with_global
          is True, the extra global line will not have markers.A
        line_styles: A list of the colors that will be used for the line
          plots.
        line_width: The line width that will be used for every line plot.
          label_grid_appendix: A string that will be added at the end of the
          label to clarify the grid specification.

    Returns:
        A list of the line plot objects.

    """
    lines = []
    for i_method in range(len(methods)):
        for i_grid_size in range(len(grid_sizes)):
            if color_methods:
                color = colors[i_method]
                line_style = line_styles[i_grid_size]
            else:
                color = colors[i_grid_size]
                line_style = line_styles[i_method]
            if as_error_lines:
                if max_level >= 0:
                    for level in range(max_level+1):
                        label = (f'{methods[i_method]};'
                                f'{grid_sizes[i_grid_size]}{label_grid_appendix};'
                                f'level {level}')
                        line = ax.plot([], [], 
                                       color = color,
                                       linewidth = line_width,
                                       linestyle = line_style,
                                       label = label,
                                       marker = markers[level],
                                       markevery = 0.05)[0]
                        lines.append(line)
                if with_global:
                    label = (f'{methods[i_method]};'
                             f'{grid_sizes[i_grid_size]}{label_grid_appendix};'
                             f'global level')
                    line = ax.plot([], [],
                                   color = color,
                                   linewidth = line_width,
                                   linestyle = line_style,
                                   label = label)[0]
                    lines.append(line)
            else:
                label = (f'{methods[i_method]};'
                         f'{grid_sizes[i_grid_size]}{label_grid_appendix}')
                line = ax.plot([], [],
                               color = color,
                               linewidth = line_width,
                               linestyle = line_style,
                               label = label)[0]
                lines.append(line)
    return lines

def setup_slice_plot(fig,
                     ax,
                     ds,
                     field_type,
                     variable,
                     zlim,
                     log_scale = False,
                     size = 10,
                     font_size = 12,
                     **kwargs):
    """
    Adds a yt slice plot to provided plot axes.

    Args:
        fig: The Matplotlib figure that owns the axes.
        ax: The Matplotlib Axes object to which the line plots will be added.
        ds: The yt dataset that will be used for the slice plot
        field_type: The first part of the yt field (field_type, variable) that
          will be sampled.
        variable: The second part of the yt field (field_type, variable) that
          will be sampled.
        zlim: The range of values of the chosen variable that will be shown.
        log_scale: If True, the values of the chosen variable will be
          logarithmically mapped onto the colorbar scale.
        size: The figure size. (Added here because this value seems to be hard
          to control outside of yt)
        font_size: The font size of all text on the figure. (Added here because
          this value seems to be hard to control outside of yt)
        **kwargs: keyword arguments are passed on to the yt `sliceplot`
          function.

    Returns:
        The yt slice plot object.
    """
    slp = yt.SlicePlot(ds, "z", 
                       (field_type, variable),
                       fontsize = font_size,
                       origin = 'native',
                       **kwargs)
    slp.set_log((field_type,variable), log_scale)
    slp.set_xlabel('x')
    slp.set_ylabel('y')
    slp.set_zlim((field_type,variable), zlim[0], zlim[1])
    slp.hide_colorbar()
    slp.set_figure_size(size)
    slp.plots[(field_type,variable)].figure = fig
    slp.plots[(field_type,variable)].axes = ax
    slp._setup_plots()
    return slp

def set_slice_data(sliceplot,ds):
    sliceplot._switch_ds(ds)

def batch_set_ray_data(lines,
                       df_list,
                       use_ray_coord = True,
                       variable = "temperature"):
    """
    Sets the data of the provided lines to the values of the provided
    dataframes as variable vs spatial coordinate. It is assumed that the lines
    were set up using the setup_batch_line_plot function and the df_list was
    generated with batch_sample_ray.

    Args:
        lines: A list of Matplotlib line plot objects of which the data will be
          set in this function. Assumed to have been set up with
          setup_batch_line_plot.
        df_list: A list of pandas dataframes (assumed to have been loaded with
          batch_sample_ray), of which the i-th dataframe will be used to set
          the data of the i-th line plot.
        use_ray_coord: If True, use the ray_coord values of the dataframe as
          the x values.
        variable: Name of the variable in the dataframes in df_list that will
          be used for the y values.
    """
    # Flatten the list if not flattened yet
    flattened_df_list = flatten_list(df_list)
    if use_ray_coord:
        [line.set_data(df["ray_coord"].values,df[variable].values) for line, df
                in zip(lines, flattened_df_list)]
    else:
        [line.set_data(df["x"].values,df[variable].values) for line, df in
                zip(lines, flattened_df_list)]

def batch_set_error_data(lines,
                        df_list,
                        max_time,
                        max_level = 0,
                        with_global = False):
    """
    Sets the data of the provided lines to the error in the provided
    dataframes as error vs time. It is assumed that the lines were set up using
    the setup_batch_line_plot function and the df_list was generated with
    batch_load_error.

    Args:
        lines: A list of Matplotlib line plot objects of which the data will be
          set in this function. Assumed to have been set up with
          setup_batch_line_plot.
        df_list: A list of pandas dataframes (assumed to have been loaded with
          batch_load_error), of which the i-th dataframe will be used to set
          the data of the i-th line plot.
        max_time: Maximum time value that will be set. 
        max_level: Maximum level that will be set. Has to correspond with the
          maximum level used for setup_batch_line_plot.
        with_global: If True, the global error data will also be set as data
          for some of the lines.
    """
    # Flatten the list if not flattened yet
    flattened_df_list = flatten_list(df_list)
    # Find the row for each dataframe for which the corresponding time is
    # closest to `max_time`
    idx_list = [df["time"].values.tolist().index(min(df["time"].values,
                key=lambda x:abs(x-max_time)))+1 for df in flattened_df_list]
   
    i_line = 0 
    for df, idx in zip(flattened_df_list, idx_list):
        if max_level >= 0:
            for level in range(max_level+1):
                lines[i_line].set_data(df["time"].values[:idx],
                                       df[f"error_level_{level}"].values[:idx])
                i_line += 1
        if with_global:
            lines[i_line].set_data(df["time"].values[:idx],
                                   df[f"error_global"].values[:idx])
            i_line += 1

def batch_plot_ray(ax,
                   root_dir,
                   case_dir,
                   methods,
                   grid_sizes,
                   i,
                   ray_start,
                   ray_end,
                   use_ray_coord = True,
                   variable = "temperature",
                   return_df_list = False,
                   **kwargs):
    """
    Adds a line plot of the of the specified variable sampled on a ray
    versus a spatial coordinate to the provided plot axes using the data of the
    i-th plot file for all the directories that are composed of the arguments as
    {root_dir}/{method}/{case_dir}/{grid_prefix}{grid_size}.

    Args:
        ax: The Matplotlib Axes object to which the line plots will be added.
        root_dir: The root directory.
        case_dir: The case directory, e.g. twodimgaussianpulse/fine_to_coarse.
        methods: A list of strings of the method names that will be looped
          through when composing all the directories.
        grid_sizes: A list of Int grid_sizes that will be looped through when
          composing all the directories.
        i: The index of the plt file that should be loaded.
        ray_start: The coordinate where the ray should start.
        ray_end: The coordinate where the ray should end.
        use_ray_coord: If True, use the ray_coord values of the dataframe as
          the x values.
        variable: Name of the variable in the yt dataset that will be used for
          the y values.
        return_df_list: If True, return the list of pandas dataframes
          containing the data sampled on the ray.
        **kwargs: Keyword arguments that will be passed to batch_load_plt and
          setup_batch_line_plt.

    Returns:
        The list of pandas dataframes containing the data sampled on the ray if
        return_df_list is True.
    """
    batch_load_plt_args = list(inspect.signature(batch_load_plt).parameters)
    batch_load_plt_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in
                           batch_load_plt_args}
    setup_batch_line_plot_args = list(inspect.signature(
                                      setup_batch_line_plot).parameters)
    setup_batch_line_plot_dict = {k: kwargs.pop(k) for k in dict(kwargs) if
                                  k in setup_batch_line_plot_args}
    if len(kwargs) != 0:
        raise ValueError(f"{next(iter(kwargs))} is not a valid keyword argument")

    ds_list = batch_load_plt(root_dir,
                             case_dir,
                             methods,
                             grid_sizes,
                             i,
                             **batch_load_plt_dict)
    ds_list = flatten_list(ds_list)
    df_list = [sample_ray(ds, ray_start, ray_end, variable = variable) for ds
               in ds_list]

    if use_ray_coord:
        ray_length = math.sqrt((ray_end[0] - ray_start[0]) ** 2 
                               + (ray_end[1] - ray_start[1]) ** 2 
                               + (ray_end[2] - ray_start[2]) ** 2)
        lines = setup_batch_line_plot(ax,
                                      methods,
                                      grid_sizes,
                                      **setup_batch_line_plot_dict)
    else:
        lines = setup_batch_line_plot(ax,
                                      methods,
                                      grid_sizes,
                                      **setup_batch_line_plot_dict)

    batch_set_ray_data(lines,
                       df_list,
                       use_ray_coord = use_ray_coord,
                       variable = variable)

    if return_df_list:
        return df_list

def batch_plot_error(ax,
                     file_name,
                     root_dir,
                     case_dir,
                     methods,
                     grid_sizes,
                     max_time,
                     var_to_load = 0,
                     max_level = 0,
                     with_global = True,
                     **kwargs):
    """
    Adds a line plot of the error versus time to the provided plot axes for all
    the directories that are composed of the
    arguments as {root_dir}/{method}/{case_dir}/{grid_prefix}{grid_size}.

    Args:
        ax: The Matplotlib Axes object to which the line plots will be added.
        file_name: The name of the error file.
        root_dir: The root directory.
        case_dir: The case directory, e.g. twodimgaussianpulse/fine_to_coarse.
        methods: A list of strings of the method names that will be looped
          through when composing all the directories.
        grid_sizes: A list of Int grid_sizes that will be looped through when
          composing all the directories.
        max_time: Maximum time value that will be set. 
        var_to_load: Index of the variable that should be loaded
          (corresponding to the order of the variables in the error file).
        max_level: Maximum level that will be set. Has to correspond with the
          maximum level used for setup_batch_line_plot.
        with_global: If True, the global error data will also be set as data
          for some of the lines.
        **kwargs: Keyword arguments that will be passed to batch_load_error and
          setup_batch_line_plt.
    """
    load_error_args = list(inspect.signature(load_error).parameters)
    load_error_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in
                       load_error_args}
    setup_batch_line_plot_args = list(inspect.signature(
                                      setup_batch_line_plot).parameters)
    setup_batch_line_plot_dict = {k: kwargs.pop(k) for k in dict(kwargs) if
                                  k in setup_batch_line_plot_args}
    if len(kwargs) != 0:
        raise ValueError(f"{next(iter(kwargs))} is not a valid keyword argument")

    error_df_list = batch_load_error(file_name,
                                     root_dir,
                                     case_dir,
                                     methods,
                                     grid_sizes,
                                     var_to_load = var_to_load,
                                     load_global = with_global,
                                     **load_error_dict)
    lines = setup_batch_line_plot(ax,
                                  methods,
                                  grid_sizes,
                                  as_error_lines = True,
                                  max_level = max_level,
                                  with_global = with_global,
                                  **setup_batch_line_plot_dict)
    batch_set_error_data(lines,
                         error_df_list,
                         max_time,
                         max_level = max_level,
                         with_global = with_global)
