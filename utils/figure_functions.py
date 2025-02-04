import numpy as np
import scipy
import pandas as pd
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import figurefirst as fifi
import fly_plot_lib_plot as fpl
import utils


class FlyWindVectors:
    def __init__(self, phi=np.pi / 4, g=0.4, psi=np.pi / 10, w=0.3, zeta=-np.pi / 4):
        """ Calculate air velocity vector from fly heading angle,
            ground velocity vector, and ambient wind velocity vector.
        """

        # Inputs
        self.phi = phi  # heading [rad]
        self.g = g  # ground velocity magnitude
        self.psi = psi  # ground velocity direction [rad]
        self.w = w  # ambient wind velocity magnitude
        self.zeta = zeta  # ambient wind velocity direction [rad]

        # Main variables
        self.phi_x = 0.0
        self.phi_y = 0.0

        self.v_para = 0.0
        self.v_perp = 0.0

        self.g_x = 0.0
        self.g_y = 0.0
        self.psi_global = 0.0

        self.a_para = 0.0
        self.a_perp = 0.0
        self.a = 0.0
        self.gamma = 0.0

        self.a_x = 0.0
        self.a_y = 0.0
        self.gamma_global = 0.0
        self.gamma_check = 0.0

        self.w_x = 0.0
        self.w_y = 0.0

        # Figure & axis
        self.fig = None
        self.ax = None

        # Run
        self.run()

    def run(self):
        """ Run main computations for fly-wind vector plot.
        """

        # Ground velocity in fly frame
        self.v_para = self.g * np.cos(self.psi)  # parallel ground velocity in fly frame
        self.v_perp = self.g * np.sin(self.psi)  # perpendicular ground velocity in fly frame

        # Air velocity in fly frame
        self.a_para = self.v_para - self.w * np.cos(self.phi - self.zeta)  # parallel air velocity in fly frame
        self.a_perp = self.v_perp + self.w * np.sin(self.phi - self.zeta)  # perpendicular air velocity in fly frame
        # self.a_para = self.v_para - self.w * np.cos(self.zeta - self.phi)
        # self.a_perp = self.v_perp - self.w * np.sin(self.zeta - self.phi)
        self.a = np.sqrt(self.a_para ** 2 + self.a_perp ** 2)  # air velocity magnitude
        self.gamma = np.arctan2(self.a_perp, self.a_para)  # air velocity direction [rad]
        # a_v = self.g * np.exp(self.psi * 1j) - self.w * np.exp((self.zeta - self.phi) * 1j)
        # self.a = np.abs(a_v)
        # self.gamma = np.angle(a_v)

        # Vector for heading, make same length as ground speed
        self.phi_x = self.g * np.cos(self.phi)  # heading x
        self.phi_y = self.g * np.sin(self.phi)  # heading y

        # Ground velocity in global frame
        self.psi_global = self.phi + self.psi  # direction of travel in global frame
        self.g_x = self.g * np.cos(self.psi_global)  # x-velocity in global frame
        self.g_y = self.g * np.sin(self.psi_global)  # y-velocity in global frame

        # Ambient wind velocity in global frame
        self.w_x = self.w * np.cos(self.zeta)  # ambient wind x in global frame
        self.w_y = self.w * np.sin(self.zeta)  # ambient wind y in global frame

        # Air velocity in global frame
        self.a_x = self.g_x - self.w_x  # x air velocity in global frame
        self.a_y = self.g_y - self.w_y  # y air velocity in global frame
        self.gamma_global = np.arctan2(self.a_y, self.a_x)  # air velocity direction in global frame
        self.gamma_check = self.gamma_global - self.phi  # air velocity direction in fly frame, should match gamma

    def compute_new_w(self, a_new=None):
        """ Compute the new ambient vector for a change in air speed
            while keeping ground velocity & air velocity direction the same.
        """

        # Set new air speed
        self.a = a_new

        # Compute new ambient wind
        w_v = self.g * np.exp(self.psi * 1j) - a_new * np.exp(self.gamma * 1j)
        self.w = np.abs(w_v)
        self.zeta = np.angle(w_v) + self.phi

        # Re-run
        self.run()

    def plot(self, ax=None, fly_origin=(0, 0), axis_size=None, axis_neg=True, show_arrow=True, fig_size=6,
             phi_color=(128 / 255, 128 / 255, 128 / 255),
             g_color=(32 / 255, 0 / 255, 255 / 255),
             a_color=(240 / 255, 118 / 255, 0 / 255),
             w_color=(47 / 255, 166 / 255, 0 / 255),
             lw=1.5, alpha=1.0):

        """ Plot fly wind vectors.
        """

        fly_origin = np.array(fly_origin)

        if axis_size is None:
            axis_size = 1.05 * np.max(np.array([self.w, self.g, self.a]))

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=100)
            self.fig = fig
            self.ax = ax

        # Plot axes
        ax.plot([axis_size, -axis_neg * axis_size], [0, 0], '--', linewidth=1.0, color='gray')
        ax.plot([0.0, 0.0], [-axis_neg * axis_size, axis_size], '--', linewidth=1.0, color='gray')

        # Plot fly-wind vectors
        ax.plot([fly_origin[0], fly_origin[0] + self.phi_x], [fly_origin[1], fly_origin[1] + self.phi_y], '-',
                linewidth=lw, color=phi_color, alpha=alpha, label=r'$\phi$')

        ax.plot([fly_origin[0], fly_origin[0] + self.g_x], [fly_origin[1], fly_origin[1] + self.g_y], '-',
                linewidth=lw, color=g_color, alpha=alpha, label=r'$\bar{g}$')

        ax.plot([fly_origin[0], fly_origin[0] + self.a_x], [fly_origin[1], fly_origin[1] + self.a_y], '-',
                linewidth=lw, color=a_color, alpha=alpha, label=r'$\bar{a}$')

        ax.plot([fly_origin[0] + self.a_x, fly_origin[0] + self.w_x + self.a_x],
                [fly_origin[1] + self.a_y, fly_origin[1] + self.w_y + self.a_y], '-',
                linewidth=lw, color=w_color, alpha=alpha, label=r'$\bar{w}$')

        # ax.plot([fly_origin[0], self.w_x], [fly_origin[1], self.w_y], '-',
        #         linewidth=lw, color='limegreen', alpha=alpha)

        ax.legend()

        # Plot arrows
        if show_arrow:
            mut = 10

            arrow_phi = FancyArrowPatch(posA=fly_origin,
                                        posB=fly_origin + (self.phi_x, self.phi_y),
                                        mutation_scale=mut, color=phi_color)

            arrow_g = FancyArrowPatch(posA=fly_origin,
                                      posB=fly_origin + (self.g_x, self.g_y),
                                      mutation_scale=mut, color=g_color)

            arrow_a = FancyArrowPatch(posA=fly_origin,
                                      posB=fly_origin + (self.a_x, self.a_y),
                                      mutation_scale=mut, color=a_color)

            arrow_w = FancyArrowPatch(posA=fly_origin + (self.a_x, self.a_y),
                                      posB=fly_origin + (self.w_x + self.a_x, self.w_y + self.a_y),
                                      mutation_scale=mut, color=w_color)

            ax.add_patch(arrow_phi)
            ax.add_patch(arrow_g)
            ax.add_patch(arrow_w)
            ax.add_patch(arrow_a)

        # Set axis properties
        ax.set_aspect(1)
        ax.autoscale()

        ax.set_xlim(-axis_size, axis_size)
        ax.set_ylim(-axis_size, axis_size)

        fifi.mpl_functions.adjust_spines(ax, [])


class LatexStates:
    """Holds LaTex format corresponding to set symbolic variables.
    """

    def __init__(self):
        self.dict = {'v_para': r'$v_{\parallel}$',
                     'v_perp': r'$v_{\perp}$',
                     'phi': r'$\phi$',
                     'phidot': r'$\dot{\phi}$',
                     'phiddot': r'$\ddot{\phi}$',
                     'w': r'$w$',
                     'zeta': r'$\zeta$',
                     'I': r'$I$',
                     'm': r'$m$',
                     'C_para': r'$C_{\parallel}$',
                     'C_perp': r'$C_{\perp}$',
                     'C_phi': r'$C_{\phi}$',
                     'km1': r'$k_{m_1}$',
                     'km2': r'$k_{m_2}$',
                     'km3': r'$k_{m_3}$',
                     'km4': r'$k_{m_4}$',
                     'd': r'$d$',
                     'psi': r'$\psi$',
                     'gamma': r'$\gamma$',
                     'alpha': r'$\alpha$',
                     'of': r'$\frac{g}{d}$',
                     'gdot': r'$\dot{g}$',
                     'v_para_dot': r'$\dot{v_{\parallel}}$',
                     'v_perp_dot': r'$\dot{v_{\perp}}$',
                     'v_para_dot_ratio': r'$\frac{\Delta v_{\parallel}}{v_{\parallel}}$'
                     }

    def convert_to_latex(self, list_of_strings, remove_dollar_signs=False):
        """ Loop through list of strings and if any match the dict, then swap in LaTex symbol.
        """

        if isinstance(list_of_strings, str):  # if single string is given instead of list
            list_of_strings = [list_of_strings]
            string_flag = True
        else:
            string_flag = False

        list_of_strings = list_of_strings.copy()
        for n, s in enumerate(list_of_strings):  # each string in list
            for k in self.dict.keys():  # check each key in Latex dict
                if s == k:  # string contains key
                    # print(s, ',', self.dict[k])
                    list_of_strings[n] = self.dict[k]  # replace string with LaTex
                    if remove_dollar_signs:
                        list_of_strings[n] = list_of_strings[n].replace('$', '')

        if string_flag:
            list_of_strings = list_of_strings[0]

        return list_of_strings





def make_color_map(color_list=None, color_proportions=None, N=256):
    """ Make a colormap from a list of colors.
    """

    if color_list is None:
        color_list = ['white', 'deepskyblue', 'mediumblue', 'yellow', 'orange', 'red', 'darkred']

    if color_proportions is None:
        color_proportions = np.linspace(0.01, 1, len(color_list) - 1)
        v = np.hstack((np.array(0.0), color_proportions))
    elif color_proportions == 'even':
        color_proportions = np.linspace(0.0, 1, len(color_list))
        v = color_proportions

    l = list(zip(v, color_list))
    cmap = LinearSegmentedColormap.from_list('rg', l, N=N)

    return cmap


def add_colorbar(fig, ax, data, cmap=None, label=None, ticks=None):
    offset_x = 0.017
    offset_y = 0.08

    cb_width = 0.75 * ax.get_position().width
    cb_height = 0.05 * ax.get_position().height

    cnorm = colors.Normalize(vmin=data.min(), vmax=data.max())
    cbax = fig.add_axes([offset_x + ax.get_position().x0, ax.get_position().y0 - offset_y, cb_width, cb_height])
    cb = fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap=cmap), cax=cbax, orientation='horizontal')
    cb.ax.tick_params(labelsize=7, direction='in')
    cbax.yaxis.set_ticks_position('left')
    cb.set_label(label, labelpad=0, size=8)
    cb.ax.set_xticks(np.round(np.linspace(data.min(), data.max(), 5), 2))


def image_from_xyz(x, y, z=None, bins=100, sigma=None):
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    z_min = np.min(z)
    z_max = np.max(z)

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[x_min - 1, x_max + 1], [y_min - 1, y_max + 1]])

    if z is None:  # use point density as color
        I = hist.T
    else:  # use z value as color
        I = np.zeros_like(hist)
        for bx in range(xedges.shape[0] - 1):
            x1 = xedges[bx]
            x2 = xedges[bx + 1]
            for by in range(yedges.shape[0] - 1):
                y1 = yedges[by]
                y2 = yedges[by + 1]
                xy_bin = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
                color_bin = z[xy_bin]
                if color_bin.shape[0] < 1:
                    I[by, bx] = z_min
                else:
                    I[by, bx] = np.mean(color_bin)

    if sigma is not None:
        I = scipy.ndimage.gaussian_filter(I, sigma=sigma, mode='reflect')

    return I


def plot_trajectory(xpos, ypos, phi, color, ax=None, size_radius=None, nskip=0,
                    colormap=None, colornorm=None, edgecolor='none', reverse=False):

    if color is None:
        color = phi

    color = np.array(color)

    # Set size radius
    xymean = np.mean(np.abs(np.hstack((xpos, ypos))))
    if size_radius is None:  # auto set
        xymean = 0.21 * xymean
        if xymean < 0.0001:
            sz = np.array(0.01)
        else:
            sz = np.hstack((xymean, 1))
        size_radius = sz[sz > 0][0]
    else:
        if isinstance(size_radius, list):  # scale defualt by scalar in list
            xymean = size_radius[0] * xymean
            sz = np.hstack((xymean, 1))
            size_radius = sz[sz > 0][0]
        else:  # use directly
            size_radius = size_radius

    if colornorm is None:
        colornorm = [np.min(color), np.max(color)]

    if reverse:
        xpos = np.flip(xpos, axis=0)
        ypos = np.flip(ypos, axis=0)
        phi = np.flip(phi, axis=0)
        color = np.flip(color, axis=0)

    if colormap is None:
        colormap = cm.get_cmap('bone_r')
        colormap = colormap(np.linspace(0.1, 1, 10000))
        colormap = ListedColormap(colormap)

    if ax is None:
        fig, ax = plt.subplots()

    fpl.colorline_with_heading(ax, np.flip(xpos), np.flip(ypos), np.flip(color, axis=0), np.flip(phi),
                               nskip=nskip,
                               size_radius=size_radius,
                               deg=False,
                               colormap=colormap,
                               center_point_size=0.0001,
                               colornorm=colornorm,
                               show_centers=False,
                               size_angle=20,
                               alpha=1,
                               edgecolor=edgecolor)

    ax.set_aspect('equal')
    xrange = xpos.max() - xpos.min()
    xrange = np.max([xrange, 0.02])
    yrange = ypos.max() - ypos.min()
    yrange = np.max([yrange, 0.02])

    if yrange < (size_radius / 2):
        yrange = 10

    if xrange < (size_radius / 2):
        xrange = 10

    ax.set_xlim(xpos.min() - 0.2 * xrange, xpos.max() + 0.2 * xrange)
    ax.set_ylim(ypos.min() - 0.2 * yrange, ypos.max() + 0.2 * yrange)

    # fifi.mpl_functions.adjust_spines(ax, [])


def pi_yaxis(ax=0.5, tickpispace=0.5, lim=None, real_lim=None):
    if lim is None:
        ax.set_ylim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_ylim(lim)

    lim = ax.get_ylim()
    ticks = np.arange(lim[0], lim[1] + 0.01, tickpispace * np.pi)
    tickpi = np.round(ticks / np.pi, 3)
    y0 = abs(tickpi) < np.finfo(float).eps  # find 0 entry, if present

    tickslabels = tickpi.tolist()
    for y in range(len(tickslabels)):
        tickslabels[y] = ('$' + str(Fraction(tickslabels[y])) + r'\pi $')

    tickslabels = np.asarray(tickslabels, dtype=object)
    tickslabels[y0] = '0'  # replace 0 entry with 0 (instead of 0*pi)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tickslabels)

    if real_lim is None:
        real_lim = np.zeros(2)
        real_lim[0] = lim[0] - 0.4
        real_lim[1] = lim[1] + 0.4

    if lim is None:
        ax.set_ylim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_ylim(lim)

    ax.set_ylim(real_lim)


def pi_xaxis(ax, tickpispace=0.5, lim=None):
    if lim is None:
        ax.set_xlim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_xlim(lim)

    lim = ax.get_xlim()
    ticks = np.arange(lim[0], lim[1] + 0.01, tickpispace * np.pi)
    tickpi = ticks / np.pi
    x0 = abs(tickpi) < np.finfo(float).eps  # find 0 entry, if present

    tickslabels = tickpi.tolist()
    for x in range(len(tickslabels)):
        tickslabels[x] = ('$' + str(Fraction(tickslabels[x])) + r'\pi$')

    tickslabels = np.asarray(tickslabels, dtype=object)
    tickslabels[x0] = '0'  # replace 0 entry with 0 (instead of 0*pi)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tickslabels)


def circplot(t, phi, jump=np.pi):
    """ Stitches t and phi to make unwrapped circular plot. """

    t = np.squeeze(t)
    phi = np.squeeze(phi)

    difference = np.abs(np.diff(phi, prepend=phi[0]))
    ind = np.squeeze(np.array(np.where(difference > jump)))

    phi_stiched = np.copy(phi)
    t_stiched = np.copy(t)
    for i in range(phi.size):
        if np.isin(i, ind):
            phi_stiched = np.concatenate((phi_stiched[0:i], [np.nan], phi_stiched[i + 1:None]))
            t_stiched = np.concatenate((t_stiched[0:i], [np.nan], t_stiched[i + 1:None]))

    return t_stiched, phi_stiched
