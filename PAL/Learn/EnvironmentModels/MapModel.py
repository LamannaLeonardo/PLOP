import os

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.path import Path

import Configuration

import numpy as np

from scipy.spatial import ConvexHull, convex_hull_plot_2d

class MapModel:

    def __init__(self):

        # x min coordinate in centimeters
        self.x_min = Configuration.MAP_X_MIN

        # y min coordinate in centimeters
        self.y_min = Configuration.MAP_Y_MIN

        # x max coordinate in centimeters
        self.x_max = Configuration.MAP_X_MAX

        # y max coordinate in centimeters
        self.y_max = Configuration.MAP_Y_MAX

        # x and y centimeters per pixel in the resized grid obstacle map
        self.dx = Configuration.MAP_GRID_DX
        self.dy = Configuration.MAP_GRID_DY

        # Initial collision cells
        self.collision_cells = []

        # x axis length on map (centimeters)
        self.x_axis_len = abs(self.x_max - self.x_min)

        # y axis length on map (centimeters)
        self.y_axis_len = abs(self.y_max - self.y_min)

        plt.style.use('default')
        plt.gca().set_aspect('equal')
        self.fig, self.axes = plt.subplots(figsize=(int(self.y_axis_len/100),  # 1 is 100 pixel, 1 pixel is 1 centimeter
                                                    int(self.x_axis_len/100)))

        # Set figure window size
        window_x = (int(self.x_min/100), int(self.x_max/100))
        window_y = (int(self.y_min/100), int(self.y_max/100))
        self.axes.set_xlim(window_x)
        self.axes.set_ylim(window_y)

        # Remove white margins
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.axis('off')

        self.fig.canvas.draw()

        self.grid = None


    def update_occupancy(self, occupancy_points, pos, angle, file_name, collision=False):
        # Draw depth occupancy points
        plt.figure(self.fig)

        try:
            self.clear_old_points(occupancy_points[:, :-1])
        except:
            pass

        self.axes.plot(occupancy_points[:, 0], occupancy_points[:, 1], 'o', markerfacecolor='none',
                       markeredgecolor='k', alpha=0.5, markersize=1)

        self.set_grid()

        if collision:
            self.fig.canvas.draw()

        if Configuration.PRINT_IMAGES and Configuration.PRINT_TOP_VIEW_IMAGES:
            count = len([el for el in os.listdir('/'.join(file_name.split('/')[:-1])) if 'topview' in el and 'grid' not in el])
            file_name = '/'.join(file_name.split('/')[:-1]) + '/topview_{}.png'.format(count)
            self.print_top_view(pos, angle, file_name)


    def clear_old_points(self, pts):

        pts = pts[(abs(pts[:,0])<12.5) & (abs(pts[:,1])<12.5)]
        hull = ConvexHull(pts)

        # DEBUG: Plot convex hull
        # plt.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], 'b')
        # plt.savefig('conves.png')
        hull_path = Path(pts[hull.vertices])

        # Update data in each line of the plot
        for line in self.axes.get_lines():
            line_pts = line.get_xydata()
            line_pts = line_pts[(abs(line_pts[:, 0]) < 12.5) & (abs(line_pts[:, 1]) < 12.5)]
            mask = hull_path.contains_points(line_pts)
            new_line_pts = line_pts[~mask, :]
            line.set_data(new_line_pts[:, 0], new_line_pts[:, 1])

        # self.fig.canvas.draw()


    def print_top_view(self, pos, angle, file_name):

        # Draw agent arrow
        agent_width = 0.2  # 20 centimeters
        agent_arrow = self.axes.arrow(pos[0], pos[1],
                        agent_width * np.cos(np.deg2rad(angle + 90)), agent_width * np.sin(np.deg2rad(angle + 90)),
                        head_width=agent_width, head_length=agent_width,
                        length_includes_head=True, fc="Red", ec="Red", alpha=0.9)

        plt.gca().set_aspect('equal')

        # Print agent top view map
        if Configuration.PRINT_TOP_VIEW_IMAGES:
            plt.savefig(file_name)

        # Remove agent arrow and white margins
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        agent_arrow.remove()
        plt.axis('off')

        self.fig.canvas.draw()


    def set_grid(self):

        # Rescale agent top view
        self.grid = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb()).convert('L')
        resized_grid = self.grid.resize((round(self.y_axis_len/self.dy), round(self.x_axis_len/self.dx)))
        self.grid = np.array(resized_grid)

        # Binarize rescaled agent top view
        self.grid[(self.grid < 250)] = 0  # 250 is an heuristic threshold
        self.grid[(self.grid >= 250)] = 1
