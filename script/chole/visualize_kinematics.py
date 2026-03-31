import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_3d_reconstruction(current_3d_position, fig_3d_plot, ax_3d_plot, tool_detected_left_right_flag=True, is_estimated_position_flag=False, 
                                elev=30, azim=225, x_range=(-7.5, 7.5), y_range=(-5, 5), z_range=(-35, 0), show_flag=True):
    """
    Visualize the 3D reconstruction of the tool tip position.
    
    Inputs:
    - current_3d_pose: The current 3D pose of the tool tip
    - fig_3d_plot: Matplotlib figure object for plotting
    - ax_3d_plot: Matplotlib axes object for plotting
    - tool_detected_left_right_flag: If True, the tool is detected on the left side of the image. If False, the tool is detected on the right side of the image. 
                                     By default, the tool is detected on the left side of the image.
    - is_estimated_position_flag: If True, the position is estimated. If False, the position is the current position.
    - elev: The elevation angle of the plot
    - azim: The azimuth angle of the plot
    - x_range: The x-axis range of the plot
    - y_range: The y-axis range of the plot
    - z_range: The z-axis range of the plot
    - show_flag: If True, the image will be displayed. If False, the image won't be displayed.
    """
    
    # Clear last plot
    ax_3d_plot.clear()
   
    if current_3d_position is not None: 
        # Plot the new 3D position
        current_3d_position_cm = current_3d_position * 100
        if tool_detected_left_right_flag:
            color = "g" # Green
        else:
            color = "r" # Red
        ax_3d_plot.scatter(current_3d_position_cm[0], current_3d_position_cm[1], current_3d_position_cm[2], c=color, marker='o')
        print(f"Current 3D position: {current_3d_position_cm}")
        ax_3d_plot.set_xlim(x_range)
        ax_3d_plot.set_ylim(y_range)
        ax_3d_plot.set_zlim(z_range)
        ax_3d_plot.set_xlabel('X Axis')
        ax_3d_plot.set_ylabel('Y Axis')
        ax_3d_plot.set_zlabel('Z Axis')
        if is_estimated_position_flag:
            ax_3d_plot.set_title(f'Predicted Next Position')
        else:
            ax_3d_plot.set_title(f'Current Position')

    # Set a consistent view angle
    ax_3d_plot.view_init(elev=elev, azim=azim)

    # Convert the Matplotlib figure to an OpenCV image using buffer_rgba
    fig_3d_plot.canvas.draw()
    image_3d_position = np.frombuffer(fig_3d_plot.canvas.tostring_rgb(), dtype=np.uint8)
    image_3d_position = image_3d_position.reshape(fig_3d_plot.canvas.get_width_height()[::-1] + (3,))
    image_3d_position = cv2.cvtColor(image_3d_position, cv2.COLOR_RGB2BGR)

    # Convert RGBA to BGR
    image_3d_position = cv2.cvtColor(image_3d_position, cv2.COLOR_RGBA2BGR)

    if show_flag:
        cv2.imshow('3D Position', image_3d_position)

    return image_3d_position


def visualize_2d_reconstruction(current_3d_position, fig_2d_plot, ax_2d_plot, tool_detected_left_right_flag=True, 
                                is_estimated_position_flag=False, x_range=(-7.5, 7.5), y_range=(-5, 5), show_flag=True):
    """
    Visualize the 2D reconstruction of the tool tip position with Z coordinate in the title.
    
    Inputs:
    - current_3d_pose: The current 3D pose of the tool tip
    - fig_2d_plot: Matplotlib figure object for plotting
    - ax_2d_plot: Matplotlib axes object for plotting
    - tool_detected_left_right_flag: If True, the tool is detected on the left side of the image.If False, the tool is detected on the right side of the image. 
                                     By default, the tool is detected on the left side of the image.
    - is_estimated_position_flag: If True, the position is estimated. If False, the position is the current position.
    - x_range: The x-axis range of the plot
    - y_range: The y-axis range of the plot
    - show_flag: If True, the image will be displayed. If False, the image won't be displayed.
    """
    
    # Clear the last plot
    ax_2d_plot.clear()
    
    if current_3d_position is not None: 
        # Plot the new 2D position (X and Y)
        current_3d_position_cm = current_3d_position * 100
        if tool_detected_left_right_flag:
            color = "g" # Green
        else:
            color = "r" # Red
        ax_2d_plot.scatter(current_3d_position_cm[0], current_3d_position_cm[1], c=color, marker='o')
        ax_2d_plot.scatter(0, 0, c='r', marker='o') # Indicate the origin (optimal tool position)
        ax_2d_plot.set_xlim(x_range)
        ax_2d_plot.set_ylim(y_range[::-1])
        ax_2d_plot.set_xlabel('X Axis')
        ax_2d_plot.set_ylabel('Y Axis')
        if is_estimated_position_flag:
            ax_2d_plot.set_title(f'Predicted Next Position - Z Coordinate: {current_3d_position_cm[2]:.2f} cm')
        else:
            ax_2d_plot.set_title(f'Current Position - Z Coordinate: {current_3d_position_cm[2]:.2f} cm')

    # Convert the Matplotlib figure to an OpenCV image using buffer_rgba
    fig_2d_plot.canvas.draw()
    image_2d_position = np.frombuffer(fig_2d_plot.canvas.tostring_rgb(), dtype=np.uint8)
    image_2d_position = image_2d_position.reshape(fig_2d_plot.canvas.get_width_height()[::-1] + (3,))
    image_2d_position = cv2.cvtColor(image_2d_position, cv2.COLOR_RGB2BGR)

    if show_flag:
        cv2.imshow('2D Position', image_2d_position)

    return image_2d_position


def generate_spiral_trajectory(num_points=100, height=35, radius=7.5, num_turns=3):
    """
    Generates a 3D spiral trajectory.
    
    Args:
    - num_points: Number of points in the trajectory.
    - height: Total height of the spiral.
    - radius: Radius of the spiral.
    - num_turns: Number of turns in the spiral.

    Returns:
    - trajectory: A numpy array of shape (num_points, 3) representing the 3D trajectory.
    """
    theta = np.linspace(0, 2 * np.pi * num_turns, num_points)
    z = np.linspace(0, height, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    trajectory = np.vstack((x, y, z)).T
    return trajectory


if __name__ == "__main__":
    # Create a Matplotlib figure and axes for 3D plotting
    fig_3d_plot = plt.figure()
    ax_3d_plot = fig_3d_plot.add_subplot(111, projection='3d')

    # Generate the spiral trajectory
    spiral_trajectory = generate_spiral_trajectory()

    # Visualize each point in the trajectory
    for current_3d_position in spiral_trajectory:
        visualize_3d_reconstruction(
            current_3d_position=current_3d_position,
            fig_3d_plot=fig_3d_plot,
            ax_3d_plot=ax_3d_plot,
            tool_detected_left_right_flag=True, # Assume tool is detected on the left for visualization
            is_estimated_position_flag=False,
            elev=30,
            azim=225,
            x_range=(-10, 10),
            y_range=(-10, 10),
            z_range=(-5, 40),
            show_flag=False
        )

    # Show the final plot
    plt.show()