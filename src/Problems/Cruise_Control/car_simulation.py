import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to update the plot in each animation frame
def update(frame):
    # Clear the previous frame
    plt.clf()
    
    # Update the position of the object
    global position
    position += velocity
    
    # Plot the object at its current position
    plt.plot(position, 0, 'ro', markersize=10)
    
    # Set plot limits
    plt.xlim(0, 10)
    plt.ylim(-1, 1)
    
    # Set plot title
    plt.title('1D Object Motion')
    
# Initialize variables
position = 0  # Initial position
velocity = 0.1  # Velocity of the object

# Create a figure and axis
fig, ax = plt.subplots()

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=range(100), interval=100)

# Show the animation
plt.show()

ani.save("src/Problems/Cruise_Control/simulations/test.gif")

