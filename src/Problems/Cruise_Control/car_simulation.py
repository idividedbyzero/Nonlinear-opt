import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List

# Function to update the plot in each animation frame
def update(frames, position:List[float], tl_position: List[float], tl_state: List[bool]):
    # Clear the previous frame
    plt.clf()
    
    # Plot the object at its current position
    plt.plot(position, 0, 'ro', markersize=10)
    plt.axhline(y=0, color='black', linestyle='-')

    for i, p in enumerate(tl_position):
        color="red" if tl_state[i] else "green"
        plt.axvline(x=p, color=color, linestyle="-")
    
    # Set plot limits
    plt.xlim(0, 50)
    plt.ylim(-1, 1)
    
    # Set plot title
    plt.title('1D Object Motion')
    
# Initialize variables
position = 0  # Initial position
velocity = 0.1  # Velocity of the object
tl_position=[30]
tl_state=[True]

# Create a figure and axis
fig, ax = plt.subplots()

interval=20
# Create the animation
ani = animation.FuncAnimation(fig, lambda x: update(x, position+x*velocity, tl_position, tl_state), frames=range(interval), interval=interval)

# Show the animation
plt.show()

ani.save("src/Problems/Cruise_Control/simulations/test.gif")

