import numpy as np
import argparse
import math

def calculate_time_to_hit_ground(height):
    # Acceleration due to gravity (m/s^2)
    g = 9.81

    # Using the equation of motion: s = ut + (1/2) * g * t^2
    # where s is the height, u is initial velocity (0 in this case), g is acceleration due to gravity, and t is time


    # Rearranging the equation to solve for time:
    # t^2 - (2 * h / g) = 0
    # Using quadratic formula: t = (-b ± sqrt(b^2 - 4ac)) / (2a)

    a = 1  # coefficient of t^2
    b = 0  # coefficient of t
    c = -2 * height / g


    # Calculate the discriminant
    discriminant = b**2 - 4*a*c

    # Check if the discriminant is non-negative (to avoid complex roots)
    if discriminant >= 0:
        # Calculate both roots of the quadratic equation
        root1 = (-b + math.sqrt(discriminant)) / (2*a)
        root2 = (-b - math.sqrt(discriminant)) / (2*a)
        

        # The positive root is the time it takes for the ball to hit the ground
        time_to_hit_ground = max(root1, root2)

        return time_to_hit_ground
    else:
        # If the discriminant is negative, it means the ball does not hit the ground
        return None
    
    # Get tower height from the user
tower_height = float(input("Enter the height of the tower in meters: "))

# Calculate the time it takes for the ball to hit the ground
time_to_hit_ground = calculate_time_to_hit_ground(tower_height)

# Check if the ball hits the ground
if time_to_hit_ground is not None:
    print(f"The time it takes for the ball to hit the ground is approximately {time_to_hit_ground:.2f} seconds.")
else:
    print("The ball does not hit the ground.")





#print time in yrs it takes for spaceship to reach destination

def calc_t(distance, speed):
    #time observed from earth
    t_earth = distance / speed

    #time observed from spaceship
    gamma = 1 / np.sqrt(1 - speed**2)
    t_ship = t_earth / gamma

    return t_ship, t_earth
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('distance', type=float, help='the distance to the destination in light years')
    parser.add_argument('speed', type=float, help='the speed of the spaceship as a fraction of the speed of light')

    args = parser.parse_args()

    t_ship, t_earth = calc_t(args.distance, args.speed)

    print(f"The time observed from the spaceship is {t_ship:.2f} years")
    print(f"The time observed from earth is {t_earth:.2f} years")

if __name__ == '__main__':
    main()



    




