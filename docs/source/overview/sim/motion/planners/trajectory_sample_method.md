# TrajectorySampleMethod

`TrajectorySampleMethod` is an enumeration that defines different strategies for sampling points along a trajectory. It provides meaningful names for various sampling methods, making trajectory planning code more readable and maintainable.

## Enum Members

- **TIME**:  
  Sample trajectory points based on fixed time intervals.  
  Example: Generate a point every 0.01 seconds.

- **QUANTITY**:  
  Sample a specified number of points along the trajectory, regardless of the time interval.  
  Example: Generate exactly 100 points between start and end.

- **DISTANCE**:  
  Sample points based on fixed distance intervals along the path.  
  Example: Generate a point every 1 cm along the trajectory.
