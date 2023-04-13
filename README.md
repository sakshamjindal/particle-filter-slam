

# Particle Filter SLAM

### Command
To run the SLAM algorithm
```
python pfs.py
```

For texture mapping, refer to `texture.ipynb`


### Code structure

```
  main-repository/
  │
  ├── pfs.py - runs the particle SLAM algorithm
  ├── texture.ipynb - implementation of texture mapping
  ├── experiments.py - extension of pfs.py and run the experiments over hyperparameter
  │
  ├── data - contains data 
  │
  └── core -
     ├── dataloaders: contains dataloading modules
     ├── robot.py: contains the functions for predict and update step for the slam
     ├── map.py: contains the functions for building and maintaining map
     ├── points.py: contains developed utilies for handling points
     ├── model.py: contains the implmentation of motion model
     ├── params.py: contains robot sensor and map parameters
     └── pr2_utils.py: contains some common utils used in the project
  
  ```
  