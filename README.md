To run simulation:

```
$ python simulation.py gas/2000k.input gas/2000Ksim.dat
```

It will produce files with parameters over times and file with coordinates.

To see how parameters change, take a look into notebook, to visualize
simulation use python script below:

```
$ python visualization.py some_unnecessary_arg gas/coords_To_2000.0_tau_0.0025_n_72000Ksim.dat offline
```

To run this you would need [vispy](http://vispy.org/index.html) - which is the best
tool for OpenGL scientific visualization in Python.
