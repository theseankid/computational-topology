computational-topology
----

This collection of scripts and notebooks written in `python` are related to computing and visualizing Persistent Homology.

- the `functions` folder contains `distances` which allows you to calculate distances based on various metrics and embeddings.
- `examples` are python notebooks with computed examples as well as visualizations that are both interactive (using the `plotly`) and static (using `matplotlib`) to explore your data
- the `theory` folder contains an extended write up of how these ideas work as well as references

Installation
----

just to use the distances functions

```
pip install cython
pip install ripser
pip install scikit-learn
```
then ensure you are in the `computational-topology` directory by navigating in the shell or in python by 

```python
import os
os.chdir('path/to/directory/of/computational-topology')
import distances
```


if you are having trouble with the `cython` installation, reinstall the `numpy` python package

```
pip uninstall numpy
pip install numpy
```

to use the examples `pip install -r requirements.txt`

Questions and Contributing
----

Reach out if you have any questions about the contents of this repo, would like to contribute, or to collaborate.
