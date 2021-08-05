Presentation/Paper Outline
----

# Introduction
1. Start with motivating example:
2. linear regression gives nearly the same result but for two very different datasets, a normally distributed blob of points, and a torus



<iframe src="two_samples.html"
  style="width:100%; height:500px;"
></iframe>



3. talk about shape as rigid in geometry vs topology
4. Talk about potential applications of why we would want to use the "shape" as a feature, or something to detect.
5. topology/homotopy is hard to compute so instead we use something combinatorial, homology
6. How to construct this combinatorial object: simplicial complexes (triangulate an object), cubical complexes, point coud of data and fill in the gaps
7. Compute homology to give topological information about this combinatorial object


# How to compute topology

We probably dont know the structure so we have to *learn* it or discover it this is **Persistent Homology**

1. Describe the persisent homology algorithm: distance metrics, filtrations, computing homology (rank of boundary matrix), show diagrams
2. break down the details with a couple of points

# Improving Persistentent Homology

Use jacobian of the underlying shape (See *Learning Varieties from Samples" by Strumfels et al 2018). This leverages local geometry and locally linear structure to  to capture topological features. Uses Epsilon Ellipsoids for the filtration. I call these the "Jacobian Ellipsoids" Filtration.

Expand the idea to SVD ellipsoids of local data at *each point*, picture below. I call these the "SVD Ellipsoids" filtration.


![Filtration Examples](figures_01-12.pdf)


Compare the persistence of Homology groups for each type of filtration.

## Show details about how each is calculated.


Details of each algorithm and how they are implemented in python.




# Extensions.

Show other things that utilize geometry.

* KNN-distance: it is definitely a filtration, but is it useful for homology? I dont think so but requires further investigation
* UMAP filtration: use UMAP to make filtration (McInnes et all 2019) uses riemannian metric to embed points where distances a strength varies between each point, alomost like a force directed graph or radial basis function interpolation. Topologically sound and cetegorically  correct.