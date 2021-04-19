
![](/doc/PC2.jpg)

# Implementation
The best method is based on (this paper)[https://hal.archives-ouvertes.fr/hal-01959578/document]. Thanks to the author, Hassan Bouchiba, for helping with questions I had.
The rendering engine first renders point clouds into a texture, estimates occlusions, then does push-pull filtering to fill empty gaps and detected occluded points.

It is not exactly ideal for the source USGS Lidar Data, because this data has very low density along building edges. This makes it very difficult to estimate points that ought to be occluded via screen space techniques.


# Todo
 1. Clean-up shaders.
 2. Instead of making 16 samples during each push-pull iteration then doing the convolution, make use of bilinear sampling by precomputing less fractional locations to sample and letting the hardware do all the work.
 3. Using the octree pytorch sparse tensor / stable-sorted cuda array, estimate density of points then make use of that to help finding occluding points. Also compute and use normals.
 4. Lighting with screen space or precomputed normals.
 5. Benchmarking and optimize kernels. Can move a lot of the fractional offset computation to vertex shaders to make the fragment shaders quicker.
