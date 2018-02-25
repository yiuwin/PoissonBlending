# PoissonPyramidBlending
Seamlessly blend an object or texture from a source image into a target image by implementing Poisson Blending in Python

In my implementation of the Poisson Blending algorithm, I have added a function generateData(back,fore,mask) that takes in the source, targe, and mask images, and generates I, J, S, B of type numpy arrays. Since I first split my images into their respective R, G, and B channels, generateData(back,fore,mask) is called and returned three times returning a total of (3*4) numpy arrays. The returned arrays are fed into a function of the scipy library to be converetd into coordinate format and then compressed to compressed sparse row format. This way, it can be solved with a call to scipy.sparse.linalg.spsolve. The three 1D vectors belonging to each of the R, G, and B channels are reshaped back to its original dimensions and outputted. 

Bonus:
An implementation for gradient mixing was implemented in the generateData(back,fore,mask) function in blendingPoisson (#in comments) to produce the following result. Instead of using just the gradient of the foreground image where the mask image appears white, I took the gradient using the formula: 

gradient = α*(gradient of foreground) + (1-α)*(gradient of background)

The α gives me control of how much the resulting pixel will be influenced by the foreground, and how much will be influenced by the background. Arbitrarily, I took α=0.5.
