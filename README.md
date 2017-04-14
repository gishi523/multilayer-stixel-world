# multilayer-stixel-world
An implementation of multi-layered stixel computation

====

![stixel-world]()

## Description
- An implementation of the Multi-Layered Stixel computation based on [1].
- Extracts the static Stixels from the input disparity map.
- Not a dynamic Stixel. It means that tracking and estimating motion of each Stixel is not supported.

## References
- [1] [The Stixel World - A Compact Medium-level Representation for Efficiently Modeling Three-dimensional Environments](https://www.mydlt.de/david/page/publications.html)

## Requirement
- OpenCV
- OpenMP (optional)

## How to build
```
$ git clone https://github.com/gishi523/multilayer-stixel-world.git
$ cd multilayer-stixel-world
$ mkdir build
$ cd build
$ cmake ../
$ make
```

## How to use
```
./stixelworld left-image-format right-image-format camera.xml
```
- left-image-format
    - the left image sequence
- right-image-format
    - the right image sequence
- camera.xml
    - the camera intrinsic and extrinsic parameters

### Example
 ```
./stixelworld images/img_c0_%09d.pgm images/img_c1_%09d.pgm ../camera.xml
```

### Data
- I tested this work using the Daimler Ground Truth Stixel Dataset
- http://www.6d-vision.com/ground-truth-stixel-dataset

## Author
gishi523