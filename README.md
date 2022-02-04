# OpenVX Training
> A practical and incremental approach to learn OpenVX

This project presents a series of simple examples that aim to introduce basic OpenVX concepts. It was developed as a companion to the *OpenVX Training* presentation.

## Preparing the Project

### Dependencies

In order to build the project you'll need the following dependencies:

* **Makefile, C and C++ compilers**:
In Debian based systems (like Ubuntu) you may run the following commands:
```bash
sudo apt install build-essential
```

* **OpenCV**

In Debian based systems (like Ubuntu) you may run the following commands:
```bash
sudo apt install libopencv-dev
```

* **A working OpenVX implementation**: 

Normally this is already included in your system. If not, you may install the [reference implmenentation from Khronos](https://github.com/KhronosGroup/OpenVX-sample-impl).


### Building

The project is based on a simple Makefile. To build it, run the following commands:

```bash
make
```

### Running Examples

To run the examples, simply invoke the executables produced during the build process as in:
```bash
./vx_training_01
```

Of course, the **00** may be changed to match any example as desired. Refer to the following sections for a description of each example.

## Examples Description

The following table summarizes the examples available in the project. They were numbered to, ideally, be consumed in order.

| Example | Description | 1st Arg | 2nd Arg |
|---------|-------------|----------|----------|
| vx_training_01 | Creates a context and shows the overall VX development pattern. | | |
| vx_training_02 | Creates an image and shows how to access the underlying memory in order to write to it. | Image path (defaults to *lena.png*) | |
| vx_training_03 | Creates a graph with a *Channel Extract* node and an output image. Does ot process the graph yet. | Image path (defaults to *lena.png*) | |
| vx_training_04 | Verifies and executes the graph. Saves the data from the output image into a PNG file. | Image path (defaults to *lena.png*) | Image path (defaults to *out.png*)|
| vx_training_05 | Adds a second *Gaussian Kernel* node and connects it to the first one using a virtual image. | Image path (defaults to *lena.png*) | Image path (defaults to *out.png*)|
| vx_training_06 | Adds a third *Warp Affine* node and shows how to pass in a *vx_reference* as a parameter. | Image path (defaults to *lena.png*) | Image path (defaults to *out.png*)|
| vx_training_07 | First example in C++. Shows how to continuously process the graph and vary a parameter with each execution. Displays the result in a window. | Image path (defaults to *lena.png*) | |
| vx_training_08 | Shows how to enable performance measurements. | Image path (defaults to *lena.png*) | |
| vx_training_09 | Modifies the previous example to be executed in a pipelining mode. | Image path (defaults to *lena.png*) | |
| vx_training_10 | Modifies the previous example to be executed in a batching mode. | Image path (defaults to *lena.png*) | |

## Questions

If you run into any problem or have any question, please do [contact us](mailto:support@ridgerun.com).






