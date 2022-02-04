/* Copyright (C) 2022 RidgeRun, LLC (http://www.ridgerun.com)
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to RidgeRun,
 * LLC.  No part of this program may be photocopied, reproduced or translated
 * into another programming language without prior written consent of
 * RidgeRun, LLC.  The user is free to modify the source code after obtaining
 * a software license from RidgeRun.  All source code changes must be provided
 * back to RidgeRun without any encumbrance.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>
#include <VX/vx_khr_pipelining.h>
#include <VX/vx.h>


template<typename T>
static std::shared_ptr<T>
smart_ref (T *ptr)
{
  return std::shared_ptr<T> (ptr, [](T *ptr) {
    vxReleaseReference ((vx_reference *)&ptr);
  });
}

static int
populate_image (vx_image image, const unsigned char *img_data)
{
  int ret = -1;

  vx_uint32 width = 0;
  vx_uint32 height = 0;
  vx_int32 channels = 3;
  vxQueryImage (image, VX_IMAGE_WIDTH, &width, sizeof (width));
  vxQueryImage (image, VX_IMAGE_HEIGHT, &height, sizeof (height));

  vx_imagepatch_addressing_t layout = { width, height, channels,
    static_cast<vx_int32>(width*channels), 0, 0, 0, 0 };
  
  const vx_rectangle_t rect = { 0, 0, width, height };

  vx_status status = vxCopyImagePatch (image, &rect, 0, &layout, (void *)img_data,
      VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
  if (VX_SUCCESS != status) {
    std::cerr << "vx-training: Unable to copy data into image: " << status << std::endl;
    ret = -1;
  } else {
    ret = 0;
  }

  return ret;
}

static int
show_image (vx_image image)
{
  int ret = -1;

  vx_uint32 width = 0;
  vx_uint32 height = 0;
  vxQueryImage (image, VX_IMAGE_WIDTH, &width, sizeof (width));
  vxQueryImage (image, VX_IMAGE_HEIGHT, &height, sizeof (height));

  const vx_rectangle_t rect = { 0, 0, width, height };
  vx_uint32 plane = 0;
  vx_map_id map_id = 0;
  vx_imagepatch_addressing_t addr =  { 0 };
  unsigned char *ptr = 0;
  vx_enum usage = VX_READ_ONLY;
  vx_enum mem_type = VX_MEMORY_TYPE_HOST;
  vx_uint32 flags = VX_NOGAP_X;
  vx_status status = vxMapImagePatch (image, &rect, plane, &map_id, &addr, (void **)&ptr, usage, mem_type, flags);
  if (VX_SUCCESS != status) {
    std::cerr << "vx-training: Unable to map image for reading:" << status << std::endl;
    ret = -1;
  } else {
    cv::Mat mat (height, width, CV_8UC1, ptr, addr.stride_y);
    cv::imshow ("Processed image", mat);
    ret = 0;
  }

  vxUnmapImagePatch (image, map_id);

  return ret;
}

static void VX_CALLBACK
context_log_callback(vx_context context, vx_reference ref, vx_status status,
    const vx_char string[])
{
  std::cout << "vx-training [dbg]: " << string << std::endl;
}

static vx_status
enqueue_input(vx_graph graph, vx_image image, unsigned char *data)
{
  vx_uint32 parameter_in = 0;

  if (0 != populate_image (image, data)) {
    return VX_FAILURE;
  }

  return vxGraphParameterEnqueueReadyRef(graph, parameter_in,
      (vx_reference*)&image, 1);
}

static vx_status
dequeue_input(vx_graph graph, vx_image *image)
{
  vx_uint32 num_refs;
  vx_uint32 parameter_in = 1;

  *image = NULL;

  return  vxGraphParameterDequeueDoneRef(graph, parameter_in,
      (vx_reference*)image, 1, &num_refs);
}

static vx_status
enqueue_output(vx_graph graph, vx_image image)
{
  vx_uint32 parameter_out = 1;

  return vxGraphParameterEnqueueReadyRef(graph, parameter_out,
      (vx_reference*)&image, 1);
}

static vx_bool
is_output_available(vx_graph graph)
{
  vx_uint32 num_refs;
  vx_uint32 parameter_out = 1;

  vxGraphParameterCheckDoneRef(graph, parameter_out, &num_refs);

  return (num_refs > 0);
}

static vx_status
dequeue_output(vx_graph graph, vx_image *image)
{
  vx_uint32 num_refs;
  vx_uint32 parameter_out = 1;
  vx_status status;

  *image = NULL;

  /* Get output reference and consume new data,
   * waits until a reference is available
   */
  status = vxGraphParameterDequeueDoneRef(graph, parameter_out,
      (vx_reference*)image, 1, &num_refs);
  if (status) {
    return status;
  }

  if(NULL != *image) {
    status = show_image (*image);
  }

  return status;
}

int
main (int argc, char *argv[])
{
  const char *filename = "lena.png";
  if (argc >= 2) {
    filename = argv[1];
  }

  const char *outname = "out.png";
  if (argc >= 3) {
    outname = argv[2];
  }
  
  auto context = smart_ref (vxCreateContext ());

  vx_status status = vxGetStatus ((vx_reference)context.get ());
  if (VX_SUCCESS != status) {
    std::cerr << "vx-training: Unable to create context:" << status << std::endl;
    return -1;
  }

  vx_bool reentrant = vx_false_e;
  vxRegisterLogCallback(context.get (), context_log_callback, reentrant);

  int width = 0;
  int height = 0;
  int channels = 0;
  auto img_data = std::shared_ptr<unsigned char>(stbi_load (filename, &width, &height, &channels, 3), free);
  if (NULL == img_data) {
    std::cerr << "vx-training: Unable to load image " << filename << std::endl;
    return -1;
  }

  int num_images = 2;
  std::vector <std::shared_ptr<_vx_image>> in_images;
  for (int i= 0; i < num_images; i++) {
    auto in_image = smart_ref(vxCreateImage(context.get (), width, height, VX_DF_IMAGE_RGB));
    status = vxGetStatus ((vx_reference)in_image.get ());
    if (VX_SUCCESS != status) {
      std::cerr << "vx-training: Unable to create input image: " << status << std::endl;
      return -1;
    }
    
    in_images.push_back (in_image);
  }

  std::vector <std::shared_ptr<_vx_image>> out_images;
  for (int i= 0; i < num_images; i++) {
    auto out_image = smart_ref(vxCreateImage(context.get (), width, height, VX_DF_IMAGE_U8));
    status = vxGetStatus ((vx_reference)out_image.get ());
    if (VX_SUCCESS != status) {
      std::cerr << "vx-training: Unable to create input image: " << status << std::endl;
      return -1;
    }
    
    out_images.push_back (out_image);
  }
  
  auto graph = smart_ref (vxCreateGraph (context.get ()));

  status = vxGetStatus ((vx_reference)graph.get ());
  if (VX_SUCCESS != status) {
    std::cerr << "vx-training: Unable to create graph: " << status << std::endl;
    return -1;
  }

  auto intermediate = smart_ref (vxCreateVirtualImage(graph.get (), width, height, VX_DF_IMAGE_U8));

  status = vxGetStatus ((vx_reference)intermediate.get ());
  if (VX_SUCCESS != status) {
    std::cerr << "vx-training: Unable to create virtual image: " << status << std::endl;
    return -1;
  }

  auto matrix = smart_ref (vxCreateMatrix(context.get (), VX_TYPE_FLOAT32, 2, 3));
  vx_enum interpolation = VX_INTERPOLATION_BILINEAR;
  
  std::vector<std::shared_ptr<_vx_node>> nodes = {
    // Input image will now be a parameter
    smart_ref (vxChannelExtractNode (graph.get (), in_images[0].get (), VX_CHANNEL_R, intermediate.get ())),
    // Ouput image will now be a parameters
    smart_ref (vxWarpAffineNode (graph.get (), intermediate.get (), matrix.get (), interpolation, out_images[0].get ()))
  };

  for (auto &node: nodes) {
    status = vxGetStatus ((vx_reference)node.get ());
    
    if (VX_SUCCESS != status) {
      std::cerr << "vx-training: Unable to create processing node: " << status << std::endl;
      return -1;
    }
  }

  vx_parameter parameter = vxGetParameterByIndex(nodes[0].get(), 0);
  vxAddParameterToGraph(graph.get (), parameter);
  vxReleaseParameter(&parameter);

  parameter = vxGetParameterByIndex(nodes[1].get(), 3);
  vxAddParameterToGraph(graph.get (), parameter);
  vxReleaseParameter(&parameter);

  vx_image in_refs[] = {
    in_images[0].get (),
    in_images[1].get (),
  };

  vx_image out_refs[] = {
    out_images[0].get (),
    out_images[1].get (),
  };
  
  std::vector<vx_graph_parameter_queue_params_t> queue_params_list(2);
  queue_params_list[0].graph_parameter_index = 0;
  queue_params_list[0].refs_list_size = in_images.size();
  queue_params_list[0].refs_list = (vx_reference*)&in_refs[0];
  queue_params_list[1].graph_parameter_index = 1;
  queue_params_list[1].refs_list_size = out_images.size();
  queue_params_list[1].refs_list = (vx_reference*)&out_refs[0];

  vxSetGraphScheduleConfig(graph.get (), VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO,
      queue_params_list.size(), queue_params_list.data());
  
  status = vxVerifyGraph (graph.get ());
  if (VX_SUCCESS != status) {
    std::cerr << "vx-training: Graph validation failed: " << status << std::endl;
    return -1;
  }

  for (auto &img: in_images) {
    vx_status status = enqueue_input (graph.get (), img.get (), img_data.get ()); 
    if (VX_SUCCESS != status) {
      std::cerr << "vx-training: Unable to enqueue input buffer: " << status << std::endl;
      return -1;
    }
  }

  for (auto &img: out_images) {
    vx_status status = enqueue_output (graph.get (), img.get ()); 
    if (VX_SUCCESS != status) {
      std::cerr << "vx-training: Unable to enqueue input buffer: " << status << std::endl;
      return -1;
    }
  }
  
  cv::namedWindow ("Processed image", cv::WINDOW_AUTOSIZE);

  vx_image in_image;
  vx_image out_image;
  vx_float32 angle = 0.0;
  while (-1 == cv::waitKey(30)) {
    /*
      Images in OpenVX have the origin of the coordinate system in the
      upper left corner. Images will rotate around the origin. To rotate
      around the center we need to translate the image so that the origin
      matches the center, rotate, and translate back. In linear algebra,
      this is achieved by multiplying transformation matrices, were 
      they are applied from right to left.
      
          [1 0 w/2] [ cos(a) -sin(a) 0 ] [1 0 -w/2]
      R = [0 1 h/2] [ sin(a)  sin(a) 0 ] [0 1 -h/2]
          [0 0   1] [      0       0 1 ] [0 0    1]

          [ cos(a) -sin(a) -cos(a)*w/2 + sin(a)*h/2 + w/2 ]
      R = [ sin(a)  sin(a) -cos(a)*h/2 - sin(a)*h/2 + h/2 ]
    	  [      0       0                              1 ]
    */
    vx_float32 rad = angle*M_PI/180.0;
    angle++;
    /* Rotation only matrix */
    //vx_float32 mat[3][2] = {
    //  {cos (rad), sin (rad)},
    //  {-sin (rad), cos (rad)},
    //  {0, 0},
    //};
    
    /* Translate + rotate + translate back */
    vx_float32 mat[3][2] = {
      {cos (rad), sin (rad)},
      {-sin (rad), cos (rad)},
      {-cos (rad)*width/2 + sin (rad)*height/2 + width/2, -cos (rad)*height/2 - sin (rad)*width/2 + height/2},
    };
    vxCopyMatrix(matrix.get (), mat, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    
    /* wait for input to be available, dequeue it -
     * BLOCKs until input can be dequeued
     */
    vx_status status = dequeue_input(graph.get (), &in_image);
    if (VX_SUCCESS != status) {
      std::cerr << "vx-training: Unable to dequeue input buffer: " << status << std::endl;
      return -1;
    }

    /* wait for input to be available, dequeue it -
     * BLOCKs until input can be dequeued
     */
    status = dequeue_output(graph.get (), &out_image);
    if (VX_SUCCESS != status) {
      std::cerr << "vx-training: Unable to dequeue output buffer: " << status << std::endl;
      return -1;
    }

    /* recycle output */
    status = enqueue_output(graph.get (), out_image);
    if (VX_SUCCESS != status) {
      std::cerr << "vx-training: Unable to enqueue input buffer: " << status << std::endl;
      return -1;
    }

    /* recycle input - fill new data and re-enqueue*/
    status = enqueue_input(graph.get (), in_image, img_data.get ());
    if (VX_SUCCESS != status) {
      std::cerr << "vx-training: Unable to enqueue output buffer: " << status << std::endl;
      return -1;
    }
  }

  /*
   * wait until all previous graph executions have completed
   */
  vxWaitGraph(graph.get ());

  /* flush output references, only required
   * if need to consume last few references
   */
  while (is_output_available(graph.get ())) {
    dequeue_output(graph.get (), &out_image);
  }

  vx_perf_t perf;
  vxQueryGraph(graph.get (), VX_GRAPH_PERFORMANCE, &perf, sizeof(perf));

  std::cout << "Graph performance:" << std::endl;
  std::cout << "\tLast measurement: " << perf.tmp << std::endl;
  std::cout << "\tFirst measurement of the set: " << perf.beg << std::endl;
  std::cout << "\tLast measurement of the set: " << perf.end << std::endl;
  std::cout << "\tAverage of durations: " << perf.avg << std::endl;
  std::cout << "\tMinimum of durations: " << perf.min << std::endl;
  std::cout << "\tMaximum of durations: " << perf.max << std::endl;
  std::cout << "\tSum of durations: " << perf.sum << std::endl;
  std::cout << "\tNumber of measurements: " << perf.num << std::endl;
  std::cout << "\t---" << std::endl;
  
  for (auto &node: nodes) {
    vx_perf_t perf;
    vxQueryNode(node.get (), VX_NODE_PERFORMANCE, &perf, sizeof(perf));

    std::cout << "Node performance:" << std::endl;
    std::cout << "\tLast measurement: " << perf.tmp << std::endl;
    std::cout << "\tFirst measurement of the set: " << perf.beg << std::endl;
    std::cout << "\tLast measurement of the set: " << perf.end << std::endl;
    std::cout << "\tAverage of durations: " << perf.avg << std::endl;
    std::cout << "\tMinimum of durations: " << perf.min << std::endl;
    std::cout << "\tMaximum of durations: " << perf.max << std::endl;
    std::cout << "\tSum of durations: " << perf.sum << std::endl;
    std::cout << "\tNumber of measurements: " << perf.num << std::endl;
    std::cout << "\t---" << std::endl;
  }
  
  cv::destroyAllWindows ();

  return 0;
}
