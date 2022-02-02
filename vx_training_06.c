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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <math.h>
#include <stdio.h>
#include <VX/vx.h>

static int
populate_image (vx_image image, const unsigned char *img_data)
{
  int ret = -1;

  int width = 0;
  int height = 0;
  int channels = 3;
  vxQueryImage (image, VX_IMAGE_WIDTH, &width, sizeof (width));
  vxQueryImage (image, VX_IMAGE_WIDTH, &height, sizeof (height));

  vx_imagepatch_addressing_t layout = { width, height, channels,
      width*channels, 0, 0, 0, 0 };
  
  const vx_rectangle_t rect = { 0, 0, width, height };

  vx_status status = vxCopyImagePatch (image, &rect, 0, &layout, (void *)img_data,
      VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
  if (VX_SUCCESS != status) {
    fprintf (stderr, "vx-training: Unable to copy data into image: %d\n", status);
    ret = -1;
  } else {
    ret = 0;
  }

  return ret;
}

static int
dump_image (vx_image image, const char *path)
{
  int ret = -1;

  int width = 0;
  int height = 0;
  vxQueryImage (image, VX_IMAGE_WIDTH, &width, sizeof (width));
  vxQueryImage (image, VX_IMAGE_WIDTH, &height, sizeof (height));

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
    fprintf (stderr, "vx-training: Unable to map image for reading: %d\n", status);
    goto out;
  }

  if (0 == stbi_write_png (path, width, height, addr.stride_x, ptr, addr.stride_y)) {
    fprintf (stderr, "vx-training: Unable to write image to %s\n", path);
    goto unmap;
  }

  ret = 0;
  
 unmap:
  vxUnmapImagePatch (image, map_id);

 out:
  return ret;
}

static void VX_CALLBACK
context_log_callback(vx_context context, vx_reference ref, vx_status status,
    const vx_char string[])
{
  printf ("vx-training [dbg]: %s\n", string);
}

int
main (int argc, char *argv[])
{
  int ret = -1;

  const char *filename = "lena.png";
  if (argc >= 2) {
    filename = argv[1];
  }

  const char *outname = "out.png";
  if (argc >= 3) {
    outname = argv[2];
  }
  
  vx_context context = vxCreateContext ();

  vx_status status = vxGetStatus ((vx_reference)context);
  if (VX_SUCCESS != status) {
    fprintf (stderr, "vx-training: Unable to create context: %d\n", status);;
    goto free_context;
  }

  vx_bool reentrant = vx_false_e;
  vxRegisterLogCallback(context, context_log_callback, reentrant);

  int width = 0;
  int height = 0;
  int channels = 0;
  unsigned char *img_data = stbi_load (filename, &width, &height, &channels, 3);
  if (NULL == img_data) {
    fprintf (stderr, "vx-training: Unable to load image \"%s\"\n", filename);
    goto free_context;
  }
  
  vx_image in_image = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);

  status = vxGetStatus ((vx_reference)in_image);
  if (VX_SUCCESS != status) {
    fprintf (stderr, "vx-training: Unable to create input image: %d\n", status);
    goto free_in_img;
  }

  if (0 != populate_image (in_image, img_data)) {
    fprintf (stderr, "vx-training: Unable to populate image\n");
    goto free_in_img;
  }

  vx_image out_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);

  status = vxGetStatus ((vx_reference)out_image);
  if (VX_SUCCESS != status) {
    fprintf (stderr, "vx-training: Unable to create output image: %d\n", status);
    goto free_out_img;
  }

  vx_graph graph = vxCreateGraph (context);

  status = vxGetStatus ((vx_reference)graph);
  if (VX_SUCCESS != status) {
    fprintf (stderr, "vx-training: Unable to create graph: %d\n", status);
    goto free_graph;
  }

  vx_image intermediates[] = {
    vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8),
    vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8),
  };

  for (int i = 0; i < sizeof (intermediates)/sizeof (vx_image); i++) {
    status = vxGetStatus ((vx_reference)intermediates[i]);
    if (VX_SUCCESS != status) {
      fprintf (stderr, "vx-training: Unable to create virtual image[%d]: %d\n", i, status);
      goto free_intermediate;
    }
  }
  
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
  vx_float32 rad = 45.0*M_PI/180.0;
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

  vx_matrix matrix = vxCreateMatrix(context, VX_TYPE_FLOAT32, 2, 3);
  vxCopyMatrix(matrix, mat, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
  vx_enum interpolation = VX_INTERPOLATION_BILINEAR;
  
  vx_node nodes[] = {
    vxChannelExtractNode (graph, in_image, VX_CHANNEL_R, intermediates[0]),
    vxGaussian3x3Node (graph, intermediates[0], intermediates[1]),
    vxWarpAffineNode (graph, intermediates[1], matrix, interpolation, out_image)
  };

  for (int i = 0; i < sizeof (nodes)/sizeof(vx_node); i++) {
    status = vxGetStatus ((vx_reference)nodes[i]);
    if (VX_SUCCESS != status) {
      fprintf (stderr, "vx-training: Unable to create processing node[%d]: %d\n", i, status);
      goto free_node;
    }
  }

  status = vxVerifyGraph (graph);
  if (VX_SUCCESS != status) {
    fprintf (stderr, "vx-training: Graph validation failed: %d\n", status);
    goto free_node;
  }

  status = vxProcessGraph (graph);
  if (VX_SUCCESS != status) {
    fprintf (stderr, "vx-training: Error processing the graph: %d\n", status);
    goto free_node;
  }

  if (0 != dump_image (out_image, outname)) {
    fprintf (stderr, "vx-training: Error writing output image to \"%s\"\n", outname);
    goto free_node;
  }

  dump_image (in_image, "test.png");
  
  ret = 0;

 free_node:
  for (int i = 0; i < sizeof (nodes)/sizeof(vx_node); i++) {
    vxReleaseNode (&nodes[i]);
  }

 free_intermediate:
  for (int i = 0; i < sizeof (intermediates)/sizeof(vx_image); i++) {
    vxReleaseImage (&intermediates[i]);
  }
  
 free_graph:
  vxReleaseGraph (&graph);

 free_out_img:
  vxReleaseImage (&out_image);
  
 free_in_img:
  vxReleaseImage (&in_image);

 free_img_data:
  free (img_data);
  
 free_context:
  vxReleaseContext (&context);

 out:
  return ret;
}
