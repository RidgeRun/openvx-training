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

#include <stdio.h>
#include <VX/vx.h>

static void VX_CALLBACK
context_log_callback(vx_context context, vx_reference ref, vx_status status,
    const vx_char string[])
{
  printf ("vx-training: %s\n", string);
}

int
main (int argc, char *argv[])
{
  int ret = -1;

  const char *filename = "lena.png";
  if (argc >= 2) {
    filename = argv[1];
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

  vx_image image = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);

  status = vxGetStatus ((vx_reference)image);
  if (VX_SUCCESS != status) {
    fprintf (stderr, "vx-training: Unable to create image: %d\n", status);
    goto free_img;
  }

  const vx_rectangle_t rect = { 0, 0, width, height };
  vx_uint32 plane = 0;
  vx_map_id map_id = 0;
  vx_imagepatch_addressing_t addr =  { 0 };
  unsigned char *ptr = 0;
  vx_enum usage = VX_READ_AND_WRITE;
  vx_enum mem_type = VX_MEMORY_TYPE_HOST;
  vx_uint32 flags = VX_NOGAP_X;
  status = vxMapImagePatch (image, &rect, plane, &map_id, &addr, (void **)&ptr, usage, mem_type, flags);
  if (VX_SUCCESS != status) {
    fprintf (stderr, "vx-training: Unable to map image for writing: %d\n", status);
    goto free_img;
  }

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      ptr[r*addr.stride_y + c*addr.stride_x + 0] = img_data[channels*(r*width + c) + 0];
      ptr[r*addr.stride_y + c*addr.stride_x + 1] = img_data[channels*(r*width + c) + 1];
      ptr[r*addr.stride_y + c*addr.stride_x + 2] = img_data[channels*(r*width + c) + 2];
    }
  }
  
  status = vxUnmapImagePatch (image, map_id);
  if (VX_SUCCESS != status) {
    fprintf (stderr, "vx-training: Unable to unmap image: %d\n", status);
    goto free_img;
  }
  
  ret = 0;

 free_img:
  vxReleaseImage (&image);
  
 free_img_data:
  free (img_data);
  
 free_context:
  vxReleaseContext (&context);

 out:
  return ret;
}
