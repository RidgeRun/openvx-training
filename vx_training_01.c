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
  int ret = 0;
  vx_context context = vxCreateContext ();

  vx_status status = vxGetStatus ((vx_reference)context);
  if (VX_SUCCESS != status) {
    fprintf (stderr, "vx-training: Unable to create context: %d\n", status);;
    ret = -1;
    goto out;
  }

  vx_bool reentrant = vx_false_e;
  vxRegisterLogCallback(context, context_log_callback, reentrant);

 out:
  vxReleaseContext (&context);
  return ret;
}
