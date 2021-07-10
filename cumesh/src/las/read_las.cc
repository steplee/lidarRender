#include "read_las.h"

#include <Eigen/StdVector>
#include <Eigen/Core>

#include <laszip/laszip_api.h>
#include <istream>
#include <fstream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

static double taketime()
{
  return (double)(clock())/CLOCKS_PER_SEC;
}
static void dll_error(laszip_POINTER laszip)
{
  if (laszip) {
    laszip_CHAR* error;
    if (laszip_get_error(laszip, &error)) fprintf(stderr,"DLL ERROR: getting error messages\n");
    fprintf(stderr,"DLL ERROR MESSAGE: %s\n", error);
  }
}
static void byebye(bool error=false, bool wait=false, laszip_POINTER laszip=0)
{
  if (error) dll_error(laszip);
  if (wait) {
    fprintf(stderr,"<press ENTER>\n");
    getc(stdin);
  }
  exit(error);
}

Eigen::Vector4f getLasTlbr(const std::string &fname) {
    laszip_POINTER laszip_reader;
    if (laszip_create(&laszip_reader)) {
      fprintf(stderr,"DLL ERROR: creating laszip reader\n");
      byebye(true, false);
    }
    laszip_BOOL exploit = 1;
    if (laszip_exploit_spatial_index(laszip_reader, exploit)) {
      fprintf(stderr,"DLL ERROR: signaling laszip reader that spatial queries are coming for '%s'\n", fname.c_str());
      byebye(true, false, laszip_reader);
    }
    laszip_BOOL is_compressed = 0;
    if (laszip_open_reader(laszip_reader, fname.c_str(), &is_compressed)) {
      fprintf(stderr,"DLL ERROR: opening laszip reader for '%s'\n", fname.c_str());
      byebye(true, false, laszip_reader);
    }
    laszip_BOOL is_indexed = 0;
    laszip_BOOL is_appended = 0;
    if (laszip_has_spatial_index(laszip_reader, &is_indexed, &is_appended)) {
      fprintf(stderr,"DLL ERROR: checking laszip reader whether spatial indexing information is present for '%s'\n", fname.c_str());
      byebye(true, false, laszip_reader);
    }

    laszip_header* header;

    if (laszip_get_header_pointer(laszip_reader, &header)) {
      fprintf(stderr,"DLL ERROR: getting header pointer from laszip reader\n");
      byebye(true, false, laszip_reader);
    }

    Eigen::Vector4f out {
      header->min_x, header->min_y,
      header->max_x, header->max_y };
    return out;
}


// Deprecated: does not handle scale, also single floats are not okay for UTM.
#if 0
std::vector<LasPoint> read_las(const std::string& fname) {
  std::vector<LasPoint> out;

  double start_time = 0.0;
  start_time = taketime();

    //fprintf(stderr,"running EXAMPLE_FOUR (reading area-of-interest from a file exploiting possibly existing spatial indexing information)\n");

    // create the reader
    laszip_POINTER laszip_reader;
    if (laszip_create(&laszip_reader)) {
      fprintf(stderr,"DLL ERROR: creating laszip reader\n");
      byebye(true, false);
    }

    // signal that spatial queries are coming
    laszip_BOOL exploit = 1;
    if (laszip_exploit_spatial_index(laszip_reader, exploit)) {
      fprintf(stderr,"DLL ERROR: signaling laszip reader that spatial queries are coming for '%s'\n", fname.c_str());
      byebye(true, false, laszip_reader);
    }

    // open the reader
    laszip_BOOL is_compressed = 0;
    if (laszip_open_reader(laszip_reader, fname.c_str(), &is_compressed)) {
      fprintf(stderr,"DLL ERROR: opening laszip reader for '%s'\n", fname.c_str());
      byebye(true, false, laszip_reader);
    }

    //fprintf(stderr,"file '%s' is %scompressed\n", fname.c_str(), (is_compressed ? "" : "un"));

    // check whether spatial indexing information is available
    laszip_BOOL is_indexed = 0;
    laszip_BOOL is_appended = 0;
    if (laszip_has_spatial_index(laszip_reader, &is_indexed, &is_appended)) {
      fprintf(stderr,"DLL ERROR: checking laszip reader whether spatial indexing information is present for '%s'\n", fname.c_str());
      byebye(true, false, laszip_reader);
    }

    //fprintf(stderr,"file '%s' does %shave spatial indexing information\n", fname.c_str(), (is_indexed ? "" : "not "));

    // get a pointer to the header of the reader that was just populated
    laszip_header* header;

    if (laszip_get_header_pointer(laszip_reader, &header)) {
      fprintf(stderr,"DLL ERROR: getting header pointer from laszip reader\n");
      byebye(true, false, laszip_reader);
    }

    // how many points does the file have
    laszip_I64 npoints = (header->number_of_point_records ? header->number_of_point_records : header->extended_number_of_point_records);

    // report how many points the file has
    //fprintf(stderr,"file '%s' contains %I64d points\n", fname.c_str(), npoints);

    // create a rectangular box enclosing a subset of points at the center of the full bounding box
    //const laszip_F64 sub = 0.05;
    const laszip_F64 sub = 1;

    laszip_F64 mid_x = (header->min_x + header->max_x) / 2;
    laszip_F64 mid_y = (header->min_y + header->max_y) / 2;

    laszip_F64 range_x = header->max_x - header->min_x;
    laszip_F64 range_y = header->max_y - header->min_y;

    laszip_F64 sub_min_x = mid_x - sub * range_x;
    laszip_F64 sub_min_y = mid_y - sub * range_y;

    laszip_F64 sub_max_x = mid_x + sub * range_x;
    laszip_F64 sub_max_y = mid_y + sub * range_y;

    // request the reader to only read this specified rectangular subset of points
    laszip_BOOL is_empty = 0;
    if (laszip_inside_rectangle(laszip_reader, sub_min_x, sub_min_y, sub_max_x, sub_max_y, &is_empty))
    {
      fprintf(stderr,"DLL ERROR: requesting points inside of rectangle [%g,%g] (%g,%g) from laszip reader\n", sub_min_x, sub_min_y, sub_max_x, sub_max_y);
      byebye(true, false, laszip_reader);
    }

    // get a pointer to the points that will be read
    laszip_point* point;
    if (laszip_get_point_pointer(laszip_reader, &point)) {
      fprintf(stderr,"DLL ERROR: getting point pointer from laszip reader\n");
      byebye(true, false, laszip_reader);
    }

    // read the points
    laszip_BOOL is_done = 0;
    laszip_I64 p_count = 0;
    while (p_count < npoints)
    {
      // read a point
      if (laszip_read_inside_point(laszip_reader, &is_done)) {
        fprintf(stderr,"DLL ERROR: reading point %I64d\n", p_count);
        byebye(true, false, laszip_reader);
      }

      if (is_done) {
        break;
      }

      float x = point->X;
      float y = point->Y;
      float z = point->Z;
      //if (p_count % 1000 == 0) printf(" - point at %f %f %f.\n", x,y,z);
      //if (p_count % 1000 == 0) printf(" - point at %d %d %d.\n", point->X,point->Y,point->Z);

      out.push_back(LasPoint{x,y,z});


      p_count++;
    }

    //fprintf(stderr,"successfully read and written %I64d points\n", p_count);

    // close the reader
    if (laszip_close_reader(laszip_reader)) {
      fprintf(stderr,"DLL ERROR: closing laszip reader\n");
      byebye(true, false, laszip_reader);
    }

    // destroy the reader
    if (laszip_destroy(laszip_reader)) {
      fprintf(stderr,"DLL ERROR: destroying laszip reader\n");
      byebye(true, false);
    }
    //fprintf(stderr,"total time: %g sec for reading %scompressed\n", taketime()-start_time, (is_compressed ? "" : "un"));

    return out;
}
#endif

void read_las_aoi(const std::string& fname, const Eigen::Vector4d& aoiTlbr, std::vector<LasPoint>& out, int stride) {

  double start_time = 0.0;
  start_time = taketime();

    //fprintf(stderr,"running EXAMPLE_FOUR (reading area-of-interest from a file exploiting possibly existing spatial indexing information)\n");

    // create the reader
    laszip_POINTER laszip_reader;
    if (laszip_create(&laszip_reader)) {
      fprintf(stderr,"DLL ERROR: creating laszip reader\n");
      byebye(true, false);
    }

    // signal that spatial queries are coming
    ///*
    laszip_BOOL exploit = 1;
    if (laszip_exploit_spatial_index(laszip_reader, exploit)) {
      fprintf(stderr,"DLL ERROR: signaling laszip reader that spatial queries are coming for '%s'\n", fname.c_str());
      byebye(true, false, laszip_reader);
    }
    //*/

    // open the reader
    laszip_BOOL is_compressed = 0;
    if (laszip_open_reader(laszip_reader, fname.c_str(), &is_compressed)) {
      fprintf(stderr,"DLL ERROR: opening laszip reader for '%s'\n", fname.c_str());
      byebye(true, false, laszip_reader);
    }

    //fprintf(stderr,"file '%s' is %scompressed\n", fname.c_str(), (is_compressed ? "" : "un"));

    // check whether spatial indexing information is available
    ///*
    laszip_BOOL is_indexed = 0;
    laszip_BOOL is_appended = 0;
    if (laszip_has_spatial_index(laszip_reader, &is_indexed, &is_appended)) {
      fprintf(stderr,"DLL ERROR: checking laszip reader whether spatial indexing information is present for '%s'\n", fname.c_str());
      byebye(true, false, laszip_reader);
    }
    //fprintf(stderr,"file '%s' does %shave spatial indexing information\n", fname.c_str(), (is_indexed ? "" : "not "));
    //*/

    // get a pointer to the header of the reader that was just populated
    laszip_header* header;

    if (laszip_get_header_pointer(laszip_reader, &header)) {
      fprintf(stderr,"DLL ERROR: getting header pointer from laszip reader\n");
      byebye(true, false, laszip_reader);
    }

    double scaleFactorX = header->x_scale_factor;
    double scaleFactorY = header->y_scale_factor;
    double scaleFactorZ = header->z_scale_factor;
    double offsetX = header->x_offset;
    double offsetY = header->y_offset;
    double offsetZ = header->z_offset;

    // how many points does the file have
    laszip_I64 npoints = (header->number_of_point_records ? header->number_of_point_records : header->extended_number_of_point_records);

    out.reserve(out.size() + (npoints+stride-1)/stride);

    // report how many points the file has
    //fprintf(stderr,"file '%s' contains %I64d points\n", fname.c_str(), npoints);

    // request the reader to only read this specified rectangular subset of points
    laszip_BOOL is_empty = 0;
    if (laszip_inside_rectangle(laszip_reader, aoiTlbr(0), aoiTlbr(1), aoiTlbr(2), aoiTlbr(3), &is_empty))
    {
      fprintf(stderr,"DLL ERROR: requesting points inside of rectangle [%g,%g] (%g,%g) from laszip reader\n", aoiTlbr(0),aoiTlbr(1),aoiTlbr(2),aoiTlbr(3));
      byebye(true, false, laszip_reader);
    }

    // get a pointer to the points that will be read
    laszip_point* point;
    if (laszip_get_point_pointer(laszip_reader, &point)) {
      fprintf(stderr,"DLL ERROR: getting point pointer from laszip reader\n");
      byebye(true, false, laszip_reader);
    }

    // read the points
    laszip_BOOL is_done = 0;
    laszip_I64 p_count = 0;
    while (p_count < npoints)
    {
      // read a point
      if (laszip_read_inside_point(laszip_reader, &is_done)) {
        fprintf(stderr,"DLL ERROR: reading point %I64d\n", p_count);
        byebye(true, false, laszip_reader);
      }

      if (is_done) {
        break;
      }

      // NOTE: Not sure if offset/scale applied in correct order.
      double xx = point->X;
      double yy = point->Y;
      float z = point->Z * scaleFactorZ - offsetZ;
      //float x = static_cast<float>((xx - (aoiTlbr(0)-offsetX)) * scaleFactorX);
      //float y = static_cast<float>((yy - (aoiTlbr(1)-offsetY)) * scaleFactorY);
      float x = static_cast<float>((xx * scaleFactorX - (aoiTlbr(0)-offsetX)));
      float y = static_cast<float>((yy * scaleFactorY - (aoiTlbr(1)-offsetY)));

      //if (p_count % 10000 == 0)
        //printf(" - pt %lf %lf -> %f %f (offset %lf %lf) (scale %lf %lf)\n",
            //xx,yy, x,y, aoiTlbr(0), aoiTlbr(1), scaleFactorX, scaleFactorY);

      if (p_count % stride == 0)
        out.push_back(LasPoint{x,y,z});


      p_count++;
    }

    //fprintf(stderr,"successfully read and written %I64d points\n", p_count);

    // close the reader
    if (laszip_close_reader(laszip_reader)) {
      fprintf(stderr,"DLL ERROR: closing laszip reader\n");
      byebye(true, false, laszip_reader);
    }

    // destroy the reader
    if (laszip_destroy(laszip_reader)) {
      fprintf(stderr,"DLL ERROR: destroying laszip reader\n");
      byebye(true, false);
    }
    //fprintf(stderr,"total time: %g sec for reading %scompressed\n", taketime()-start_time, (is_compressed ? "" : "un"));
}
