#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef struct vec3
{
  float x;
  float y;
  float z;
} __attribute__((packed)) vec3;

inline vec3 vec3_sub(const vec3 *a, const vec3 *b)
{
  return (vec3){a->x - b->x, a->y - b->y, a->z - b->z};
}
inline vec3 vec3_scale(const vec3 *a, const float b)
{
  return (vec3){a->x * b, a->y * b, a->z * b};
}
inline float vec3_dot(const vec3 *a, const vec3 *b)
{
  return a->x * b->x + a->y * b->y + a->z * b->z;
}
inline vec3 vec3_negate(const vec3 *v)
{
  return (vec3){-v->x, -v->y, -v->z};
}
inline float vec3_norm(const vec3 *v)
{
  return sqrtf(v->x * v->x + v->y * v->y + v->z * v->z);
}
inline void vec3_normalize(vec3 *v)
{
  float tmp = 1.f / vec3_norm(v);
  v->x *= tmp;
  v->y *= tmp;
  v->z *= tmp;
  return;
}

typedef struct material
{
  float refractive_index; // how much do transparent objects bend light
  float albedo[4]; // %diffuse color, specular reflection(high=smooth), reflectiveness, transparency
  vec3 diffuse_color; // material color
  float specular_exponent; // how iluminated by white light
  float roughness; // 0-1 (smooth-rough)
} material;

#define MAT_DEFAULT    {1.0, {2.0, 0.0, 0.0, 0.0},  {0.0, 0.0, 0.0}, 0.0,    0}

#define MAT_WHITE_WALL {1.0, {2.0, 0.0, 0.0, 0.0},  {0.7, 0.7, 0.7}, 0.0,    0}
#define MAT_RED_WALL   {1.0, {2.0, 0.0, 0.0, 0.0},  {0.7, 0.0, 0.0}, 0.0,    0}
#define MAT_GREEN_WALL {1.0, {2.0, 0.0, 0.0, 0.0},  {0.0, 0.7, 0.0}, 0.0,    0}
#define MAT_CYAN_WALL  {1.0, {2.0, 0.0, 0.0, 0.0},  {0.2, 0.3, 0.3}, 0.0,    0}

#define MAT_IVORY      {1.0, {0.9, 0.5, 0.1, 0.0},  {0.4, 0.4, 0.3}, 50.0,   0}
#define MAT_GLASS      {1.5, {0.0, 0.9, 0.1, 0.8},  {0.6, 0.7, 0.8}, 125.0,  0}
#define MAT_RED_RUBBER {1.0, {1.4, 0.3, 0.0, 0.0},  {0.3, 0.1, 0.1}, 10.0,   0}
#define MAT_GRE_RUBBER {1.0, {1.4, 0.3, 0.0, 0.0},  {0.1, 0.3, 0.1}, 10.0,   0}
#define MAT_BLU_RUBBER {1.0, {1.4, 0.3, 0.0, 0.0},  {0.1, 0.1, 0.3}, 10.0,   0}
#define MAT_MIRROR     {1.0, {0.0, 16.0, 0.8, 0.0}, {1.0, 1.0, 1.0}, 1425.0, 0}
#define MAT_METAL      {1.0, {0.9, 7.0, 0.4, 0.0},  {0.3, 0.3, 0.3}, 10.0,   0.05}

typedef struct sphere
{
  vec3 center;
  float radius;
  material mat;
} sphere;

static const sphere scene_spheres[] = {
    {{-3.0,  -3.5, -16.0}, 2.0, MAT_IVORY},
    {{-1.0, -3.5, -12.0}, 2.0, MAT_GLASS},
    {{ 1.5, -2.5, -18.0}, 3.0, MAT_RED_RUBBER},
    {{ 7.0,  1.0, -16.0}, 3.5, MAT_MIRROR}};

static const vec3 scene_lights[] = {
    {-20, 20,  20},
    { 30, 50, -25},
    { 30, 20,  30}};

/* Calculates a rays reflection direction.
 */
inline vec3 reflect(const vec3 *I, const vec3 *N)
{
  float tmp = 2.f * vec3_dot(I, N);
  return (vec3){I->x - (N->x * tmp),
                I->y - (N->y * tmp),
                I->z - (N->z * tmp)};
}

/* Calculates a rays refraction direction using Snell's law.
 */
vec3 refract(const vec3 *I, const vec3 *N, float eta_t, float eta_i)
{ 
  float cosi = -fmax(-1.f, fmin(1.f, vec3_dot(I, N)));
  if (cosi < 0)
  {
    vec3 N_neg = vec3_negate(N);
    cosi = -fmax(-1.f, fmin(1.f, vec3_dot(I, &N_neg)));
    float eta = eta_t / eta_i;
    float k = 1.f - eta * eta * (1.f - cosi * cosi);
    float tmp = eta * cosi - sqrtf(k);
    // k < 0 = total reflection, no ray to refract, return has no physical meaning.
    return k < 0 ? (vec3){1.f, 0.f, 0.f} : (vec3){I->x * eta + N_neg.x * tmp, 
                                                  I->y * eta + N_neg.y * tmp, 
                                                  I->z * eta + N_neg.z * tmp};
  }
  float eta = eta_i / eta_t;
  float k = 1.f - eta * eta * (1.f - cosi * cosi);
  float tmp = eta * cosi - sqrtf(k);
  // k < 0 = total reflection, no ray to refract, return has no physical meaning.
  return k < 0 ? (vec3){1.f, 0.f, 0.f} : (vec3){I->x * eta + N->x * tmp, 
                                                I->y * eta + N->y * tmp, 
                                                I->z * eta + N->z * tmp};
}

typedef struct sphere_hit
{
  int hit; // boolean to indicate if there is an intersection
  float dist; // distance at which the intersection occurs
} sphere_hit;

/* Returns a struct indicating if a given sphere was intersected by a given ray,
 * and the distance to that sphere.
 */
sphere_hit ray_sphere_intersect(const vec3 *orig, const vec3 *dir, const sphere *s)
{
  vec3 L = vec3_sub(&s->center, orig);
  float tca = vec3_dot(&L, dir);
  float d2 = vec3_dot(&L, &L) - tca * tca;
  float r2 = s->radius * s->radius;
  if (d2 > r2)
  { // no intersection found
    return (sphere_hit){0, 0.f};
  }
  float thc = sqrtf(r2 - d2);
  float t0 = tca - thc;
  float t1 = tca + thc;
  // offset the original point by .001 to avoid occlusion by the object itself
  if (t0 > 0.001f)
    return (sphere_hit){1, t0};
  if (t1 > 0.001f)
    return (sphere_hit){1, t1};
  return (sphere_hit){0, 0.f};
}

typedef struct scene_hit
{
  int hit; // boolean to indicate if there is an intersection
  vec3 point;
  vec3 normal; // normal to the surface
  material mat; // material of surface
} scene_hit;

/* Returns a struct indicating the closest object a ray intersects with in the
 * scene.
 */
scene_hit scene_intersect(const vec3 *orig, const vec3 *dir)
{
  vec3 pt, N;
  material mat = MAT_DEFAULT;

  float nearest_dist = 1e10f;

  float d = -(orig->y + 6) / dir->y;
  vec3 p = {orig->x + dir->x * d, orig->y + dir->y * d, orig->z + dir->z * d};
  if (d > .001f && d < nearest_dist /*&& fabsf(p.x) < 100 && p.z < -1 && p.z > -300*/)
  {
    nearest_dist = d;
    pt = p;
    N = (vec3){0.f, 1.f, 0.f};
    mat.albedo[1] = 0.3;
    mat.albedo[2] = 0.2;
    mat.diffuse_color = ((int)(.2*pt.x+1000.f) + (int)(.2*pt.z+1000.f)) & 1 ? (vec3){.5f, .5f, .5f} : (vec3){.0f, .0f, .0f};
  }

  sphere_hit r;
  for (int i = 0; i < sizeof(scene_spheres)/sizeof(scene_spheres[0]); ++i)
  {
    r = ray_sphere_intersect(orig, dir, &scene_spheres[i]);
    if (!r.hit || r.dist > nearest_dist)
      continue;
    nearest_dist = r.dist;
    pt = (vec3){orig->x + dir->x * nearest_dist, 
                orig->y + dir->y * nearest_dist, 
                orig->z + dir->z * nearest_dist};
    N = vec3_sub(&pt, &scene_spheres[i].center);
    vec3_normalize(&N);
    mat = scene_spheres[i].mat;
  }

  return (scene_hit){nearest_dist < 1000.f, pt, N, mat};
}

/* Return the color of a pixel by casting a ray.
 */
vec3 cast_ray(const vec3 *orig, const vec3 *dir, const int depth)
{
  scene_hit r = scene_intersect(orig, dir);
  if (depth > 4 || !r.hit)
  {
    return (vec3){0.54, 0.81, 0.94}; // background color
  }

  vec3 reflect_dir = reflect(dir, &r.normal);
  vec3_normalize(&reflect_dir);
  vec3 refract_dir = refract(dir, &r.normal, r.mat.refractive_index, 1.f);
  vec3_normalize(&refract_dir);
  vec3 reflect_color = cast_ray(&r.point, &reflect_dir, depth + 1);
  vec3 refract_color = cast_ray(&r.point, &refract_dir, depth + 1);

  float diffuse_light_intensity = 0.f;
  float specular_light_intensity = 0.f;
  
  vec3 light_dir;
  vec3 tmp1, tmp2;
  scene_hit lr;
  for (int i = 0; i < sizeof(scene_lights)/sizeof(scene_lights[0]); ++i)
  { // check if the point is in the shadow of the light
    light_dir = vec3_sub(&scene_lights[i], &r.point);
    vec3_normalize(&light_dir);
    lr = scene_intersect(&r.point, &light_dir);
    tmp1 = vec3_sub(&lr.point, &r.point);
    tmp2 = vec3_sub(&scene_lights[i], &r.point);
    if (lr.hit && vec3_norm(&tmp1) < vec3_norm(&tmp2))
      continue;
    diffuse_light_intensity += fmax(0.f, vec3_dot(&light_dir, &r.normal));
    vec3 neg_light_dir = vec3_negate(&light_dir);
    vec3 reflect_dir = reflect(&neg_light_dir, &r.normal);
    specular_light_intensity += pow(fmax(0.f, -vec3_dot(&reflect_dir, dir)), r.mat.specular_exponent);
  }
  
  vec3 pixel; // {R, G, B}
  float tmp3 = diffuse_light_intensity * r.mat.albedo[0];
  float tmp4 = .1f * specular_light_intensity * r.mat.albedo[1];
  pixel.x = r.mat.diffuse_color.x * tmp3 + tmp4
            + reflect_color.x * r.mat.albedo[2] 
            + refract_color.x * r.mat.albedo[3];
  pixel.y = r.mat.diffuse_color.y * tmp3 + tmp4
            + reflect_color.y * r.mat.albedo[2] 
            + refract_color.y * r.mat.albedo[3];
  pixel.z = r.mat.diffuse_color.z * tmp3 + tmp4 
            + reflect_color.z * r.mat.albedo[2] 
            + refract_color.z * r.mat.albedo[3];
  return pixel;
}

void usage(const char *prog_name)
{
  fprintf(stderr, "usage: %s <WIDTH> <HEIGHT> <N>\n", prog_name);
  fprintf(stderr, "  WIDTH   horizontal resolution\n");
  fprintf(stderr, "  HEIGHT  vertical resolution\n");
  fprintf(stderr, "  N       number of processors\n");
  exit(1);
}

int main(int argc, char *argv[])
{
  if (argc != 1 && argc != 4)
  {
    usage(argv[0]);
  }
  // default options
  int user_width = 3840; //7680; //3840; //1920;
  int user_height = 2160; //4320; //2160; //1080;
  int user_thread_count = omp_get_num_threads();
  if (argc == 4)
  {
    user_width = atoi(argv[1]);
    user_height = atoi(argv[2]);
    user_thread_count = atoi(argv[3]);
  }
  if (user_width < 1 || user_height < 1 || user_thread_count < 1)
  {
    usage(argv[0]);
  }

  const int thread_count = user_thread_count;
  const int width = user_width;
  const int height = user_height;
  const int buf_size = width * height;
  const float fov = 1.0472; // 60 degrees field of view in radians
  vec3 *framebuffer = malloc(buf_size * sizeof(vec3));
  vec3 origin = (vec3){0.f, 0.f, 0.f};

  struct timeval start, end;
  double time_taken = 0;
  gettimeofday(&start, NULL); // start timer

  #pragma omp parallel for
  for (int pix = 0; pix < buf_size; ++pix)
  {
    vec3 dir;
    dir.x = (pix % width + .5f) - width / 2.f;
    dir.y = -(pix / width + .5f) + height / 2.f; // this flips the image at the same time
    dir.z = -height / (2.f * tanf(fov / 2.f));
    vec3_normalize(&dir);
    framebuffer[pix] = cast_ray(&origin, &dir, 0);
  }

  gettimeofday(&end, NULL); // stop timer
  time_taken = end.tv_sec * 1e3 + end.tv_usec / 1e3 -
               start.tv_sec * 1e3 - start.tv_usec / 1e3; // in ms
  printf("Execution Time: %dms\n", (int) time_taken);

  // write framebuffer to output file.
  FILE *fp = fopen("output.ppm", "wb");
  fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (int i = 0; i < buf_size; ++i) //vec3 &color : framebuffer)
    {
      float max = fmax(1.f, fmax(framebuffer[i].x, 
                                   fmax(framebuffer[i].y, framebuffer[i].z)));
      static unsigned char color[3];
      color[0] = (char)(255 * framebuffer[i].x / max); // red
      color[1] = (char)(255 * framebuffer[i].y / max); // green
      color[2] = (char)(255 * framebuffer[i].z / max); // blue
      fwrite(color, 1, 3 * sizeof(char), fp);
  }
  fclose(fp);

  free(framebuffer);

  return 0;
}
