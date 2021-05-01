#include <Eigen/Core>

struct SortedOctree;

void estimateNormals(
    std::vector<Eigen::Vector3f>& normals,
    std::vector<Eigen::Vector3f>& basePts,
    const std::vector<Eigen::Vector3f>& points);


void estimateVisibility(
    SortedOctree& so,
    std::vector<uint8_t>& visibleAngleMasks,
    std::vector<Eigen::Vector3f>& basePts,
    const std::vector<Eigen::Vector3f>& points, int topLvls);

