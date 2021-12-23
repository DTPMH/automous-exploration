#include <active_perception/frontier_finder.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>
// #include <path_searching/astar2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <plan_env/edt_environment.h>
#include <active_perception/perception_utils.h>
#include <active_perception/graph_node.h>

// use PCL region growing segmentation
// #include <pcl/point_types.h>
// #include <pcl/search/search.h>
// #include <pcl/search/kdtree.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/segmentation/region_growing.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Eigenvalues>

namespace fast_planner {
FrontierFinder::FrontierFinder(const EDTEnvironment::Ptr& edt, ros::NodeHandle& nh) {
  this->edt_env_ = edt;
  int voxel_num = edt->sdf_map_->getVoxelNum();
  frontier_flag_ = vector<char>(voxel_num, 0);
  fill(frontier_flag_.begin(), frontier_flag_.end(), 0);

  nh.param("frontier/cluster_min", cluster_min_, -1);//100，最小的前沿簇数量
  nh.param("frontier/cluster_size_xy", cluster_size_xy_, -1.0);//2.0
  nh.param("frontier/cluster_size_z", cluster_size_z_, -1.0);//10.0
  nh.param("frontier/min_candidate_dist", min_candidate_dist_, -1.0);//0.5，最小的候选距离
  nh.param("frontier/min_candidate_clearance", min_candidate_clearance_, -1.0);//0.21
  nh.param("frontier/candidate_dphi", candidate_dphi_, -1.0);//15°，真值为弧度
  nh.param("frontier/candidate_rmax", candidate_rmax_, -1.0);//2.5
  nh.param("frontier/candidate_rmin", candidate_rmin_, -1.0);//1.5
  nh.param("frontier/candidate_rnum", candidate_rnum_, -1);//3
  nh.param("frontier/down_sample", down_sample_, -1);//3
  nh.param("frontier/min_visib_num", min_visib_num_, -1);//30
  nh.param("frontier/min_view_finish_fraction", min_view_finish_fraction_, -1.0);//0.2

  raycaster_.reset(new RayCaster);
  resolution_ = edt_env_->sdf_map_->getResolution();//0.1
  Eigen::Vector3d origin, size;
  edt_env_->sdf_map_->getRegion(origin, size);//根据地图的大小设置
  raycaster_->setParams(resolution_, origin);

  percep_utils_.reset(new PerceptionUtils(nh));//相机的感知工具，即根据相机的视场确定可视范围
}

FrontierFinder::~FrontierFinder() {
}

void FrontierFinder::searchFrontiers() {
  ros::Time t1 = ros::Time::now();
  tmp_frontiers_.clear();//每一次搜索的时候都会将tmp_frontiers_清空，tmp_frontiers_里装的是每一次搜索的前沿簇

  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max, true);//
  //update_min，update_max表示每帧点云的最大最小的坐标点范围

  // 将已经改变了的前沿从前沿列表中删掉的函数
  auto resetFlag = [&](list<Frontier>::iterator& iter, list<Frontier>& frontiers) {
    Eigen::Vector3i idx;
    for (auto cell : iter->cells_) {//cells表示组成前沿的坐标点
      edt_env_->sdf_map_->posToIndex(cell, idx);
      frontier_flag_[toadr(idx)] = 0;
    }
    iter = frontiers.erase(iter);//erase函数是list列表的删除函数
  };

  std::cout << "Before remove: " << frontiers_.size() << std::endl;//frontiers_是全局的前沿变量

  removed_ids_.clear();
  int rmv_idx = 0;
  for (auto iter = frontiers_.begin(); iter != frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
        isFrontierChanged(*iter)) {//haveOverlap函数计算该前沿框是否与局部点云图有重叠的范围，若有说明该前沿可能发生改变
      //isFrontierChanged函数是用来检测前沿是否发生变化的函数，原理十分简单：遍历组成该前沿的所有点，看其是否还符合前沿条件
      //若所有都符合，则没有改变，若有些不符合则说明变化了。
      resetFlag(iter, frontiers_);//将发生改变的前沿删掉
      removed_ids_.push_back(rmv_idx);//将删掉前沿的索引放入removed_ids_中
    } else {
      ++rmv_idx;
      ++iter;
    }
  }
  std::cout << "After remove: " << frontiers_.size() << std::endl;
  //dormant_frontiers_中装的是没有viewpoint的前沿，viewpoint表示观测点，以下操作与上同理
  for (auto iter = dormant_frontiers_.begin(); iter != dormant_frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
        isFrontierChanged(*iter))
      resetFlag(iter, dormant_frontiers_);
    else
      ++iter;
  }

  // 通过轻微的对局部地图更新栅格进行膨胀，在更新范围内寻找符合前沿的点
  Vector3d search_min = update_min - Vector3d(1, 1, 0.5);//search_min，search_max表示搜索的点的栅格范围
  Vector3d search_max = update_max + Vector3d(1, 1, 0.5);
  Vector3d box_min, box_max;
  edt_env_->sdf_map_->getBox(box_min, box_max);//getBox函数得到想要探索的地图范围的坐标点
  //box_max表示想要探索地图中的最大坐标点
  //box_max表示想要探索地图中的最大坐标点
  for (int k = 0; k < 3; ++k) {
    search_min[k] = max(search_min[k], box_min[k]);
    search_max[k] = min(search_max[k], box_max[k]);
  }
  Eigen::Vector3i min_id, max_id;
  edt_env_->sdf_map_->posToIndex(search_min, min_id);//将搜索的坐标范围转换为栅格坐标范围
  edt_env_->sdf_map_->posToIndex(search_max, max_id);
//在搜索范围中开始搜索
  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z) {
        // 第一步为了减小计算量，首先计算可能是前沿的点，之后对可能是前沿的点进行扩展
        Eigen::Vector3i cur(x, y, z);
        //已经扩展过的前沿点，它对应的frontier_flag_==1，knownfree(cur) && isNeighborUnknown(cur)是前沿条件
        if (frontier_flag_[toadr(cur)] == 0 && knownfree(cur) && isNeighborUnknown(cur)) {
          // 如果改点符合前沿条件并且没有被扩展过，对其进行扩展，计算该点对应的前沿簇
          expandFrontier(cur);//扩展前沿，并且对每一个前沿都进行信息计算，将扩展的前沿放置在tmp_frontiers_中
        }
      }
  splitLargeFrontiers(tmp_frontiers_);//将大型的前沿分给为多个小前沿，方便计算

  ROS_WARN_THROTTLE(5.0, "Frontier t: %lf", (ros::Time::now() - t1).toSec());//每5s打印一次
}

void FrontierFinder::expandFrontier(const Eigen::Vector3i& first) {

  auto t1 = ros::Time::now();

  // 定义需要计算的类型
  queue<Eigen::Vector3i> cell_queue;//cell_queue中装的是正在处于扩展中的前沿点
  vector<Eigen::Vector3d> expanded;//expanded中装的是已经被扩展过的符合前沿条件的坐标点
  Vector3d pos;

  edt_env_->sdf_map_->indexToPos(first, pos);
  expanded.push_back(pos);//将该前沿点坐标放入expanded
  cell_queue.push(first);//将该前沿的栅格坐标放入cell_queue，开始对其进行扩展
  frontier_flag_[toadr(first)] = 1;//将该前沿点对应的扩展标志位置1

  // 利用while循环搜索同属于一个前沿的点
  //原理是将该点周围的26个点进行遍历，将符合前沿条件且还没被扩展过的点以及处于探索地图范围的点的栅格坐标放入cell_queue中，
  //每扩展完一个点（即将其对应的26个邻居都遍历完）则将该点从cell_queue中删掉
  //直到cell_queue中没有点，这样就会将同属于一个前沿的点都集中在一个前沿中
  while (!cell_queue.empty()) {
    auto cur = cell_queue.front();//cur为cell_queue的第一个点
    cell_queue.pop();//删掉第一个点
    auto nbrs = allNeighbors(cur);//寻找26个邻居
    for (auto nbr : nbrs) {//遍历26个邻居
      // 若其已经被扩展过或者不在探索地图范围内或者不符合前沿条件，则将其跳过
      int adr = toadr(nbr);
      if (frontier_flag_[adr] == 1 || !edt_env_->sdf_map_->isInBox(nbr) ||
          !(knownfree(nbr) && isNeighborUnknown(nbr)))
        continue;
      //若是符合前沿条件，且在地图范围内，并且没有被扩展过，则将其对应的点坐标放入expanded中
      //将对应的扩展标志位frontier_flag_置1
      //将该邻居点加入cell_queue中
      edt_env_->sdf_map_->indexToPos(nbr, pos);
      expanded.push_back(pos);
      cell_queue.push(nbr);
      frontier_flag_[adr] = 1;
    }
  }
  if (expanded.size() > cluster_min_) {//如果搜索完该集合的所有点，发现该前沿对应的前沿点数量大于100，则将其认为是一个真正的前沿
    // Compute detailed info
    Frontier frontier;
    frontier.cells_ = expanded;//将该前沿的点的坐标放入前沿frontier类中的cells中
    computeFrontierInfo(frontier);//计算该前沿的信息（包含前沿的中心点以及约束框大小）
    //中心点放置在前沿frontier_类中的average_中
    //约束框放置在前沿frontier_类中的box_min_以及box_max_中
    tmp_frontiers_.push_back(frontier);//将计算过信息的前沿放入每一次的临时前沿变量tmp_frontiers_中
  }
}

void FrontierFinder::splitLargeFrontiers(list<Frontier>& frontiers) {
  list<Frontier> splits, tmps;
  for (auto it = frontiers.begin(); it != frontiers.end(); ++it) {
    // 遍历所有前沿，检查其是否需要分割
    if (splitHorizontally(*it, splits)) {//如果检测其需要分割，则将分割后的前沿放置在临时前沿变量tmps列表中
      tmps.insert(tmps.end(), splits.begin(), splits.end());
      splits.clear();
    } else
      tmps.push_back(*it);//若是该前沿不需要分割，则将其完整放入tmps
  }
  frontiers = tmps;
}

bool FrontierFinder::splitHorizontally(const Frontier& frontier, list<Frontier>& splits) {
  // Split a frontier into small piece if it is too large
  auto mean = frontier.average_.head<2>();//mean表示前沿的中心点的想，x,y坐标
  bool need_split = false;
  for (auto cell : frontier.filtered_cells_) {//遍历该前沿的体素滤波后的点坐标
    if ((cell.head<2>() - mean).norm() > cluster_size_xy_) {//如果其中有一个点距离中心点超出阈值2m,则说明其需要被分割
      need_split = true;
      break;
    }
  }
  if (!need_split) return false;

  Eigen::Matrix2d cov;//计算平方差矩阵
  cov.setZero();
  for (auto cell : frontier.filtered_cells_) {
    Eigen::Vector2d diff = cell.head<2>() - mean;
    cov += diff * diff.transpose();
  }
  cov /= double(frontier.filtered_cells_.size());

  // Find eigenvector corresponds to maximal eigenvector
  Eigen::EigenSolver<Eigen::Matrix2d> es(cov);
  auto values = es.eigenvalues().real();//eigenvalues求的是特征值向量
  auto vectors = es.eigenvectors().real();//eigenvectors求的是特征向量矩阵
  int max_idx;
  double max_eigenvalue = -1000000;
  for (int i = 0; i < values.rows(); ++i) {
    if (values[i] > max_eigenvalue) {
      max_idx = i;
      max_eigenvalue = values[i];
    }
  }
  //max_idx表示最大的特征值对应的点索引，
  Eigen::Vector2d first_pc = vectors.col(max_idx);//first_pc表示最大特征值对应的特征向量
  std::cout << "max idx: " << max_idx << std::endl;
  std::cout << "mean: " << mean.transpose() << ", first pc: " << first_pc.transpose() << std::endl;

  // 以下原理不懂，是利用特征向量，与中心点差值进行点乘的结果进行判定
  Frontier ftr1, ftr2;
  for (auto cell : frontier.cells_) {
    if ((cell.head<2>() - mean).dot(first_pc) >= 0)
      ftr1.cells_.push_back(cell);
    else
      ftr2.cells_.push_back(cell);
  }
  computeFrontierInfo(ftr1);
  computeFrontierInfo(ftr2);

  // 
  list<Frontier> splits2;
  if (splitHorizontally(ftr1, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
    splits2.clear();
  } else
    splits.push_back(ftr1);

  if (splitHorizontally(ftr2, splits2))
    splits.insert(splits.end(), splits2.begin(), splits2.end());
  else
    splits.push_back(ftr2);

  return true;
}

bool FrontierFinder::isInBoxes(
    const vector<pair<Vector3d, Vector3d>>& boxes, const Eigen::Vector3i& idx) {
  Vector3d pt;
  edt_env_->sdf_map_->indexToPos(idx, pt);
  for (auto box : boxes) {
    // Check if contained by a box
    bool inbox = true;
    for (int i = 0; i < 3; ++i) {
      inbox = inbox && pt[i] > box.first[i] && pt[i] < box.second[i];
      if (!inbox) break;
    }
    if (inbox) return true;
  }
  return false;
}

void FrontierFinder::updateFrontierCostMatrix() {
  std::cout << "cost mat size before remove: " << std::endl;
  for (auto ftr : frontiers_)
    std::cout << "(" << ftr.costs_.size() << "," << ftr.paths_.size() << "), ";
  std::cout << "" << std::endl;

  std::cout << "cost mat size remove: " << std::endl;
  if (!removed_ids_.empty()) {
    // Delete path and cost for removed clusters
    for (auto it = frontiers_.begin(); it != first_new_ftr_; ++it) {
      auto cost_iter = it->costs_.begin();
      auto path_iter = it->paths_.begin();
      int iter_idx = 0;
      for (int i = 0; i < removed_ids_.size(); ++i) {
        // Step iterator to the item to be removed
        while (iter_idx < removed_ids_[i]) {
          ++cost_iter;
          ++path_iter;
          ++iter_idx;
        }
        cost_iter = it->costs_.erase(cost_iter);
        path_iter = it->paths_.erase(path_iter);
      }
      std::cout << "(" << it->costs_.size() << "," << it->paths_.size() << "), ";
    }
    removed_ids_.clear();
  }
  std::cout << "" << std::endl;

  auto updateCost = [](const list<Frontier>::iterator& it1, const list<Frontier>::iterator& it2) {
    std::cout << "(" << it1->id_ << "," << it2->id_ << "), ";
    // Search path from old cluster's top viewpoint to new cluster'
    Viewpoint& vui = it1->viewpoints_.front();
    Viewpoint& vuj = it2->viewpoints_.front();
    vector<Vector3d> path_ij;
    double cost_ij = ViewNode::computeCost(
        vui.pos_, vuj.pos_, vui.yaw_, vuj.yaw_, Vector3d(0, 0, 0), 0, path_ij);
    // Insert item for both old and new clusters
    it1->costs_.push_back(cost_ij);
    it1->paths_.push_back(path_ij);
    reverse(path_ij.begin(), path_ij.end());
    it2->costs_.push_back(cost_ij);
    it2->paths_.push_back(path_ij);
  };

  std::cout << "cost mat add: " << std::endl;
  // Compute path and cost between old and new clusters
  for (auto it1 = frontiers_.begin(); it1 != first_new_ftr_; ++it1)
    for (auto it2 = first_new_ftr_; it2 != frontiers_.end(); ++it2)
      updateCost(it1, it2);

  // Compute path and cost between new clusters
  for (auto it1 = first_new_ftr_; it1 != frontiers_.end(); ++it1)
    for (auto it2 = it1; it2 != frontiers_.end(); ++it2) {
      if (it1 == it2) {
        std::cout << "(" << it1->id_ << "," << it2->id_ << "), ";
        it1->costs_.push_back(0);
        it1->paths_.push_back({});
      } else
        updateCost(it1, it2);
    }
  std::cout << "" << std::endl;
  std::cout << "cost mat size final: " << std::endl;
  for (auto ftr : frontiers_)
    std::cout << "(" << ftr.costs_.size() << "," << ftr.paths_.size() << "), ";
  std::cout << "" << std::endl;
}

void FrontierFinder::mergeFrontiers(Frontier& ftr1, const Frontier& ftr2) {
  // Merge ftr2 into ftr1
  ftr1.average_ =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  ftr1.cells_.insert(ftr1.cells_.end(), ftr2.cells_.begin(), ftr2.cells_.end());
  computeFrontierInfo(ftr1);
}

bool FrontierFinder::canBeMerged(const Frontier& ftr1, const Frontier& ftr2) {
  Vector3d merged_avg =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  // Check if it can merge two frontier without exceeding size limit
  for (auto c1 : ftr1.cells_) {
    auto diff = c1 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  for (auto c2 : ftr2.cells_) {
    auto diff = c2 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  return true;
}

bool FrontierFinder::haveOverlap(
    const Vector3d& min1, const Vector3d& max1, const Vector3d& min2, const Vector3d& max2) {
  // min1，max1表示的是每一个前沿框的栅格坐标大小，min2与max2表示的是局部地图更新的栅格坐标大小
  Vector3d bmin, bmax;
  //以下函数的作用是，求两个框的最大的最小栅格范围，以及最小的最大栅格范围，
  //如果最大的最小栅格范围大于最小的最大栅范围，那么说明两个框没有交点，所以没有重叠
  //如果没有理解的话，可以自己话两个长方形的框，比较以下。
  //bmin[i]表示的是两个框中三个轴的最小栅格坐标中的最大值
  //bmax[i]表示的是两个框中三个轴的最大栅格坐标中的最小值
  for (int i = 0; i < 3; ++i) {
    bmin[i] = max(min1[i], min2[i]);
    bmax[i] = min(max1[i], max2[i]);
    if (bmin[i] > bmax[i] + 1e-3) return false;
  }
  return true;
}

bool FrontierFinder::isFrontierChanged(const Frontier& ft) {
  for (auto cell : ft.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!(knownfree(idx) && isNeighborUnknown(idx))) return true;
  }
  return false;
}

void FrontierFinder::computeFrontierInfo(Frontier& ftr) {
  // 计算该前沿对应的中心点坐标以及该前沿对应的约束框大小（坐标点形式）（对应论文中的AABB）
  //中心点放置在前沿frontier_类中的average_中
  //约束框放置在前沿frontier_类中的box_min_以及box_max_中
  ftr.average_.setZero();
  ftr.box_max_ = ftr.cells_.front();
  ftr.box_min_ = ftr.cells_.front();
  for (auto cell : ftr.cells_) {
    ftr.average_ += cell;
    for (int i = 0; i < 3; ++i) {
      ftr.box_min_[i] = min(ftr.box_min_[i], cell[i]);
      ftr.box_max_[i] = max(ftr.box_max_[i], cell[i]);
    }
  }
  ftr.average_ /= double(ftr.cells_.size());

  downsample(ftr.cells_, ftr.filtered_cells_);//对前沿进行下采样（即体素滤波）
  //将滤波过后的前沿点放置在前沿frontier_类中的filtered_cells_中
}

void FrontierFinder::computeFrontiersToVisit() {
  first_new_ftr_ = frontiers_.end();
  int new_num = 0;
  int new_dormant_num = 0;
  // Try find viewpoints for each cluster and categorize them according to viewpoint number
  for (auto& tmp_ftr : tmp_frontiers_) {
    // Search viewpoints around frontier
    sampleViewpoints(tmp_ftr);//计算观测点的坐标点以及观测点对应的偏航角以及能够看到前沿点的数量
    if (!tmp_ftr.viewpoints_.empty()) {//如果采样点非空
      ++new_num;
      list<Frontier>::iterator inserted = frontiers_.insert(frontiers_.end(), tmp_ftr);//则将该前沿加入到全局前沿变量frontiers_中
      // Sort the viewpoints by coverage fraction, best view in front，对观测点进行排序，能够观测到的最多前沿点数量的观测点排至第一位
      sort(
          inserted->viewpoints_.begin(), inserted->viewpoints_.end(),
          [](const Viewpoint& v1, const Viewpoint& v2) { return v1.visib_num_ > v2.visib_num_; });
      if (first_new_ftr_ == frontiers_.end()) first_new_ftr_ = inserted;
    } else {//若观测点是空的，则将该前沿加入dormant_frontiers_中
      dormant_frontiers_.push_back(tmp_ftr);
      ++new_dormant_num;
    }
  }
  //重新计算frontiers_的数量以及对应的索引
  int idx = 0;
  for (auto& ft : frontiers_) {
    ft.id_ = idx++;
    std::cout << ft.id_ << ", ";
  }
  std::cout << "\nnew num: " << new_num << ", new dormant: " << new_dormant_num << std::endl;
  std::cout << "to visit: " << frontiers_.size() << ", dormant: " << dormant_frontiers_.size()
            << std::endl;
}
void FrontierFinder::computeNBVFrontiersToVisit() {
  nbv_frontiers.clear();
  first_new_ftr_ = nbv_frontiers.end();
  int new_num = 0;
  int new_dormant_num = 0;
  // Try find viewpoints for each cluster and categorize them according to viewpoint number
  for (auto& tmp_ftr : tmp_frontiers_) {
    // Search viewpoints around frontier
    sampleViewpoints(tmp_ftr);//计算观测点的坐标点以及观测点对应的偏航角以及能够看到前沿点的数量
    if (!tmp_ftr.viewpoints_.empty()) {//如果采样点非空
      ++new_num;
      list<Frontier>::iterator inserted = nbv_frontiers.insert(nbv_frontiers.end(), tmp_ftr);//则将该前沿加入到全局前沿变量frontiers_中
      // Sort the viewpoints by coverage fraction, best view in front，对观测点进行排序，能够观测到的最多前沿点数量的观测点排至第一位
      sort(
          inserted->viewpoints_.begin(), inserted->viewpoints_.end(),
          [](const Viewpoint& v1, const Viewpoint& v2) { return v1.visib_num_ > v2.visib_num_; });
      if (first_new_ftr_ == nbv_frontiers.end()) first_new_ftr_ = inserted;
    } else {//若观测点是空的，则将该前沿加入dormant_frontiers_中
      dormant_frontiers_.push_back(tmp_ftr);
      ++new_dormant_num;
    }
  }
  std::cout << "\nnbv num: " << nbv_frontiers.size()<<std::endl;
 }

void FrontierFinder::getTopViewpointsInfo(
    const Vector3d& cur_pos, vector<Eigen::Vector3d>& points, vector<double>& yaws,
    vector<Eigen::Vector3d>& averages,vector<int>& ids_) {
  points.clear();
  yaws.clear();
  averages.clear();
  for (auto frontier : frontiers_) {//遍历所有的前沿
    bool no_view = true;
    for (auto view : frontier.viewpoints_) {//遍历每个前沿的观测点
      // Retrieve the first viewpoint that is far enough and has highest coverage
      if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;//如果该观测点与起始点的距离小于0.5m,则将该点舍弃
      points.push_back(view.pos_);//否则将这个观测点放入points
      yaws.push_back(view.yaw_);//将该观测点的偏航角放入yaws
      averages.push_back(frontier.average_);//将该前沿的中心点放入averages中
      ids_.push_back(frontier.id_);
      no_view = false;
      break;
    }
    if (no_view) {
      // 如果该前沿所有的观测点都离起始点太近，则将第一个观测点放入
      auto view = frontier.viewpoints_.front();
      points.push_back(view.pos_);
      yaws.push_back(view.yaw_);
      averages.push_back(frontier.average_);
    }
  }
}
/*
void FrontierFinder::getViewpointsInfo(
    const Vector3d& cur_pos, const vector<int>& ids, vector<Eigen::Vector3d>& points, vector<double>& yaws,
    vector<Eigen::Vector3d>& averages) {
  points.clear();
  yaws.clear();
  averages.clear();
  for (auto id : ids) {
    // Scan all frontiers to find one with the same id
    for (auto frontier : frontiers_) {
      if (frontier.id_ == id) {
    bool no_view = true;
    for (auto view : frontier.viewpoints_) {//遍历每个前沿的观测点
      // Retrieve the first viewpoint that is far enough and has highest coverage
      if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;//如果该观测点与起始点的距离小于0.5m,则将该点舍弃
      points.push_back(view.pos_);//否则将这个观测点放入points
      yaws.push_back(view.yaw_);//将该观测点的偏航角放入yaws
      averages.push_back(frontier.average_);//将该前沿的中心点放入averages中
      no_view = false;
      break;
    }
    if (no_view) {
      // 如果该前沿所有的观测点都离起始点太近，则将第一个观测点放入
      auto view = frontier.viewpoints_.front();
      points.push_back(view.pos_);
      yaws.push_back(view.yaw_);
      averages.push_back(frontier.average_);
    }
      }
    }
  }
}
*/

void FrontierFinder::getViewpointsInfo(
    const Vector3d& cur_pos, const vector<int>& ids, const int& view_num, const double& max_decay,
    vector<vector<Eigen::Vector3d>>& points, vector<vector<double>>& yaws) {
  points.clear();
  yaws.clear();
  for (auto id : ids) {
    // Scan all frontiers to find one with the same id
    for (auto frontier : frontiers_) {
      if (frontier.id_ == id) {
        // Get several top viewpoints that are far enough
        vector<Eigen::Vector3d> pts;
        vector<double> ys;
        int visib_thresh = frontier.viewpoints_.front().visib_num_ * max_decay;//visib_thresh表示观测点看到最多的前沿点数量*0.8
        for (auto view : frontier.viewpoints_) {
          if (pts.size() >= view_num || view.visib_num_ <= visib_thresh) break;
          if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;//min_candidate_dist_=0.5
          pts.push_back(view.pos_);
          ys.push_back(view.yaw_);
        }
        if (pts.empty()) {
          // All viewpoints are very close, ignore the distance limit
          for (auto view : frontier.viewpoints_) {
            if (pts.size() >= view_num || view.visib_num_ <= visib_thresh) break;
            pts.push_back(view.pos_);
            ys.push_back(view.yaw_);
          }
        }
        points.push_back(pts);
        yaws.push_back(ys);
      }
    }
  }
}

void FrontierFinder::getFrontiers(vector<vector<Eigen::Vector3d>>& clusters) {
  clusters.clear();
  for (auto frontier : frontiers_)
    clusters.push_back(frontier.cells_);//三维矩阵，里面存储的是每个前沿的所有前沿点
  // clusters.push_back(frontier.filtered_cells_);
}

void FrontierFinder::getnbvFrontiers(vector<Vector3d>& points_,vector<double>& yaws_) {
  points_.clear();
  yaws_.clear();
  for (auto frontier : nbv_frontiers)
  {
    points_.push_back(frontier.viewpoints_.back().pos_);
    yaws_.push_back(frontier.viewpoints_.back().yaw_);
  }
}

void FrontierFinder::getDormantFrontiers(vector<vector<Vector3d>>& clusters) {
  clusters.clear();
  for (auto ft : dormant_frontiers_)
    clusters.push_back(ft.cells_);
}

void FrontierFinder::getFrontierBoxes(vector<pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes) {
  boxes.clear();
  for (auto frontier : frontiers_) {
    Vector3d center = (frontier.box_max_ + frontier.box_min_) * 0.5;
    Vector3d scale = frontier.box_max_ - frontier.box_min_;
    boxes.push_back(make_pair(center, scale));
  }
}

void FrontierFinder::getPathForTour(
    const Vector3d& pos, const vector<int>& frontier_ids, vector<Vector3d>& path) {
  // Make an frontier_indexer to access the frontier list easier
  vector<list<Frontier>::iterator> frontier_indexer;
  for (auto it = frontiers_.begin(); it != frontiers_.end(); ++it)
    frontier_indexer.push_back(it);

  // Compute the path from current pos to the first frontier
  vector<Vector3d> segment;
  ViewNode::searchPath(pos, frontier_indexer[frontier_ids[0]]->viewpoints_.front().pos_, segment);
  path.insert(path.end(), segment.begin(), segment.end());

  // Get paths of tour passing all clusters
  for (int i = 0; i < frontier_ids.size() - 1; ++i) {
    // Move to path to next cluster
    auto path_iter = frontier_indexer[frontier_ids[i]]->paths_.begin();
    int next_idx = frontier_ids[i + 1];
    for (int j = 0; j < next_idx; ++j)
      ++path_iter;
    path.insert(path.end(), path_iter->begin(), path_iter->end());
  }
}

void FrontierFinder::getFullCostMatrix(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
    Eigen::MatrixXd& mat) {
  if (false) {
    // Use symmetric TSP formulation
    int dim = frontiers_.size() + 2;
    mat.resize(dim, dim);  // current pose (0), sites, and virtual depot finally

    int i = 1, j = 1;
    for (auto ftr : frontiers_) {
      for (auto cs : ftr.costs_)
        mat(i, j++) = cs;
      ++i;
      j = 1;
    }

    // Costs from current pose to sites
    for (auto ftr : frontiers_) {
      Viewpoint vj = ftr.viewpoints_.front();
      vector<Vector3d> path;
      mat(0, j) = mat(j, 0) =
          ViewNode::computeCost(cur_pos, vj.pos_, cur_yaw[0], vj.yaw_, cur_vel, cur_yaw[1], path);
      ++j;
    }
    // Costs from depot to sites, the same large vaule
    for (j = 1; j < dim - 1; ++j) {
      mat(dim - 1, j) = mat(j, dim - 1) = 100;
    }
    // Zero cost to depot to ensure connection
    mat(0, dim - 1) = mat(dim - 1, 0) = -10000;

  } else {
    // Use Asymmetric TSP
    int dimen = frontiers_.size();
    mat.resize(dimen + 1, dimen + 1);
    // std::cout << "mat size: " << mat.rows() << ", " << mat.cols() << std::endl;
    // Fill block for clusters
    int i = 1, j = 1;
    for (auto ftr : frontiers_) {
      for (auto cs : ftr.costs_) {
        // std::cout << "(" << i << ", " << j << ")"
        // << ", ";
        mat(i, j++) = cs;
      }
      ++i;
      j = 1;
    }
    // std::cout << "" << std::endl;

    // Fill block from current state to clusters
    mat.leftCols<1>().setZero();
    for (auto ftr : frontiers_) {
      // std::cout << "(0, " << j << ")"
      // << ", ";
      Viewpoint vj = ftr.viewpoints_.front();
      vector<Vector3d> path;
      mat(0, j++) =
          ViewNode::computeCost(cur_pos, vj.pos_, cur_yaw[0], vj.yaw_, cur_vel, cur_yaw[1], path);
    }
    // std::cout << "" << std::endl;
  }
}

/*
void FrontierFinder::findViewpoints(
    const Vector3d& sample, const Vector3d& ftr_avg, vector<Viewpoint>& vps) {
  if (!edt_env_->sdf_map_->isInBox(sample) ||
      edt_env_->sdf_map_->getInflateOccupancy(sample) == 1 || isNearUnknown(sample))
    return;

  double left_angle_, right_angle_, vertical_angle_, ray_length_;

  // Central yaw is determined by frontier's average position and sample
  auto dir = ftr_avg - sample;
  double hc = atan2(dir[1], dir[0]);

  vector<int> slice_gains;
  // Evaluate info gain of different slices
  for (double phi_h = -M_PI_2; phi_h <= M_PI_2 + 1e-3; phi_h += M_PI / 18) {
    // Compute gain of one slice
    int gain = 0;
    for (double phi_v = -vertical_angle_; phi_v <= vertical_angle_; phi_v += vertical_angle_ / 3) {
      // Find endpoint of a ray
      Vector3d end;
      end[0] = sample[0] + ray_length_ * cos(phi_v) * cos(hc + phi_h);
      end[1] = sample[1] + ray_length_ * cos(phi_v) * sin(hc + phi_h);
      end[2] = sample[2] + ray_length_ * sin(phi_v);

      // Do raycasting to check info gain
      Vector3i idx;
      raycaster_->input(sample, end);
      while (raycaster_->nextId(idx)) {
        // Hit obstacle, stop the ray
        if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 || !edt_env_->sdf_map_->isInBox(idx))
          break;
        // Count number of unknown cells
        if (edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) ++gain;
      }
    }
    slice_gains.push_back(gain);
  }

  // Sum up slices' gain to get different yaw's gain
  vector<pair<double, int>> yaw_gains;
  for (int i = 0; i < 6; ++i)  // [-90,-10]-> [10,90], delta_yaw = 20, 6 groups
  {
    double yaw = hc - M_PI_2 + M_PI / 9.0 * i + right_angle_;
    int gain = 0;
    for (int j = 2 * i; j < 2 * i + 9; ++j)  // 80 degree hFOV, 9 slices
      gain += slice_gains[j];
    yaw_gains.push_back(make_pair(yaw, gain));
  }

  // Get several yaws with highest gain
  vps.clear();
  sort(
      yaw_gains.begin(), yaw_gains.end(),
      [](const pair<double, int>& p1, const pair<double, int>& p2) {
        return p1.second > p2.second;
      });
  for (int i = 0; i < 3; ++i) {
    if (yaw_gains[i].second < min_visib_num_) break;
    Viewpoint vp = { sample, yaw_gains[i].first, yaw_gains[i].second };
    while (vp.yaw_ < -M_PI)
      vp.yaw_ += 2 * M_PI;
    while (vp.yaw_ > M_PI)
      vp.yaw_ -= 2 * M_PI;
    vps.push_back(vp);
  }
}
*/

void FrontierFinder::findViewpoints(
    const Vector3d& sample, const Vector3d& ftr_avg, int& slince_num) {
  if (!edt_env_->sdf_map_->isInBox(sample) ||
      edt_env_->sdf_map_->getInflateOccupancy(sample) == 1 || isNearUnknown(sample))
    return;

  double  vertical_angle_, ray_length_=4.5;
  vertical_angle_=30.0*M_PI/180.0;
  // Central yaw is determined by frontier's average position and sample
  auto dir = ftr_avg - sample;
  double hc = atan2(dir[1], dir[0]);

  vector<int> slice_gains;
  int max_slince_num=0;
  // Evaluate info gain of different slices
  for (double phi_h = -M_PI_2; phi_h <= M_PI_2 + 1e-3; phi_h += M_PI / 18) {
    // Compute gain of one slice
    int gain = 0;
    for (double phi_v = -vertical_angle_; phi_v <= vertical_angle_; phi_v += vertical_angle_ / 3) {
      // Find endpoint of a ray
      Vector3d end;
      end[0] = sample[0] + ray_length_ * cos(phi_v) * cos(hc + phi_h);
      end[1] = sample[1] + ray_length_ * cos(phi_v) * sin(hc + phi_h);
      end[2] = sample[2] + ray_length_ * sin(phi_v);

      // Do raycasting to check info gain
      Vector3i idx;
      raycaster_->input(sample, end);
      while (raycaster_->nextId(idx)) {
        // Hit obstacle, stop the ray
        if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 || !edt_env_->sdf_map_->isInBox(idx))
          break;
        // Count number of unknown cells
        if (edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) ++gain;
      }
    }
    if (gain>max_slince_num)
    {
      max_slince_num=gain;
      slince_num=max_slince_num;
    }
  }
}


// Sample viewpoints around frontier's average position, check coverage to the frontier cells
void FrontierFinder::sampleViewpoints(Frontier& frontier) {
//candidate_rmin_=1.5,candidate_rmax_=2.5,candidate_rnum_=3
//以下两个for循环的原理是，以前沿中心点为圆心，在半径1.5-2.5之间按照平均距离做三个圆环
//以15°与间隔，在每个圆环上都采样一个点sample_pos
  for (double rc = candidate_rmin_, dr = (candidate_rmax_ - candidate_rmin_) / candidate_rnum_;
       rc <= candidate_rmax_ + 1e-3; rc += dr)
       //candidate_dphi_=15°对应的弧度
    for (double phi = -M_PI; phi < M_PI; phi += candidate_dphi_) {
      const Vector3d sample_pos = frontier.average_ + rc * Vector3d(cos(phi), sin(phi), 0);

      // 检验采样的sample_pos是否处于探索地图范围内且检验这个点是否是膨胀后的障碍点且检验其周围（0.2范围内的点）是否有点处于未知区域
      //如果是，则舍弃该采样点
      if (!edt_env_->sdf_map_->isInBox(sample_pos) ||
          edt_env_->sdf_map_->getInflateOccupancy(sample_pos) == 1 || isNearUnknown(sample_pos))
        continue;

      // 计算该采样点对应的平均偏航角
      auto& cells = frontier.filtered_cells_;//cells表示该前沿过滤后的所有点坐标
      Eigen::Vector3d ref_dir = (cells.front() - sample_pos).normalized();//ref_dir表示第一个点与采样点的差的单位向量
      double avg_yaw = 0.0;
      for (int i = 1; i < cells.size(); ++i) {
        Eigen::Vector3d dir = (cells[i] - sample_pos).normalized();
        double yaw = acos(dir.dot(ref_dir));
        if (ref_dir.cross(dir)[2] < 0) yaw = -yaw;
        avg_yaw += yaw;
      }
      avg_yaw = avg_yaw / cells.size() + atan2(ref_dir[1], ref_dir[0]);
      //根据中心点与该前沿点过滤后的所有点之间的偏航角求一个平均的偏航角
      wrapYaw(avg_yaw);//将该偏航角的值限定到（-π，π）
      // Compute the fraction of covered and visible cells
      int visib_num = countVisibleCells(sample_pos, avg_yaw, cells);//计算该采样点能够看到的点的数量
      if (visib_num > min_visib_num_) {//如果能看到点的数量大于30，则将其加入前沿的viewpoints中，其中对应的信息有采样点的位置、平均偏航角以及能够看到的点的数量
        Viewpoint vp = { sample_pos, avg_yaw, visib_num };
        frontier.viewpoints_.push_back(vp);
        // int gain = findMaxGainYaw(sample_pos, frontier, sample_yaw);
      }
      // }
    }
}

bool FrontierFinder::isFrontierCovered() {
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max);

  auto checkChanges = [&](const list<Frontier>& frontiers) {
    for (auto ftr : frontiers) {
      if (!haveOverlap(ftr.box_min_, ftr.box_max_, update_min, update_max)) continue;
      const int change_thresh = min_view_finish_fraction_ * ftr.cells_.size();
      int change_num = 0;
      for (auto cell : ftr.cells_) {
        Eigen::Vector3i idx;
        edt_env_->sdf_map_->posToIndex(cell, idx);
        if (!(knownfree(idx) && isNeighborUnknown(idx)) && ++change_num >= change_thresh)
          return true;
      }
    }
    return false;
  };

  if (checkChanges(frontiers_) || checkChanges(dormant_frontiers_)) return true;

  return false;
}

bool FrontierFinder::isNearUnknown(const Eigen::Vector3d& pos) {
  const int vox_num = floor(min_candidate_clearance_ / resolution_);//min_candidate_clearance_=0.21/0.1=2.1
  //floor表示小于x的最大整数，因此vox_num=2
  for (int x = -vox_num; x <= vox_num; ++x)
    for (int y = -vox_num; y <= vox_num; ++y)
      for (int z = -1; z <= 1; ++z) {
        Eigen::Vector3d vox;
        vox << pos[0] + x * resolution_, pos[1] + y * resolution_, pos[2] + z * resolution_;
        if (edt_env_->sdf_map_->getOccupancy(vox) == SDFMap::UNKNOWN) return true;
      }
  return false;
}

int FrontierFinder::countVisibleCells(
    const Eigen::Vector3d& pos, const double& yaw, const vector<Eigen::Vector3d>& cluster) {
  percep_utils_->setPose(pos, yaw);//根据设定的相机视线
  int visib_num = 0;
  Eigen::Vector3i idx;
  for (auto cell : cluster) {
    // Check if frontier cell is inside FOV
    if (!percep_utils_->insideFOV(cell)) continue;

    // Check if frontier cell is visible (not occulded by obstacles)
    raycaster_->input(cell, pos);
    bool visib = true;
    while (raycaster_->nextId(idx)) {
      if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 ||
          edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        visib = false;
        break;
      }
    }
    if (visib) visib_num += 1;
  }
  return visib_num;
}

void FrontierFinder::downsample(
    const vector<Eigen::Vector3d>& cluster_in, vector<Eigen::Vector3d>& cluster_out) {
  // 对前沿点进行体素下采样
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudf(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto cell : cluster_in)
    cloud->points.emplace_back(cell[0], cell[1], cell[2]);//将输入点云放置在cloud->points中

  const double leaf_size = edt_env_->sdf_map_->getResolution() * down_sample_;//0.1*3=0.3，采样叶子大小，即体素采样的正方体边长
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(leaf_size, leaf_size, leaf_size);
  sor.filter(*cloudf);//将采样后的点放置在cloudf指针对应的地址中

  cluster_out.clear();
  for (auto pt : cloudf->points)
    cluster_out.emplace_back(pt.x, pt.y, pt.z);//将采样后的点放置在输出点云中
}

void FrontierFinder::wrapYaw(double& yaw) {
  while (yaw < -M_PI)
    yaw += 2 * M_PI;
  while (yaw > M_PI)
    yaw -= 2 * M_PI;
}

Eigen::Vector3i FrontierFinder::searchClearVoxel(const Eigen::Vector3i& pt) {
  queue<Eigen::Vector3i> init_que;
  vector<Eigen::Vector3i> nbrs;
  Eigen::Vector3i cur, start_idx;
  init_que.push(pt);
  // visited_flag_[toadr(pt)] = 1;

  while (!init_que.empty()) {
    cur = init_que.front();
    init_que.pop();
    if (knownfree(cur)) {
      start_idx = cur;
      break;
    }

    nbrs = sixNeighbors(cur);
    for (auto nbr : nbrs) {
      int adr = toadr(nbr);
      // if (visited_flag_[adr] == 0)
      // {
      //   init_que.push(nbr);
      //   visited_flag_[adr] = 1;
      // }
    }
  }
  return start_idx;
}

inline vector<Eigen::Vector3i> FrontierFinder::sixNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(6);
  Eigen::Vector3i tmp;

  tmp = voxel - Eigen::Vector3i(1, 0, 0);
  neighbors[0] = tmp;
  tmp = voxel + Eigen::Vector3i(1, 0, 0);
  neighbors[1] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 1, 0);
  neighbors[2] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 1, 0);
  neighbors[3] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 0, 1);
  neighbors[4] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 0, 1);
  neighbors[5] = tmp;

  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::tenNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(10);
  Eigen::Vector3i tmp;
  int count = 0;

  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      if (x == 0 && y == 0) continue;
      tmp = voxel + Eigen::Vector3i(x, y, 0);
      neighbors[count++] = tmp;
    }
  }
  neighbors[count++] = tmp - Eigen::Vector3i(0, 0, 1);
  neighbors[count++] = tmp + Eigen::Vector3i(0, 0, 1);
  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::allNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(26);
  Eigen::Vector3i tmp;
  int count = 0;
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) {
        if (x == 0 && y == 0 && z == 0) continue;
        tmp = voxel + Eigen::Vector3i(x, y, z);
        neighbors[count++] = tmp;
      }
  return neighbors;
}

inline bool FrontierFinder::isNeighborUnknown(const Eigen::Vector3i& voxel) {
  // At least one neighbor is unknown
  auto nbrs = sixNeighbors(voxel);
  for (auto nbr : nbrs) {
    if (edt_env_->sdf_map_->getOccupancy(nbr) == SDFMap::UNKNOWN) return true;
  }
  return false;
}

inline int FrontierFinder::toadr(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->toAddress(idx);
}

inline bool FrontierFinder::knownfree(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::FREE;
}

inline bool FrontierFinder::inmap(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->isInMap(idx);
}

}  // namespace fast_planner