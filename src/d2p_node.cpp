#include <cv_bridge/cv_bridge.h>
#include <depth_image_proc/depth_conversions.h>
#include <eigen_conversions/eigen_msg.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_geometry/pinhole_camera_model.h>
#include <message_filters/subscriber.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Geometry>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>

using namespace std;

class Params {
 public:
  void readParams(ros::NodeHandle &node_handle) {
    string config_file = readParam<std::string>(node_handle, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
      ROS_ERROR("ERROR: Wrong path to settings");
      return;
    }

    // ros topics
    fsSettings["depth_topic"] >> depth_topic;
    fsSettings["odom_topic"] >> odom_topic;

    // depth params
    fx = fsSettings["fx"];
    fy = fsSettings["fy"];
    cx = fsSettings["cx"];
    cy = fsSettings["cy"];
    height = fsSettings["height"];
    width = fsSettings["width"];
    packCamInfo();
    model.fromCameraInfo(cam_info);

    // extrinsic
    use_extrinsic = fsSettings["use_extrinsic"];
    if (use_extrinsic) {
      cv::Mat cv_R, cv_T;
      fsSettings["extrinsicRotation"] >> cv_R;
      fsSettings["extrinsicTranslation"] >> cv_T;
      cv::cv2eigen(cv_R, R_co);
      cv::cv2eigen(cv_T, t_co);
    } else {
      R_co = Eigen::Matrix3f::Identity();
      t_co = Eigen::Vector3f(0, 0, 0);
    }
  }

  string depth_topic;
  string odom_topic;
  float fx;
  float fy;
  float cx;
  float cy;
  int height;
  int width;
  sensor_msgs::CameraInfo cam_info;
  image_geometry::PinholeCameraModel model;

  int use_extrinsic;
  Eigen::Matrix3f R_co;  // rotation from depth to odom
  Eigen::Vector3f t_co;  // translation from depth to odom

 private:
  void packCamInfo() {
    cam_info.height = height;
    cam_info.width = width;
    cam_info.distortion_model = "plumb_bob";

    cam_info.K[0] = fx;
    cam_info.K[2] = cx;
    cam_info.K[4] = fy;
    cam_info.K[5] = cy;
    cam_info.K[8] = 1;

    cam_info.R[0] = 1;
    cam_info.R[4] = 1;
    cam_info.R[8] = 1;

    cam_info.P[0] = fx;
    cam_info.P[2] = cx;
    cam_info.P[5] = fy;
    cam_info.P[6] = cy;
    cam_info.P[10] = 1;
  }
  template <typename T>
  T readParam(ros::NodeHandle &n, std::string name) {
    T ans;
    if (n.getParam(name, ans)) {
      ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    } else {
      ROS_ERROR_STREAM("Failed to load " << name);
      n.shutdown();
    }
    return ans;
  }
};

class D2pNode {
 public:
  D2pNode() : node_handle_("~") {
    params_.readParams(node_handle_);
    sub_odom_ = node_handle_.subscribe<geometry_msgs::PoseStamped>(
        params_.odom_topic, 100, &D2pNode::odomCb, this);
    sub_img_ = node_handle_.subscribe<sensor_msgs::Image>(
        params_.depth_topic, 10, &D2pNode::depthCb, this);
    pub_point_cloud_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
        "pointcloud", 10, true);
    pub_odom_ = node_handle_.advertise<geometry_msgs::TransformStamped>(
        "transform", 10, true);
  }
  void process() {
    while (1) {
      std::unique_lock<std::mutex> lk(buf_lk_);
      pair<cv_bridge::CvImagePtr, geometry_msgs::PoseStamped> img_odom_pair;
      cond_.wait(lk, [&] { return getMeasurements(img_odom_pair); });
      lk.unlock();
      ROS_ERROR("Enter process");
      // convert depth to pointcloud
      sensor_msgs::ImageConstPtr depth_msg = img_odom_pair.first->toImageMsg();
      sensor_msgs::PointCloud2::Ptr cloud_msg;
      convertDepth2Cloud(depth_msg, cloud_msg);

      geometry_msgs::TransformStamped trans_msg;
      trans_msg.header = img_odom_pair.second.header;
      trans_msg.transform.rotation = img_odom_pair.second.pose.orientation;
      trans_msg.transform.translation.x = img_odom_pair.second.pose.position.x;
      trans_msg.transform.translation.y = img_odom_pair.second.pose.position.y;
      trans_msg.transform.translation.z = img_odom_pair.second.pose.position.z;

      pub_odom_.publish(trans_msg);
      pub_point_cloud_.publish(cloud_msg);

      if (!img_buf.empty()) {
        cond_.notify_one();
      }
    }
  }

 private:
  // callback
  void depthCb(const sensor_msgs::Image::ConstPtr &depth_msg) {
    cv_bridge::CvImagePtr ptr(new cv_bridge::CvImage);
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      ptr = cv_bridge::toCvCopy(depth_msg,
                                sensor_msgs::image_encodings::TYPE_16UC1);
    } else if (depth_msg->encoding ==
               sensor_msgs::image_encodings::TYPE_32FC1) {
      ptr = cv_bridge::toCvCopy(depth_msg,
                                sensor_msgs::image_encodings::TYPE_32FC1);
    } else {
      throw "Unsupported image type!";
    }

    buf_lk_.lock();
    img_buf.push(ptr);
    buf_lk_.unlock();
    cond_.notify_one();
  }

  void odomCb(const geometry_msgs::PoseStamped::ConstPtr &odom_msg) {
    buf_lk_.lock();
    odom_buf.push(odom_msg);
    buf_lk_.unlock();
    cond_.notify_one();
  }

  bool getMeasurements(
      pair<cv_bridge::CvImagePtr, geometry_msgs::PoseStamped> &data) {
    if (img_buf.empty() || odom_buf.empty()) {
      return false;
    }
    if (img_buf.front()->header.stamp.toSec() ==
        odom_buf.front()->header.stamp.toSec()) {
      data.first = img_buf.front();
      data.second = *odom_buf.front();
      img_buf.pop();
      odom_buf.pop();
      return true;
    }
    if (img_buf.front()->header.stamp.toSec() >
        odom_buf.back()->header.stamp.toSec()) {
      return false;
    }
    if (img_buf.front()->header.stamp.toSec() <
        odom_buf.front()->header.stamp.toSec()) {
      img_buf.pop();
      return false;
    }
    cv_bridge::CvImagePtr img_ptr = img_buf.front();
    img_buf.pop();
    geometry_msgs::PoseStamped::ConstPtr pose1 = nullptr;
    geometry_msgs::PoseStamped::ConstPtr pose2 = nullptr;
    while (!odom_buf.empty() && odom_buf.front()->header.stamp.toSec() <
                                    img_ptr->header.stamp.toSec()) {
      pose1 = odom_buf.front();
      odom_buf.pop();
    }
    if (!odom_buf.empty()) {
      pose2 = odom_buf.front();
      odom_buf.pop();
    }
    if (pose1 != nullptr && pose2 != nullptr) {
      double t = img_ptr->header.stamp.toSec();
      data.second = poseInterp(pose1, pose2, t);
      ROS_INFO("poseInterp: t1 = %lf, t = %lf, t2 = %lf",
               pose1->header.stamp.toSec(), t, pose2->header.stamp.toSec());
    }
    return true;
  };

  geometry_msgs::PoseStamped poseInterp(
      geometry_msgs::PoseStamped::ConstPtr pose1,
      geometry_msgs::PoseStamped::ConstPtr pose2, double t) {
    double alpha = 0;
    double t1 = pose1->header.stamp.toSec();
    double t2 = pose2->header.stamp.toSec();
    if (t1 < t2) {
      alpha = (t - t1) / (t2 - t1);
    } else {
      ROS_ERROR("Invalid Timestamp of Given Pose.");
    }

    // convert data
    Eigen::Vector3d p1, p2, p;
    Eigen::Quaternion<double> q1, q2, q;
    tf::pointMsgToEigen(pose1->pose.position, p1);
    tf::pointMsgToEigen(pose2->pose.position, p2);
    tf::quaternionMsgToEigen(pose1->pose.orientation, q1);
    tf::quaternionMsgToEigen(pose2->pose.orientation, q2);

    // interpolation
    p = (1.f - alpha) * p1 + alpha * p2;
    q = q1.slerp(alpha, q2);

    // pack data
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = pose1->header.frame_id;
    pose.header.stamp.fromSec(t);
    tf::pointEigenToMsg(p, pose.pose.position);
    tf::quaternionEigenToMsg(q, pose.pose.orientation);

    return pose;
  }

  void convertDepth2Cloud(sensor_msgs::ImageConstPtr &depth_msg,
                          sensor_msgs::PointCloud2::Ptr &cloud_msg) {
    cloud_msg = boost::make_shared<sensor_msgs::PointCloud2>();
    cloud_msg->header = depth_msg->header;
    cloud_msg->height = depth_msg->height;
    cloud_msg->width = depth_msg->width;
    cloud_msg->is_bigendian = false;
    cloud_msg->is_dense = false;
    sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
    pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
    depth_image_proc::convert<uint16_t>(depth_msg, cloud_msg, params_.model);
  }

  Params params_;

  ros::NodeHandle node_handle_;
  ros::Subscriber sub_odom_;
  ros::Subscriber sub_img_;
  ros::Publisher pub_point_cloud_;
  ros::Publisher pub_odom_;

  queue<cv_bridge::CvImagePtr> img_buf;
  queue<geometry_msgs::PoseStamped::ConstPtr> odom_buf;

  std::mutex buf_lk_;
  std::condition_variable cond_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "d2p");
  D2pNode d2p;
  thread process_thread(&D2pNode::process, &d2p);
  while (ros::ok()) {
    ros::spin();
  }

  return 0;
}