import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
import math
import threading

# 3D点云和姿态计算的核心依赖
import open3d as o3d
from scipy.spatial.transform import Rotation

from tf2_ros import TransformBroadcaster

# 导入您的 Service 类型
from custom_msg_srv.srv import GetBestPose

# 用于点云转换的ROS库
from sensor_msgs_py import point_cloud2

class YoloPoseService(Node):
    def __init__(self):
        super().__init__('giim_pose_service_node_3d')

        # --- 缓存最新消息的变量和线程锁 ---
        self._latest_rgb_msg = None
        self._latest_depth_msg = None
        self._lock = threading.Lock()

        self.max_time_diff_ns = 0.05 * 1e9 

        # --- 初始化相机、模型等 ---
        self.camera_intrinsics_matrix = None
        self.o3d_intrinsics = None
        self.intrinsics_set = False

        try:
            yolo_model_path = "../best.pt"
            self.segmentation_model = YOLO(yolo_model_path)
            self.get_logger().info(f"YOLO model loaded successfully from {yolo_model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            return

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        # === 调试参数 ===
        self.declare_parameter('debug_publish_pcd', True)
        self.declare_parameter('debug_save_pcd', False)

        # 以下坐标系基于 camera_color_optical_frame
        # X: 右为正, Y: 下为正, Z: 前为正
        self.declare_parameter('roi.x_min', -1.0) # 左 0.5米
        self.declare_parameter('roi.x_max', 1.0)
        self.declare_parameter('roi.y_min', -0.5) # 上 0.3米
        self.declare_parameter('roi.y_max', 0.1)  # 下 0.5米
        self.declare_parameter('roi.z_min', 0.2)  # 近 0.2米
        self.declare_parameter('roi.z_max', 5.0)  # 远 1.5米

        # === 核心修改 1: 将角度容差参数化 ===
        self.declare_parameter('ransac.table_angle_tolerance_deg', 30.0) # 30deg

        # === 回调组 ===
        self.service_group = ReentrantCallbackGroup()
        self.subscription_group = ReentrantCallbackGroup()

        # --- ROS 通信接口 ---
        self.pose_image_pub = self.create_publisher(Image, "/giim_pose/result_image", 10)
        
        # (调试点云发布者)
        self.pub_pc_scene = self.create_publisher(PointCloud2, "/giim_pose/debug/scene", 10)
        self.pub_pc_object = self.create_publisher(PointCloud2, "/giim_pose/debug/object", 10)
        self.pub_pc_table = self.create_publisher(PointCloud2, "/giim_pose/debug/table", 10)
        self.pub_pc_top = self.create_publisher(PointCloud2, "/giim_pose/debug/top_surface", 10)
        
        self.srv = self.create_service(
            GetBestPose, 'get_best_pose', 
            self._service_callback,
            callback_group=self.service_group)
        
        # (订阅者部分)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/camera/camera/aligned_depth_to_color/camera_info",
            self.camera_info_callback, 10,
            callback_group=self.subscription_group)
        self.rgb_sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw", 
            self._rgb_callback, 1,
            callback_group=self.subscription_group)
        self.depth_sub = self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw", 
            self._depth_callback, 1,
            callback_group=self.subscription_group)
        
        self.get_logger().info("YOLO 3D Pose Service (Open3D, Z-down-fixed) is ready.")

    # ( _o3d_to_ros_pcd, _rgb_callback, _depth_callback, camera_info_callback 保持不变 )
    # ...
    def _o3d_to_ros_pcd(self, o3d_pc, header):
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        points_xyz = np.asarray(o3d_pc.points, dtype=np.float32)
        ros_msg = point_cloud2.create_cloud(header, fields, points_xyz)
        return ros_msg
    def _rgb_callback(self, msg):
        with self._lock: self._latest_rgb_msg = msg
    def _depth_callback(self, msg):
        with self._lock: self._latest_depth_msg = msg
    def camera_info_callback(self, msg):
        if self.intrinsics_set: return
        try:
            self.camera_intrinsics_matrix = np.array(msg.k).reshape(3, 3)
            self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                msg.width, msg.height, msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.intrinsics_set = True
            self.destroy_subscription(self.camera_info_sub)
            self.get_logger().info(f"Camera intrinsics loaded and set for Open3D.")
        except Exception as e:
            self.get_logger().error(f"Failed to parse CameraInfo: {e}")
    # ...

    def _service_callback(self, request, response):
        """
        V11.2 (PCA-based Orientation):
        - Fix: Changed get_covariance() to compute_mean_and_covariance()
        - Position: 3D RANSAC Top Surface Centroid
        - Orientation Z-Axis: 3D RANSAC Top Surface Normal
        - Orientation X-Axis: 3D PCA of Top Surface (Length)
        """
        with self._lock:
            rgb_msg = self._latest_rgb_msg
            depth_msg = self._latest_depth_msg

        # (All checks for data, intrinsics, and timestamp sync remain the same)
        if rgb_msg is None or depth_msg is None:
            response.success = False; response.message = "No image data received."
            return response
        if not self.intrinsics_set:
            response.success = False; response.message = "Camera intrinsics not set."
            return response
        if abs(Time.from_msg(rgb_msg.header.stamp).nanoseconds - 
               Time.from_msg(depth_msg.header.stamp).nanoseconds) > self.max_time_diff_ns:
            response.success = False; response.message = "Data is not synchronized."
            return response
        
        debug_save = self.get_parameter('debug_save_pcd').get_parameter_value().bool_value
        debug_publish = self.get_parameter('debug_publish_pcd').get_parameter_value().bool_value
            
        try:
            # --- Steps 1-3 (Data Prep, Point Cloud, ROI, Table RANSAC) ---
            # (These remain identical to your provided code)
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            depth_image = depth_image_raw.astype(np.uint16)
            results = self.segmentation_model(source=rgb_image, verbose=False)
            result = results[0]
            if result.masks is None or len(result.masks) == 0:
                response.success = False; response.message = "No objects detected."
                return response
            confidences = result.boxes.conf.cpu().numpy()
            if len(confidences) == 0:
                response.success = False; response.message = "No objects detected."
                return response
            best_idx = np.argmax(confidences)
            
            o3d_rgb = o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            o3d_depth = o3d.geometry.Image(depth_image)
            scene_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_rgb, o3d_depth, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
            PC_Scene_Raw = o3d.geometry.PointCloud.create_from_rgbd_image(scene_rgbd, self.o3d_intrinsics)
            PC_Scene_Raw.remove_non_finite_points()

            target_height, target_width = depth_image.shape
            mask_proto = result.masks.data[best_idx].cpu().numpy()
            mask_full_size = cv2.resize(mask_proto, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            mask_bool = (mask_full_size > 0)
            masked_depth_image = depth_image.copy(); masked_depth_image[mask_bool == False] = 0
            o3d_masked_depth = o3d.geometry.Image(masked_depth_image)
            obj_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_rgb, o3d_masked_depth, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
            PC_Object = o3d.geometry.PointCloud.create_from_rgbd_image(obj_rgbd, self.o3d_intrinsics)
            PC_Object.remove_non_finite_points()
            
            if not PC_Object.has_points():
                response.success = False; response.message = "Object mask has no valid depth."
                return response
            
            if debug_save: o3d.io.write_point_cloud("debug_object_raw.pcd", PC_Object)
            if debug_publish: self.pub_pc_object.publish(self._o3d_to_ros_pcd(PC_Object, rgb_msg.header))

            min_x = self.get_parameter('roi.x_min').get_parameter_value().double_value
            max_x = self.get_parameter('roi.x_max').get_parameter_value().double_value
            min_y = self.get_parameter('roi.y_min').get_parameter_value().double_value
            max_y = self.get_parameter('roi.y_max').get_parameter_value().double_value
            min_z = self.get_parameter('roi.z_min').get_parameter_value().double_value
            max_z = self.get_parameter('roi.z_max').get_parameter_value().double_value
            min_bound = np.array([min_x, min_y, min_z]); max_bound = np.array([max_x, max_y, max_z])
            roi_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            PC_Scene = PC_Scene_Raw.crop(roi_box)
            
            if not PC_Scene.has_points():
                response.success = False; response.message = "No points left in Scene after ROI cropping."
                return response
            
            if debug_save: o3d.io.write_point_cloud("debug_scene_cropped.pcd", PC_Scene)

            PC_Scene_Down = PC_Scene.voxel_down_sample(voxel_size=0.01)
            if debug_publish: self.pub_pc_scene.publish(self._o3d_to_ros_pcd(PC_Scene_Down, rgb_msg.header))

            PC_Scene_Down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            
            table_angle_tolerance_deg = self.get_parameter('ransac.table_angle_tolerance_deg').get_parameter_value().double_value
            angle_tolerance = np.deg2rad(table_angle_tolerance_deg)
            expected_normal = np.array([0, 0, -1]) # camera_rgb_optical_frame inverse Z: out of screen
            
            # 历场景中的每一个点，检查它的法线朝向。如果这个朝向与我定义的‘完美水平面’（[0, 0, -1]）的夹角在15度以内（无论是朝上还是朝下），就把这个点的索引（index）告诉我
            scene_normals = np.asarray(PC_Scene_Down.normals)
            dot_products = np.dot(scene_normals, expected_normal)
            horizontal_indices = np.where(np.abs(dot_products) > np.cos(angle_tolerance))[0] # 找出所有点积的绝对值大于0.966的点”。换句话说，\
            # 就是找出所有法线与“完美水平面”（expected_normal）的夹角在 0° 到 15° 之间，或者在 165° 到 180° 之间的点
            
            if len(horizontal_indices) == 0:
                response.success = False; response.message = "No horizontal surfaces (Z-axis) found in ROI."
                return response
                
            PC_Horizontal_Surfaces = PC_Scene_Down.select_by_index(horizontal_indices)
            table_plane_model, table_inliers = PC_Horizontal_Surfaces.segment_plane(
                distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            PC_Table = PC_Horizontal_Surfaces.select_by_index(table_inliers)
            
            if debug_save: o3d.io.write_point_cloud("debug_table_plane.pcd", PC_Table)
            if debug_publish: self.pub_pc_table.publish(self._o3d_to_ros_pcd(PC_Table, rgb_msg.header))

            N_Table = table_plane_model[:3];
            if N_Table[2] > 0: 
                N_Table = -N_Table

            # --- Step 4: RANSAC for Top Surface (Store Top Normal) ---
            PC_Top_Surface = None
            N_Top = None # <--- Initialize Top Normal here
            try:
                PC_Object.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
                obj_normals = np.asarray(PC_Object.normals)
                obj_dot_products = np.dot(obj_normals, N_Table)
                top_indices = np.where(np.abs(obj_dot_products) > np.cos(np.deg2rad(20)))[0]
                if len(top_indices) > 0:
                    PC_Top_Candidates = PC_Object.select_by_index(top_indices)
                    top_plane_model, top_inliers = PC_Top_Candidates.segment_plane(
                        distance_threshold=0.005, ransac_n=3, num_iterations=500)
                    PC_Top_Surface = PC_Top_Candidates.select_by_index(top_inliers)
                    N_Top = top_plane_model[:3] # <--- Store the top plane's normal
            except Exception: pass

            # --- Step 5: Compute Position and Fallback ---
            if PC_Top_Surface is None or len(PC_Top_Surface.points) < 20:
                self.get_logger().warn("RANSAC for top surface failed, falling back to 'highest points'.")
                table_d = table_plane_model[3]
                obj_points = np.asarray(PC_Object.points)
                distances_to_table = np.dot(obj_points, N_Table) + table_d
                max_height_dist = np.min(distances_to_table)
                top_indices = np.where(distances_to_table < max_height_dist + 0.005)[0]
                PC_Top_Surface = PC_Object.select_by_index(top_indices)
                N_Top = N_Table # Fallback: Top normal is the same as table normal
            
            if not PC_Top_Surface.has_points():
                response.success = False; response.message = "Could not isolate top surface."
                return response

            if debug_save: o3d.io.write_point_cloud("debug_top_surface.pcd", PC_Top_Surface)
            if debug_publish: self.pub_pc_top.publish(self._o3d_to_ros_pcd(PC_Top_Surface, rgb_msg.header))

            
            # === [CORE MODIFICATION] Step 5: Compute Final Pose ===

            # 5a. Position AND Covariance: Use compute_mean_and_covariance()
            # === [THE FIX] ===
            final_position, cov_matrix = PC_Top_Surface.compute_mean_and_covariance()
            final_X, final_Y, final_Z = final_position
            # === [END FIX] ===
            
            # 5b. Orientation
            # 1. Define Z-Axis (Grasp Axis) from the Top Normal
            if N_Top[2] > 0:
                N_Top = -N_Top
            Z_axis = -N_Top # Grasp direction is *into* the top surface
            Z_axis = Z_axis / np.linalg.norm(Z_axis)

            # 2. Define X-Axis (Length Axis) using PCA on the top surface
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            X_vec_candidate = eigenvectors[:, np.argmax(eigenvalues)]

            # 3. Build an orthonormal basis (Gram-Schmidt)
            X_axis = X_vec_candidate - np.dot(X_vec_candidate, Z_axis) * Z_axis
            X_axis_norm = np.linalg.norm(X_axis)
            
            if X_axis_norm < 1e-6:
                self.get_logger().warn("X-axis (length) is parallel to Z-axis (normal). Using arbitrary X.")
                arbitrary_X = np.array([1.0, 0.0, 0.0])
                X_axis = arbitrary_X - np.dot(arbitrary_X, Z_axis) * Z_axis
                X_axis_norm = np.linalg.norm(X_axis)

            X_axis = X_axis / X_axis_norm
            
            # 4. Define Y-Axis
            Y_axis = np.cross(Z_axis, X_axis)

            # 5. Create Rotation Matrix and convert to Quaternion
            rot_matrix = np.array([X_axis, Y_axis, Z_axis]).T 
            final_orientation = Rotation.from_matrix(rot_matrix).as_quat()
            final_x, final_y, final_z, final_w = final_orientation
            # === [MODIFICATION END] ===

            
            # --- Step 6: Publish and Respond (Unchanged) ---
            class_name = result.names[int(result.boxes[best_idx].cls.cpu().numpy()[0])]
            
            t = TransformStamped()
            t.header.stamp = rgb_msg.header.stamp; t.header.frame_id = rgb_msg.header.frame_id
            t.child_frame_id = f"target_{class_name}_3D_best"
            
            t.transform.translation.x = final_X; t.transform.translation.y = final_Y; t.transform.translation.z = final_Z
            t.transform.rotation.x = final_x; t.transform.rotation.y = final_y; t.transform.rotation.z = final_z; t.transform.rotation.w = final_w
            self.tf_broadcaster.sendTransform(t)

            # (Visualization logic remains the same)
            annotated_img = rgb_image.copy()
            box = result.boxes[best_idx]
            xyxy = box.xyxy.cpu().numpy()[0].astype(int)
            mask_contour = result.masks.xy[best_idx].astype(np.int32)
            cv2.drawContours(annotated_img, [mask_contour], -1, (255, 255, 0), 2) # Cyan
            conf = float(box.conf.cpu().numpy()[0])
            label_text = f"{class_name} {conf:.2f}"
            cv2.putText(annotated_img, label_text, (xyxy[0], xyxy[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            K = self.camera_intrinsics_matrix
            proj_point_3d = (K @ final_position).flatten()
            u = int(proj_point_3d[0] / proj_point_3d[2]); v = int(proj_point_3d[1] / proj_point_3d[2])
            cv2.drawMarker(annotated_img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 40, 4)
            
            img_msg = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8"); img_msg.header = rgb_msg.header
            self.pose_image_pub.publish(img_msg)

            response.success = True; response.message = f"Successfully found 3D pose for: {class_name}"
            response.pose.header = rgb_msg.header
            response.pose.pose.position.x = final_X; response.pose.pose.position.y = final_Y; response.pose.pose.position.z = final_Z
            response.pose.pose.orientation.x = final_x; response.pose.pose.orientation.y = final_y; response.pose.pose.orientation.z = final_z; response.pose.pose.orientation.w = final_w
            
            return response

        except Exception as e:
            self.get_logger().error(f"An error occurred during 3D processing: {e}", exc_info=True)
            response.success = False
            response.message = str(e)
            return response


# === 启动多线程执行器 (不变) ===
def main(args=None):
    rclpy.init(args=args)
    yolo_pose_service_node = YoloPoseService()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(yolo_pose_service_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        yolo_pose_service_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()